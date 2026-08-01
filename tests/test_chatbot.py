import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import chatbot


class TestEnvParsing(unittest.TestCase):
    def test_load_chat_config_uses_defaults_for_missing_values(self):
        with patch.dict("os.environ", {}, clear=True):
            config = chatbot.load_chat_config()

        self.assertEqual(config.model_name, chatbot.DEFAULT_MODEL_NAME)
        self.assertEqual(config.temperature, chatbot.DEFAULT_TEMPERATURE)
        self.assertEqual(config.max_output_tokens, chatbot.DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(config.system_instruction, chatbot.SYSTEM_INSTRUCTION)

    def test_load_chat_config_reads_valid_env_values(self):
        env = {
            "GEMINI_MODEL": "gemini-test",
            "GEMINI_TEMPERATURE": "1.25",
            "GEMINI_MAX_OUTPUT_TOKENS": "2048",
        }

        with patch.dict("os.environ", env, clear=True):
            config = chatbot.load_chat_config()

        self.assertEqual(config.model_name, "gemini-test")
        self.assertEqual(config.temperature, 1.25)
        self.assertEqual(config.max_output_tokens, 2048)

    def test_load_chat_config_falls_back_for_invalid_env_values(self):
        env = {
            "GEMINI_MODEL": "   ",
            "GEMINI_TEMPERATURE": "hot",
            "GEMINI_MAX_OUTPUT_TOKENS": "0",
        }

        with patch.dict("os.environ", env, clear=True):
            config = chatbot.load_chat_config()

        self.assertEqual(config.model_name, chatbot.DEFAULT_MODEL_NAME)
        self.assertEqual(config.temperature, chatbot.DEFAULT_TEMPERATURE)
        self.assertEqual(config.max_output_tokens, chatbot.DEFAULT_MAX_OUTPUT_TOKENS)


class TestEmptyResponses(unittest.TestCase):
    def test_empty_response_message_reports_block_reason(self):
        response = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
            candidates=[],
        )

        message = chatbot.get_empty_response_message(response)

        self.assertEqual(message, "Bot: The request was blocked: SAFETY.")

    def test_empty_response_message_reports_finish_reasons(self):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(finish_reason="MAX_TOKENS"),
                SimpleNamespace(finish_reason="STOP"),
            ]
        )

        message = chatbot.get_empty_response_message(response)

        self.assertEqual(
            message,
            "Bot: I could not produce text. Finish reason: MAX_TOKENS, STOP.",
        )

    def test_get_response_text_handles_missing_or_invalid_text(self):
        class ResponseWithValueError:
            @property
            def text(self):
                raise ValueError("no text")

        self.assertEqual(chatbot.get_response_text(SimpleNamespace(text=" hello ")), "hello")
        self.assertEqual(chatbot.get_response_text(SimpleNamespace(text=None)), "")
        self.assertEqual(chatbot.get_response_text(ResponseWithValueError()), "")


class TestRetryBehavior(unittest.TestCase):
    def test_send_message_returns_first_successful_response(self):
        chat = Mock()
        chat.send_message.return_value = "ok"

        response = chatbot.send_message_with_retry(chat, "hello")

        self.assertEqual(response, "ok")
        chat.send_message.assert_called_once_with("hello")

    def test_send_message_retries_transient_errors(self):
        transient_error = type("TransientError", (Exception,), {})
        chat = Mock()
        chat.send_message.side_effect = [transient_error("try again"), "ok"]

        with patch.object(chatbot, "TRANSIENT_GEMINI_ERRORS", (transient_error,)):
            with patch.object(chatbot.time, "sleep") as sleep:
                response = chatbot.send_message_with_retry(chat, "hello")

        self.assertEqual(response, "ok")
        self.assertEqual(chat.send_message.call_args_list, [call("hello"), call("hello")])
        sleep.assert_called_once_with(chatbot.MESSAGE_RETRY_DELAY_SECONDS)

    def test_send_message_does_not_retry_non_transient_errors(self):
        chat = Mock()
        chat.send_message.side_effect = ValueError("bad request")

        with self.assertRaises(ValueError):
            chatbot.send_message_with_retry(chat, "hello")

        chat.send_message.assert_called_once_with("hello")


class TestCommandHandling(unittest.TestCase):
    def test_help_command_prints_available_commands(self):
        session = Mock()

        with patch("builtins.print") as mocked_print:
            returned_session = chatbot.handle_help(session)

        self.assertIs(returned_session, session)
        mocked_print.assert_any_call("Commands:")
        mocked_print.assert_any_call("  /clear - Clear the current chat history")

    def test_clear_command_resets_chat_history(self):
        model = Mock()
        new_chat = Mock()
        model.start_chat.return_value = new_chat
        session = chatbot.ChatSession(model=model, chat=Mock(), model_name="gemini-test")

        with patch("builtins.print") as mocked_print:
            returned_session = chatbot.handle_clear(session)

        self.assertIs(returned_session, session)
        self.assertIs(session.chat, new_chat)
        model.start_chat.assert_called_once_with(history=[])
        mocked_print.assert_called_once_with("Chat history cleared.")

    def test_model_command_prints_current_model(self):
        session = chatbot.ChatSession(model=Mock(), chat=Mock(), model_name="gemini-test")

        with patch("builtins.print") as mocked_print:
            returned_session = chatbot.handle_model(session)

        self.assertIs(returned_session, session)
        mocked_print.assert_called_once_with("Current model: gemini-test")

    def test_run_chat_loop_dispatches_commands_and_exit(self):
        session = Mock()

        with patch("builtins.input", side_effect=["/help", "bye"]):
            with patch("builtins.print") as mocked_print:
                chatbot.run_chat_loop(session)

        mocked_print.assert_any_call("Commands:")
        mocked_print.assert_any_call("Goodbye.")


class TestCreateChat(unittest.TestCase):
    def test_create_chat_uses_supplied_config(self):
        config = chatbot.ChatConfig(
            model_name="gemini-test",
            temperature=0.2,
            max_output_tokens=512,
            system_instruction="Be brief.",
        )
        model = Mock()
        chat = Mock()
        model.start_chat.return_value = chat

        with patch.object(chatbot.genai, "configure") as configure:
            with patch.object(chatbot.genai, "GenerativeModel", return_value=model) as model_cls:
                session = chatbot.create_chat("api-key", config)

        configure.assert_called_once_with(api_key="api-key")
        model_cls.assert_called_once_with(
            "gemini-test",
            system_instruction="Be brief.",
            generation_config={"temperature": 0.2, "max_output_tokens": 512},
        )
        model.start_chat.assert_called_once_with(history=[])
        self.assertEqual(session.chat, chat)
        self.assertEqual(session.model_name, "gemini-test")


if __name__ == "__main__":
    unittest.main()

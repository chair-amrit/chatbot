import os
import logging
from dataclasses import dataclass
from typing import Any, Callable
from dotenv import load_dotenv
import google.generativeai as genai

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024
MIN_MAX_OUTPUT_TOKENS = 1
MAX_MAX_OUTPUT_TOKENS = 65536
SYSTEM_INSTRUCTION = (
    "You are a helpful, concise chatbot. Answer clearly and ask clarifying "
    "questions when the user's request is ambiguous."
)
EXIT_COMMANDS = {"bye", "exit", "quit"}
GEMINI_API_ERRORS = (google_exceptions.GoogleAPIError,) if google_exceptions else None


@dataclass
class ChatSession:
    model: Any
    chat: Any
    model_name: str


CommandHandler = Callable[[ChatSession], ChatSession]


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY is missing.")
        print("Add it to .env in this project folder, for example:")
        print("GEMINI_API_KEY=your_api_key_here")
        raise SystemExit(1)

    return api_key


def get_float_env(
    name: str,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed_value = float(value)
    except ValueError:
        logging.warning("Invalid %s value %r. Using default: %s", name, value, default)
        return default

    if not min_value <= parsed_value <= max_value:
        logging.warning(
            "%s value %r is outside %s-%s. Using default: %s",
            name,
            value,
            min_value,
            max_value,
            default,
        )
        return default

    return parsed_value


def get_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed_value = int(value)
    except ValueError:
        logging.warning("Invalid %s value %r. Using default: %s", name, value, default)
        return default

    if not min_value <= parsed_value <= max_value:
        logging.warning(
            "%s value %r is outside %s-%s. Using default: %s",
            name,
            value,
            min_value,
            max_value,
            default,
        )
        return default

    return parsed_value


def get_str_env(name: str, default: str) -> str:
    value = os.getenv(name)

    if value is None:
        return default

    stripped_value = value.strip()
    if not stripped_value:
        logging.warning("Invalid %s value %r. Using default: %s", name, value, default)
        return default

    return stripped_value


def get_empty_response_message(response: Any) -> str:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)

    if block_reason:
        return f"Bot: The request was blocked: {block_reason}."

    candidates = getattr(response, "candidates", None) or []
    finish_reasons = [
        str(finish_reason)
        for candidate in candidates
        if (finish_reason := getattr(candidate, "finish_reason", None)) is not None
    ]

    if finish_reasons:
        return f"Bot: I could not produce text. Finish reason: {', '.join(finish_reasons)}."

    return "Bot: I did not receive a valid response. Please try again."


def get_response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", "")
    except ValueError:
        return ""

    if not isinstance(text, str):
        return ""

    return text.strip()


def create_chat(api_key: str) -> ChatSession:
    genai.configure(api_key=api_key)

    model_name = get_str_env("GEMINI_MODEL", DEFAULT_MODEL_NAME)
    generation_config = {
        "temperature": get_float_env(
            "GEMINI_TEMPERATURE",
            DEFAULT_TEMPERATURE,
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
        ),
        "max_output_tokens": get_int_env(
            "GEMINI_MAX_OUTPUT_TOKENS",
            DEFAULT_MAX_OUTPUT_TOKENS,
            MIN_MAX_OUTPUT_TOKENS,
            MAX_MAX_OUTPUT_TOKENS,
        ),
    }
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=generation_config,
    )

    return ChatSession(
        model=model,
        chat=model.start_chat(history=[]),
        model_name=model_name,
    )


def show_help() -> None:
    print("Commands:")
    print("  /help  - Show available commands")
    print("  /clear - Clear the current chat history")
    print("  /reset - Clear the current chat history")
    print("  /model - Show the current model")
    print("  bye, exit, quit - Stop the chatbot")


def print_bot_reply(reply: str) -> None:
    print()
    print("Bot:")
    print(reply)
    print()


def handle_help(session: ChatSession) -> ChatSession:
    show_help()
    return session


def handle_clear(session: ChatSession) -> ChatSession:
    print("Chat history cleared.")
    session.chat = session.model.start_chat(history=[])
    return session


def handle_model(session: ChatSession) -> ChatSession:
    print(f"Current model: {session.model_name}")
    return session


def get_command_handlers() -> dict[str, CommandHandler]:
    return {
        "/help": handle_help,
        "/clear": handle_clear,
        "/reset": handle_clear,
        "/model": handle_model,
    }


def run_chat_loop(session: ChatSession) -> None:
    print("Chatbot ready. Type '/help' for commands or 'bye', 'exit', or 'quit' to stop.")
    command_handlers = get_command_handlers()

    try:
        while True:
            user_input = input("You: ")
            user_input = user_input.strip()
            normalized_input = user_input.lower()

            if normalized_input in EXIT_COMMANDS:
                print("Goodbye.")
                break

            if not user_input:
                continue

            command_handler = command_handlers.get(normalized_input)
            if command_handler is not None:
                session = command_handler(session)
                continue

            if normalized_input.startswith("/"):
                print("Unknown command. Type /help for available commands.")
                continue

            try:
                response = session.chat.send_message(user_input)

                reply = get_response_text(response)

                if not reply:
                    print(get_empty_response_message(response))
                    continue

                print_bot_reply(reply)

            except Exception as error:
                if GEMINI_API_ERRORS is not None and isinstance(error, GEMINI_API_ERRORS):
                    logging.exception("Gemini request failed")
                    print("Bot: Sorry, the Gemini API request failed. Please try again.")
                    continue

                raise
    except KeyboardInterrupt:
        print("\nExiting chatbot.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    api_key = load_api_key()
    session = create_chat(api_key)
    run_chat_loop(session)


if __name__ == "__main__":
    main()

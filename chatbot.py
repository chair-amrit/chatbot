import os
import logging
import random
import re
import time
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
MESSAGE_RETRY_ATTEMPTS = 3
MESSAGE_RETRY_DELAY_SECONDS = 1.0
MESSAGE_RETRY_JITTER_RATIO = 0.25
MIN_MESSAGE_RETRY_ATTEMPTS = 1
MAX_MESSAGE_RETRY_ATTEMPTS = 10
MIN_MESSAGE_RETRY_DELAY_SECONDS = 0.0
MAX_MESSAGE_RETRY_DELAY_SECONDS = 60.0
MODEL_NAME_PATTERN = re.compile(r"^(?:models/)?[A-Za-z0-9][A-Za-z0-9._-]*$")
GEMINI_API_ERRORS = (google_exceptions.GoogleAPIError,) if google_exceptions else None
NETWORK_SEND_ERRORS = (TimeoutError, ConnectionError, OSError)
SEND_MESSAGE_ERRORS = (
    (google_exceptions.GoogleAPIError,) if google_exceptions else ()
) + NETWORK_SEND_ERRORS

if google_exceptions:
    TRANSIENT_GEMINI_ERRORS = tuple(
        error_type
        for name in (
            "ResourceExhausted",
            "TooManyRequests",
            "ServiceUnavailable",
            "DeadlineExceeded",
            "GatewayTimeout",
            "InternalServerError",
        )
        if (error_type := getattr(google_exceptions, name, None)) is not None
    )
else:
    TRANSIENT_GEMINI_ERRORS = ()


class MissingApiKeyError(Exception):
    """Raised when GEMINI_API_KEY is unavailable."""


class InvalidModelNameError(ValueError):
    """Raised when GEMINI_MODEL contains an unsupported model name format."""


@dataclass
class ChatSession:
    model: Any
    chat: Any
    model_name: str
    retry_attempts: int = MESSAGE_RETRY_ATTEMPTS
    retry_delay_seconds: float = MESSAGE_RETRY_DELAY_SECONDS


@dataclass(frozen=True)
class ChatConfig:
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    retry_attempts: int = MESSAGE_RETRY_ATTEMPTS
    retry_delay_seconds: float = MESSAGE_RETRY_DELAY_SECONDS
    system_instruction: str = SYSTEM_INSTRUCTION


InputFunc = Callable[[str], str]
PrintFunc = Callable[..., None]
SendMessageFunc = Callable[[ChatSession, str], Any]
CommandHandler = Callable[[ChatSession, PrintFunc], ChatSession]


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise MissingApiKeyError("GEMINI_API_KEY is missing.")

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


def validate_model_name(model_name: str) -> str:
    if not MODEL_NAME_PATTERN.fullmatch(model_name):
        raise InvalidModelNameError(
            "GEMINI_MODEL must contain only letters, numbers, dots, underscores, "
            "or hyphens, with an optional 'models/' prefix."
        )

    return model_name


def load_chat_config() -> ChatConfig:
    return ChatConfig(
        model_name=validate_model_name(get_str_env("GEMINI_MODEL", DEFAULT_MODEL_NAME)),
        temperature=get_float_env(
            "GEMINI_TEMPERATURE",
            DEFAULT_TEMPERATURE,
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
        ),
        max_output_tokens=get_int_env(
            "GEMINI_MAX_OUTPUT_TOKENS",
            DEFAULT_MAX_OUTPUT_TOKENS,
            MIN_MAX_OUTPUT_TOKENS,
            MAX_MAX_OUTPUT_TOKENS,
        ),
        retry_attempts=get_int_env(
            "GEMINI_RETRY_ATTEMPTS",
            MESSAGE_RETRY_ATTEMPTS,
            MIN_MESSAGE_RETRY_ATTEMPTS,
            MAX_MESSAGE_RETRY_ATTEMPTS,
        ),
        retry_delay_seconds=get_float_env(
            "GEMINI_RETRY_DELAY_SECONDS",
            MESSAGE_RETRY_DELAY_SECONDS,
            MIN_MESSAGE_RETRY_DELAY_SECONDS,
            MAX_MESSAGE_RETRY_DELAY_SECONDS,
        ),
    )


def get_empty_response_message(response: Any) -> str:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)

    if block_reason:
        return f"The request was blocked: {block_reason}."

    candidates = getattr(response, "candidates", None) or []
    finish_reasons = [
        str(finish_reason)
        for candidate in candidates
        if (finish_reason := getattr(candidate, "finish_reason", None)) is not None
    ]

    if finish_reasons:
        return f"I could not produce text. Finish reason: {', '.join(finish_reasons)}."

    return "I did not receive a valid response. Please try again."


def get_response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", "")
    except ValueError:
        return ""

    if not isinstance(text, str):
        return ""

    return text.strip()


def is_transient_send_error(error: BaseException) -> bool:
    return isinstance(error, NETWORK_SEND_ERRORS) or (
        bool(TRANSIENT_GEMINI_ERRORS) and isinstance(error, TRANSIENT_GEMINI_ERRORS)
    )


def get_retry_sleep_seconds(delay_seconds: float) -> float:
    jitter_seconds = random.uniform(0, delay_seconds * MESSAGE_RETRY_JITTER_RATIO)
    return delay_seconds + jitter_seconds


def send_message_with_retry(
    chat: Any,
    user_input: str,
    retry_attempts: int = MESSAGE_RETRY_ATTEMPTS,
    retry_delay_seconds: float = MESSAGE_RETRY_DELAY_SECONDS,
) -> Any:
    delay_seconds = retry_delay_seconds

    for attempt in range(1, retry_attempts + 1):
        try:
            return chat.send_message(user_input)
        except SEND_MESSAGE_ERRORS as error:
            if not is_transient_send_error(error):
                raise

            if attempt == retry_attempts:
                raise

            sleep_seconds = get_retry_sleep_seconds(delay_seconds)
            logging.warning(
                "Gemini request failed transiently. Retrying in %.1f seconds.",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            delay_seconds *= 2

    raise RuntimeError("Retry loop ended without a response.")


def send_session_message(session: ChatSession, user_input: str) -> Any:
    return send_message_with_retry(
        session.chat,
        user_input,
        retry_attempts=session.retry_attempts,
        retry_delay_seconds=session.retry_delay_seconds,
    )


def create_chat(api_key: str, config: ChatConfig | None = None) -> ChatSession:
    genai.configure(api_key=api_key)

    if config is None:
        config = load_chat_config()

    model_name = validate_model_name(config.model_name)
    generation_config = {
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }
    model = genai.GenerativeModel(
        model_name,
        system_instruction=config.system_instruction,
        generation_config=generation_config,
    )

    return ChatSession(
        model=model,
        chat=model.start_chat(history=[]),
        model_name=model_name,
        retry_attempts=config.retry_attempts,
        retry_delay_seconds=config.retry_delay_seconds,
    )


def show_help(print_func: PrintFunc = print) -> None:
    print_func("Commands:")
    print_func("  /help  - Show available commands")
    print_func("  /clear - Clear the current chat history")
    print_func("  /reset - Clear the current chat history")
    print_func("  /model - Show the current model")
    print_func("  bye, exit, quit - Stop the chatbot")


def print_bot_reply(reply: str, print_func: PrintFunc = print) -> None:
    print_func()
    print_func("Bot:")
    print_func(reply)
    print_func()


def print_bot_message(message: str, print_func: PrintFunc = print) -> None:
    print_func(f"Bot: {message}")


def handle_help(session: ChatSession, print_func: PrintFunc = print) -> ChatSession:
    show_help(print_func)
    return session


def handle_clear(session: ChatSession, print_func: PrintFunc = print) -> ChatSession:
    print_func("Chat history cleared.")
    session.chat = session.model.start_chat(history=[])
    return session


def handle_model(session: ChatSession, print_func: PrintFunc = print) -> ChatSession:
    print_func(f"Current model: {session.model_name}")
    return session


def get_command_handlers() -> dict[str, CommandHandler]:
    return {
        "/help": handle_help,
        "/clear": handle_clear,
        "/reset": handle_clear,
        "/model": handle_model,
    }


def run_chat_loop(
    session: ChatSession,
    input_func: InputFunc | None = None,
    print_func: PrintFunc = print,
    send_message_func: SendMessageFunc = send_session_message,
) -> None:
    if input_func is None:
        input_func = input

    print_func("Chatbot ready. Type '/help' for commands or 'bye', 'exit', or 'quit' to stop.")
    command_handlers = get_command_handlers()

    try:
        while True:
            user_input = input_func("You: ")
            user_input = user_input.strip()
            normalized_input = user_input.lower()

            if normalized_input in EXIT_COMMANDS:
                print_func("Goodbye.")
                break

            if not user_input:
                continue

            command_handler = command_handlers.get(normalized_input)
            if command_handler is not None:
                session = command_handler(session, print_func)
                continue

            if normalized_input.startswith("/"):
                print_func("Unknown command. Type /help for available commands.")
                continue

            try:
                response = send_message_func(session, user_input)

                reply = get_response_text(response)

                if not reply:
                    print_bot_message(get_empty_response_message(response), print_func)
                    continue

                print_bot_reply(reply, print_func)

            except Exception as error:
                if GEMINI_API_ERRORS is not None and isinstance(error, GEMINI_API_ERRORS):
                    logging.exception("Gemini request failed")
                    print_bot_message(
                        "Sorry, the Gemini API request failed. Please try again.",
                        print_func,
                    )
                    continue

                raise
    except KeyboardInterrupt:
        print_func("\nExiting chatbot.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        api_key = load_api_key()
        session = create_chat(api_key)
    except MissingApiKeyError as error:
        print(f"Error: {error}")
        print("Add it to .env in this project folder, for example:")
        print("GEMINI_API_KEY=your_api_key_here")
        raise SystemExit(1)
    except InvalidModelNameError as error:
        print(f"Error: {error}")
        raise SystemExit(1)

    run_chat_loop(session)


if __name__ == "__main__":
    main()

import os
import logging
from typing import Any
from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 1024
SYSTEM_INSTRUCTION = (
    "You are a helpful, concise chatbot. Answer clearly and ask clarifying "
    "questions when the user's request is ambiguous."
)
EXIT_COMMANDS = {"bye", "exit", "quit"}
SLASH_COMMANDS = {"/help", "/clear", "/model"}


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY is missing. Add it to your .env file.")
        raise SystemExit(1)

    return api_key


def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        logging.warning("Invalid %s value %r. Using default: %s", name, value, default)
        return default


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid %s value %r. Using default: %s", name, value, default)
        return default


def create_chat(api_key: str) -> tuple[Any, Any, str]:
    genai.configure(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)
    generation_config = {
        "temperature": get_float_env("GEMINI_TEMPERATURE", DEFAULT_TEMPERATURE),
        "max_output_tokens": get_int_env(
            "GEMINI_MAX_OUTPUT_TOKENS",
            DEFAULT_MAX_OUTPUT_TOKENS,
        ),
    }
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=generation_config,
    )

    return model, model.start_chat(history=[]), model_name


def show_help() -> None:
    print("Commands:")
    print("  /help  - Show available commands")
    print("  /clear - Clear the current chat history")
    print("  /model - Show the current model")
    print("  bye, exit, quit - Stop the chatbot")


def run_chat_loop(model: Any, chat: Any, model_name: str) -> None:
    print("Chatbot ready. Type '/help' for commands or 'bye', 'exit', or 'quit' to stop.")

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

            if normalized_input in SLASH_COMMANDS:
                if normalized_input == "/help":
                    show_help()
                elif normalized_input == "/clear":
                    chat = model.start_chat(history=[])
                    print("Chat history cleared.")
                elif normalized_input == "/model":
                    print(f"Current model: {model_name}")
                continue

            if normalized_input.startswith("/"):
                print("Unknown command. Type /help for available commands.")
                continue

            try:
                response = chat.send_message(user_input)

                reply = getattr(response, "text", "").strip()

                if not reply:
                    print("Bot: I did not receive a valid response. Please try again.")
                    continue

                print("Bot:", reply)

            except Exception:
                logging.exception("Gemini request failed")
                print("Bot: Sorry, something went wrong. Please try again.")
    except KeyboardInterrupt:
        print("\nExiting chatbot.")


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    api_key = load_api_key()
    model, chat, model_name = create_chat(api_key)
    run_chat_loop(model, chat, model_name)


if __name__ == "__main__":
    main()
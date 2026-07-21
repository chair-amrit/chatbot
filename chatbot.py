import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are a helpful, concise chatbot. Answer clearly and ask clarifying "
    "questions when the user's request is ambiguous."
)
EXIT_COMMANDS = {"bye", "exit", "quit"}
SLASH_COMMANDS = {"/help", "/clear", "/model"}


def load_api_key():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY is missing. Add it to your .env file.")
        raise SystemExit(1)

    return api_key


def create_chat(api_key):
    genai.configure(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    return model, model.start_chat(history=[]), model_name


def show_help():
    print("Commands:")
    print("  /help  - Show available commands")
    print("  /clear - Clear the current chat history")
    print("  /model - Show the current model")
    print("  bye, exit, quit - Stop the chatbot")


def run_chat_loop(model, chat, model_name):
    print("Chatbot ready. Type '/help' for commands or 'bye', 'exit', or 'quit' to stop.")

    try:
        while True:
            user_input = input("You: ")
            user_input = user_input.strip()

            if user_input.lower() in EXIT_COMMANDS:
                print("Goodbye.")
                break

            if not user_input:
                continue

            if user_input.lower() in SLASH_COMMANDS:
                if user_input.lower() == "/help":
                    show_help()
                elif user_input.lower() == "/clear":
                    chat = model.start_chat(history=[])
                    print("Chat history cleared.")
                elif user_input.lower() == "/model":
                    print(f"Current model: {model_name}")
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


def main():
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    api_key = load_api_key()
    model, chat, model_name = create_chat(api_key)
    run_chat_loop(model, chat, model_name)


if __name__ == "__main__":
    main()

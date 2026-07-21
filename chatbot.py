import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai


MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are a helpful, concise chatbot. Answer clearly and ask clarifying "
    "questions when the user's request is ambiguous."
)
EXIT_COMMANDS = {"bye", "exit", "quit"}


def main():
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY is missing. Add it to your .env file.")
        raise SystemExit(1)

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat(history=[])

    print("Chatbot ready. Type 'bye', 'exit', or 'quit' to stop.")

    try:
        while True:
            user_input = input("You:")
            user_input = user_input.strip()

            if user_input.lower() in EXIT_COMMANDS:
                break

            if not user_input:
                continue

            try:
                response = chat.send_message(user_input)

                reply = getattr(response, "text", "").strip()

                if not reply:
                    print("Bot: I did not receive a valid response. Please try again.")
                    continue

                print("Bot:", reply)

            except Exception as e:
                logging.exception("Gemini request failed")
                print("Bot: Sorry, something went wrong. Please try again.")
    except KeyboardInterrupt:
        print("\nExiting chatbot.")


if __name__ == "__main__":
    main()

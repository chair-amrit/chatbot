import os
from dotenv import load_dotenv
import google.generativeai as genai


def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")

    if not api_key:
        print("Error: API_KEY is missing. Add it to your .env file.")
        raise SystemExit(1)

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")
    chat = model.start_chat(history=[])

    while True:
        user_input = input("You:")

        if user_input.lower() == "bye":
            break

        try:
            response = chat.send_message(user_input)

            reply = response.text

            print("Bot:", reply)

        except Exception as e:
            print(f"Error:{e}")


if __name__ == "__main__":
    main()

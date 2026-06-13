import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

model=genai.GenerativeModel("gemini-2.5-flash")


while True:
    user_input=input("You:")

    if user_input.lower()=="bye":
        break

    try:
        response=model.generate_content(user_input)

        reply=response.text

        print("Bot:",reply)

    except Exception as e:
        print(f"Error:{e}")
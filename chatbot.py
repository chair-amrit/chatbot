from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages=[
    {"role":"system", "content":"You are a helpful assistant."},
]

while True:
    user_input=input("You:")

    if user_input=="bye":
        break

    messages.append({"role":"user", "content":user_input})

    reply=f"you said: {user_input}"

    print("bot:",reply)

    messages.append({"role":"assistant", "content":"reply"})

    print(messages)





"""    try:
        response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
        )
        print(response)
    except Exception as e:
        print(f"Error:{e}") """

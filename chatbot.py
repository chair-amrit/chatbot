from openai import OpenAI

client=OpenAI(api_key="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")

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

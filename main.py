from cronus import Assistant
from voice import listen, speak

agent = Assistant()

while True:
    user_text = listen()
    if not user_text:
        continue

    print(f"🧑: {user_text}")

    response = agent.run(user_text)

    print(f"🤖: {response}")
    speak(response)

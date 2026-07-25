from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-20b")

while True: 
    user_input = input("You")
    if user_input == "Exit":
        break

    response = model.invoke(user_input)
    print(response.content)
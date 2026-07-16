# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# load_dotenv()

# model = init_chat_model("groq:qwen/qwen3-32b")
# # for chunk in model.stream("write me 200 words on Artificial Intelligence?"):
# #     print(chunk.text, end="" , flush=True)


# responses = model.batch([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ])

# for response in responses:
#     print(response.text)


from langchain.tools import tool
from langchain.chat_models import init_chat_model


model = init_chat_model("groq:qwen/qwen3-32b")

@tool
def get_weather(location: str) -> str:
    "Get the current weather for a given location."
    
    return f"The current weather in {location} is sunny."

chat_model = model.bind_tools([get_weather]) 

response = chat_model.chat("What's the weather like in New York?")
print(response)  # Output: The current weather in New York is sunny.
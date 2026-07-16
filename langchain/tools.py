from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


model = init_chat_model("groq:qwen/qwen3-32b")

@tool
def get_weather(location: str) -> str:
    "Get the current weather for a given location."

    return f"The current weather in {location} is sunny."

chat_model = model.bind_tools([get_weather]) 

# response = chat_model.invoke("What's the weather like in New York?")
# print(response)  # Output: The current weather in New York is sunny.

# for tool_call in response.tool_calls:
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")


message = [{"role": "user", "content": "What's the weather like in New York?"}]
ai_msg = chat_model.invoke(message)
message.append(ai_msg)
print(message)  # Output: The current weather in New York is sunny.

for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    message.append(tool_result)


response = chat_model.invoke(message)
print(response.text)  # Output: The current weather in New York is sunny.
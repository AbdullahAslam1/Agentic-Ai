from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("groq:qwen/qwen3-32b")

ai_msg = AIMessage(
    content = [], 
    tool_calls = [{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

weather_result = 'The current weather in San Francisco is sunny.'
tool_msg = ToolMessage(
    content = weather_result,
    tool_call_id = "call_123" 
)

message = [HumanMessage(content="What's the weather like in San Francisco?"), ai_msg, tool_msg]

model_response = model.invoke(message)
print(model_response.text)  # Output: The current weather in San Francisco is sunny.
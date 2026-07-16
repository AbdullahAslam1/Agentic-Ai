from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage  , HumanMessage , ToolMessage 
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

# model = init_chat_model("groq:qwen/qwen3-32b")

agent = create_agent(
    model = init_chat_model("groq:qwen/qwen3-32b"),
    checkpointer = InMemorySaver(),
    middleware = [SummarizationMiddleware(
        model = init_chat_model("groq:qwen/qwen3-32b"),
        trigger = ("messages",10), 
        keep = ("messages", 4)
    )]
)

config = {"configurable" : {"thread_id" : "test-1"}}

questions = [
    "What is 2+2?",
    "What is 10*5?",
    "What is 100/4?",
    "What is 15-7?",
    "What is 3*3?",
    "What is 4*4?",
]

for q in questions:
    response = agent.invoke({"messages": [HumanMessage(content=q)]}, config = config)
    print(f"Messages: {response}")
    print(f"Messages: {len(response['messages'])}")

    
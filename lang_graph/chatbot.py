from typing import Annotated
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph , START , END
from langgraph.graph import StateGraph, add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

model = init_chat_model("groq:qwen/qwen3.6-27b")
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list , add_messages]


# def chatbot(state: State):
#     return {"messages": model.invoke(state["messages"])}

# graph_builder = StateGraph(State)

# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)


# graph = graph_builder.compile()


# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception: 
#     pass 


# response = graph.invoke({"messages": "Hello, how are you?"})
# response = graph.invoke({"messages": "Can u tell me what to learn in Agentic Ai in 2026?"})
# print(response["messages"][1].content)  # Print the content of the second message in the respons


## Chatbot with Tool 

tool = TavilySearch(max_results=3)
tool.invoke("How to learn Agentic AI in 2026?")

def multiply(a: int , b: int) -> int:
    '''Multiply a and b
    
    Args:
        a (int): The first number to multiply.  
        b (int): The second number to multiply.

    Returns:
        int: The product of a and b.
    '''
    return a * b


tools =[tool , multiply]
model_with_tools = model.bind_tools(tools)


def tool_chatbot(state: State):
    return {"messages": model_with_tools.invoke(state["messages"])}

builder = StateGraph(State)
builder.add_node("tool_chatbot", tool_chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_chatbot")
builder.add_conditional_edges("tool_chatbot", tools_condition)
builder.add_edge("tools", "tool_chatbot")
graph = builder.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "1"}}

response = graph.invoke({"messages": "My name is abdullah"}, config=config)

response = graph.invoke({"messages": "What is my name"}, config=config)
print(response["messages"][1].content)  # Print the content of the second message in the response

response = graph.invoke({"messages": "how do u remember my name"}, config=config)
print(response["messages"][1].content)


def superBot(state: State):
    return {"messages": model_with_tools.invoke(state["messages"])}

graph = StateGraph(State)

graph.add_node("superBot", superBot)

graph.add_edge(START, "superBot")
graph.add_edge("superBot", END)

graph_builder = graph.compile(checkpointer=memory)

display(Image(graph_builder.get_graph().draw_mermaid_png()))

config  = {"configurable": {"thread_id": "1"}}
for chunk in graph_builder.stream({"messages": "My name is abdullah"}, config=config , stream_modes = "values"):
    print(chunk)

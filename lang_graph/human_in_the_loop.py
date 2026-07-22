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
from langgraph.types import Command , interrupt
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
load_dotenv()

model = init_chat_model("groq:qwen/qwen3.6-27b")
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list , add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    '''This function is used to get human assistance for a query.
    
    Args:
        query (str): The query for which human assistance is needed.

    Returns:
        str: The response from the human.
    '''
    human_response = interrupt({"query": query})
    return human_response["response"]


tool = TavilySearch(max_results=3)
tools = [tool , human_assistance]
tools_model = model.bind_tools(tools)

def chatBot(state:State):
    message = tools_model.invoke(state["messages"])
    return {"messages": [message]}

# 1. Register your Nodes
graph_builder.add_node("chatbot", chatBot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 2. Add Entry and Loop Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

# 3. Add the Conditional Routing (This handles BOTH "tools" and END paths!)
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition
)

# 4. Compile with Memory
graph = graph_builder.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))

user_input = "I need some expert guidance and assistance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": user_input},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"response": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
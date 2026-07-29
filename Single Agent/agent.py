from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model 
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain.agents import create_agent
from langchain.tools import tool 
import os 
import requests
from langchain_core.prompts import PromptTemplate

load_dotenv() 
llm = init_chat_model("groq:openai/gpt-oss-120b")

search_tool = TavilySearchResults(max_results = 2)


@tool
def get_weather_data(city: str)-> str: 
    """  Fetch current weather information for a city.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key: 
        return "Error: OpenWeather API key is not set."

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric" 

    response = requests.get(url)
    if response.status_code != 200:
        return f"Error fetching weather for {city}: {response.json().get('message', 'Unknown error')}"

    data = response.json()
    temp = data["main"]["temp"]
    condition = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]

    return f"Weather in {city}: {temp}°C with {condition}. Humidity: {humidity}%."



agent = create_agent(
    model = llm , 
    tools= [search_tool , get_weather_data],
    system_prompt= "You are a helpful assistant with access to web search and real-time weather tools."
)

response = agent.invoke({"messages" : [("user", "what is the weather in Lahore and also what is the latest news in the world of AI as of 28 July 2026")]})
print(response["messages"][-1].content)

agent.invoke({
    "messages" : [
        ("user" , "what is the weather condition in  London")
    ]
})

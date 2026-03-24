from email import message
from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core import messages
import requests

load_dotenv()

@tool('get_weather', description="return weather information for given city", return_direct=False)
def get_weather(city:str)-> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()
    

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [get_weather],
    system_prompt = "You are a helpful assistant"
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
print(result['messages'][-1].content)
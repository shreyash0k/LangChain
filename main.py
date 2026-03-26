from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
import requests
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id:str


@dataclass
class ResponseFormat:
    one_line_summary: str
    temp_celsius: float
    temp_fahrenheit: float
    humidity: float

@tool('get_weather', description="return weather information for given city", return_direct=False)
def get_weather(city:str)-> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()
    

@tool('locate_user', description="lookup a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Mumbai'
        case 'PQR123':
            return 'Charlotte'
        case _:
            return 'Unknown'

model = init_chat_model(
    model = 'claude-haiku-4-5',
    temperature = 0.3
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = "You are a helpful assistant",
    context_schema= Context,
    response_format= ResponseFormat,
    checkpointer= checkpointer
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "My secret codename for this chat is BLUE-7. Remember that."}]},
    config,
    context=Context(user_id="PQR123"),
)

print(response['structured_response'])
print(response['structured_response'].one_line_summary)
print(response['structured_response'].temp_celsius)
print(response['structured_response'].temp_fahrenheit)
print(response['structured_response'].humidity)


config = {"configurable": {"thread_id": "2"}}


response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my secret codename?"}]},
        config,
        context = Context(user_id="PQR123"),
    )


print(response['structured_response'])
print(response['structured_response'].one_line_summary)
print(response['structured_response'].temp_celsius)
print(response['structured_response'].temp_fahrenheit)
print(response['structured_response'].humidity)
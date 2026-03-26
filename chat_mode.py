from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite-preview'
)

conversations = [
    SystemMessage("You are helpful assistant"),
    HumanMessage("is 2020 a leap year"),
    AIMessage("yes 2020 is a leap year"),
    HumanMessage("why is 2020 a leap year"),
]

for chunk in model.stream(conversations):
    print(chunk.text, end = '', flush = True)

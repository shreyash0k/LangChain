from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite-preview'
)

response = model.invoke("What is python")

print(response.content)
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

embeddings = VoyageAIEmbeddings(model="voyage-3.5")

documents = [
    "I like to play cricket",
    "I am a software engineer",
    "Space is vast",
    "I like watching bollywood movies"
]

retriever = KNNRetriever.from_texts(documents, embeddings)


@tool("search_personal_facts", description="Search through personal facts and documents to answer questions about the user")
def search_personal_facts(query: str) -> str:
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])


model = init_chat_model(
    model="claude-haiku-4-5",
    temperature=0.3
)

agent = create_agent(
    model=model,
    tools=[search_personal_facts],
    system_prompt="You are a helpful assistant. Use the search_personal_facts tool to look up information about the user when needed.",
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What do I like to do for fun?"}]},
    config,
)

print(response["messages"][-1].content)

from dotenv import load_dotenv
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "DocumentLoaders\FutureofAI.txt"


data = TextLoader(file_path, encoding="utf-8")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarizes the text"),
     ("human", "{data}")]
)

model = ChatMistralAI( model = "mistral-small-2506")

prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)
from dotenv import load_dotenv
from pathlib import Path

from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 

url = "https://groww.in/blog/current-market-condition-sectors-future"

data = WebBaseLoader(url)
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system","Your AI that summarizes the complete website and answer human question"),
        ("human", "{data}, which one sector will the most profitable in future")
    ]
)

prompt = template.format_messages(data = docs)

model = ChatMistralAI(model = "mistral-small-2506")

result = model.invoke(prompt)

print(result.content)
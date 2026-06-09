from dotenv import load_dotenv
from pathlib import Path 

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate 

load_dotenv() 

BASE_DIR = Path(__file__).resolve().parent 

file_path = BASE_DIR / "TransformersArch.pdf"

data = PyPDFLoader(file_path)

docs = data.load() 

template = ChatPromptTemplate([
    ("system",'Consider your a AI that summarizes PDF'),
    ("human", "{data}")
])

prompt = template.format_messages(data = docs)

model = ChatMistralAI(model = "mistral-small-2506")

result = model.invoke(prompt)

print(result.content)
from dotenv import load_dotenv
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "DocumentLoaders\TransformersArch.pdf"


data = PyPDFLoader(file_path)
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarizes the text"),
     ("human", "{data}")]
)

model = ChatMistralAI( model = "mistral-small-2506")

prompt = template.format_messages(data = chunks)

result = model.invoke(prompt)

print(result.content)
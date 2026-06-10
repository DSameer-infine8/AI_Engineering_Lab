#--------------------------------
# Steps
#---------------------------------

# Load PDF -> Split into Chunks -> Create Embeddings -> Strore into ChromaDB

from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "DocumentLoaders\Hyperfocus.pdf"

data = PyPDFLoader(file_path)
docs = data.load() 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#Specifies where the Store the embeddings 
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding= embedding_model,
    persist_directory= "Chroma_BookDB",
)




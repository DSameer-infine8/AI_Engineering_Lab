from dotenv import load_dotenv
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#Specifies where the Store the embeddings 
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding= embedding_model,
    persist_directory= "Chroma_DB",
)

similar = vectorstore.similarity_search("What is used for data analysis?", k=2)
# Note: Vector Stores are not responsible for answering our question ,they are just responsible for retriving the information

for i in similar:
    print(i)
    
#-------------------
# Retrivers
#-------------------

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
    
    
print('-----------------------')
print(f"Top 1:{docs[0].page_content}")
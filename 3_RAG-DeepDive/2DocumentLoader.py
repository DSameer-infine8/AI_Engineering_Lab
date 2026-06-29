import os 
import tempfile
from pathlib import Path 
from dotenv import load_dotenv 
from langchain_community.document_loaders import ( DirectoryLoader ,
                                                  WebBaseLoader,
                                                  PyPDFLoader,
                                                  TextLoader)

env_path = Path(__file__).resolve().parent.parent/"2_RAG-Fundamentals"/".env"
load_dotenv(dotenv_path=env_path)

def load_text_file():
    # Creating a temporary text file 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is sample text file.\n This file is used to understand how loader works")
        temp_file_path = temp_file.name 
    
    try:
        # Load the text file using Text Loaders
        loader= TextLoader(temp_file_path)
        documents = loader.load()
        print(temp_file.name) 
        print("================================")
        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
        print("================================")

        
        for doc in documents:
            print("Document Content")
            print(doc)
            print(doc.page_content)
            
    finally:
        #Clean up the Temprory file 
        os.remove(temp_file_path)
        
        
def pdf_loader():
    BASE_DIR = Path(__file__).resolve().parent
    loader = PyPDFLoader(BASE_DIR / "docs/langchain_demo.pdf")
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print("======="*5)
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")
        print("======="*5)
    


if __name__== "__main__":
    #load_text_file()
    pdf_loader()
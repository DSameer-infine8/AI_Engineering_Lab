from dotenv import load_dotenv 
from pathlib import Path 

from langchain_mistralai import ChatMistralAI 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate


load_dotenv() 


# Embedding model selected 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vectore Store is loaded
vectorstore = Chroma(
    persist_directory= "Chroma_BookDB",
    embedding_function= embedding_model
)



# Retriever is built

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "fetch_k":10,         #first it fetchs 10 docs and later mmr is applied on these 10 docs and 4 are selected and retrieved
        "lambda_mult":0.5     #helps in retrieving diverse results 1=no diverse and 0=more diverse
    }
)



# LLM is ready

llm = ChatMistralAI(model = "mistral-small-2506")


# Prompt template 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """  
         You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say: "I could not find the answer in the document."
        """),
        
        ("human", 
         """
         Context:{context}
         Question:{question}
         """)
    ]
)

print("==========================")
print("Fully Loaded RAG System")
print("==========================")

print("Press 0 to EXIT")

while True:
    query = input("You: ")
    if query == '0':
        break
    
    docs = retriever.invoke(query)
    
    # Combining All retrieved docs 
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)
    
    print(f"\n AI: {response.content}")
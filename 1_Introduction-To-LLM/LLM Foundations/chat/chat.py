from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access API key
api_key = os.getenv("GROQ_API_KEY")



import langchain 


from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=api_key,
    model="qwen/qwen3-32b"
)

messages = [
    (
        "system",
        "You are a helpful Software Engineer, assistent new tech employees with there questions and provide roadmap to learn there skills. ",
    ),
    ("human", "What is GenAI."),
]
ai_msg = llm.invoke(messages)


print(ai_msg.content)
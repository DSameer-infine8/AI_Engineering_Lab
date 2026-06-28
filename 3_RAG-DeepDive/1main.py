from dotenv import load_dotenv 
from importlib.metadata import version
from pathlib import Path

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Get the project root
env_path = Path(__file__).resolve().parent.parent /"2_RAG-Fundamentals"/".env"

load_dotenv(dotenv_path=env_path)

print(f"langchain-core version:{core_version}" )
print(f"lang-graph version:{lg_version}" )


print("Hello World from Lang-Chain..")

def main():
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    response = llm.invoke("Say 'Set-Up Complete' in one word")
    print(f"Mistral Responded: {response.content}")

        
    llm2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    response = llm2.invoke("Say 'Set-Up Complete' in one word")
    print(f"Gemini Responded: {response}")
    
    
if __name__ == "__main__":
    main()
    
    

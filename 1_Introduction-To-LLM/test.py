from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access API key
api_key = os.getenv("GROQ_API_KEY")

print(api_key)


import langchain 

print(langchain.__version__)


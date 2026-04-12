import os 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv 


load_dotenv() 

hugging_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_key


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)
model = ChatHuggingFace(llm=llm)

response = model.invoke("who are you and how can you help developers?")

print(response.content)
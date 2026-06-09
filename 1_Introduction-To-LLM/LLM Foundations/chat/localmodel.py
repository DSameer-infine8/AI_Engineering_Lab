import os 
from dotenv import load_dotenv 

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text_generation",
    pipeline_kwargs= dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03
    )
)

chat_model = ChatHuggingFace(llm = llm)

result = chat_model.invoke("What is Data Science")

print(result.content)
from dotenv import load_dotenv 
import os


load_dotenv() 

mistral_key=os.getenv("MISTRAL_API_KEY")

from langchain_mistralai import ChatMistralAI 


#Parameters Used: Temperature and max_tokens
'''
Temprature(ranges from 0 to 1): 0 reflects the model should give me more reasoning and logical side with 0 creativity , where as value increase to 1 means to add randomness or creativity.
max_tokens: To limit the number of words to be genetrated by the model.
'''



model = ChatMistralAI(model = "mistral-small-2506",api_key=mistral_key, temperature=0.8, max_tokens=300)   # temperature between 0(No creative ,only used for logical/reasoning answer) to 1(for creative work and answering)

response = model.invoke("Give me a poem on -How fast the world is changing")

print(response.content)
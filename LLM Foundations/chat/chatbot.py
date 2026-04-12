from dotenv import load_dotenv 
import os


load_dotenv() 

mistral_key=os.getenv("MISTRAL_API_KEY")

from langchain_mistralai import ChatMistralAI 

# manual way to save previous messages with chatbot

messages = []


model = ChatMistralAI(model = "mistral-small-2506",api_key=mistral_key, temperature=0.8)   # temperature between 0(No creative ,only used for logical/reasoning answer) to 1(for creative work and answering)

print("------------Enter 0 to exit the chat-----------------")
while True:
    prompt = input("You :")
    messages.append(prompt)
    if prompt == "0":
        break
    
    response = model.invoke(messages)
    messages.append(response.content)
    print("Bot :", response.content)
    
    
print(messages)
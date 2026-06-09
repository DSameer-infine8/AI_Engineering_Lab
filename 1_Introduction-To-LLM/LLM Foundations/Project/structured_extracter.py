from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(model="mistral-small-2506",api_key=api_key)

class Movie(BaseModel):
    title: str
    release_year : Optional[int]
    genre:List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph and if not present save it as NULL
{format_instructions}
"""),
    ("human", "{paragraph}")
])


para = input("Give me movie review paragraph :")

final = prompt.invoke(
    {
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    }
)
response = model.invoke(final)

print(response.content)
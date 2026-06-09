from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(model="mistral-small-2506",api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    
"""
You are an intelligent information extraction assistant.

Your task is to analyze the given movie review paragraph and extract all important details in a clear and structured format.

### Instructions:
1. Read the paragraph carefully.
3. If information is missing -> write NULL
2. Identify key information such as:
   - Movie Title
   - Director
   - Release Year
   - Main Cast
   - Genre
   - Plot Summary (short, 2–3 lines)
   - Key Themes or Highlights
   - Music Composer (if mentioned)
   - Ratings (IMDb or others if available)
   - Notable Achievements or Recognition

3. Present the extracted information in a well-structured table with two columns:
   - "Attribute"
   - "Details"

4. After the table, provide a short summary of the paragraph in 2–3 sentences.

### Output Format:

| Attribute | Details |
|----------|--------|
| Movie Title | ... |
| Director | ... |
| Release Year | ... |
| Cast | ... |
| Genre | ... |
| Plot | ... |
| Highlights | ... |
| Music | ... |
| Rating | ... |
| Recognition | ... |

### Summary:
Write a concise summary here.

"""
),
("human",
 """
 Input Paragraph
{paragraph}

"""
)])

para = input("Provide the review :")
final = prompt.invoke({
    "paragraph": para
})



response = model.invoke(final)

print(response.content)
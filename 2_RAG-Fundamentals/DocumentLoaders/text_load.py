from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter

'''
#Character-based Splitter
splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 1000,
    chunk_overlap = 100
)


#Token-Based Splitter
splitter = TokenTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)



#Recursive Character-Based Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
'''




# BEST SPLITTER 
# Semantic(Meaning)- Based Splitting

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "FutureofAI.txt"

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

docs = loader.load()

chunks = splitter.split_documents(docs)

print(len(chunks))

for i in range(len(chunks)):
 print(chunks[i].page_content)
 print('------------')
 print()
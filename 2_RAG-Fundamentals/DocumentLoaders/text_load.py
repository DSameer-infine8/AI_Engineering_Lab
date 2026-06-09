from pathlib import Path
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "FutureofAI.txt"

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

docs = loader.load()

print(docs[0].page_content)
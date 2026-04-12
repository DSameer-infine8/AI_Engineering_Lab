from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

texts = [
    "Hello everyone, hope you all are doing good..!",
    "Hello my name is Sameer",
    "And I am on my way to become AI Engineer"
]

vector = embeddings.embed_documents(texts)

print(vector)
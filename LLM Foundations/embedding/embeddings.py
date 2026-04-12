from dotenv import load_dotenv 

load_dotenv() 

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    dimensions=64
)

texts = [
    "Hello everyone, hope you all are doing good..!",
    "Hello my name is Sameer",
    "And I am on my way to become AI Engineer"
]

vector = embeddings.embed_documents(texts)

print(vector)
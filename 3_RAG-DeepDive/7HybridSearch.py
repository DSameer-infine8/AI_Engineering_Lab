from pathlib import Path 
from dotenv import load_dotenv 
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever






env_path = Path(__file__).resolve().parent.parent/ "2_RAG-Fundamentals" / ".env"
load_dotenv(dotenv_path=env_path)



# Documents with both semantic content AND specific identifiers

documents = [Document(page_content='Product SKU-7742X is our flagship router. It supports '
             'gigabit speeds and advanced QoS features.',
             metadata={'type': 'product'}
             ),
    Document(page_content='For network connectivity issues, first check the '
             'ethernet cable and router status lights.',
    metadata={'type': 'troubleshooting'}
    ),
    Document(page_content='Error code E_CONN_REFUSED indicates the server '
             'rejected the connection. Check firewall settings.',
    metadata={'type': 'error'}
    ),
    Document(page_content='The authentication process requires valid credentials. '
             'Use OAuth2 for secure API access.',
    metadata={'type': 'auth'}
    ),
    Document(page_content='Router configuration guide: Access the admin panel '
             'at 192.168.1.1 to modify settings.',
    metadata={'type': 'config'}
    ),
    Document(page_content='WCAG 2.1 compliance requires all images to have '
             'alt text and sufficient color contrast.',
    metadata={'type': 'compliance'}
    )
]


print(f"Loaded {len(documents)} documents")

#Creating Embedding and VectoreStor 

embeddings_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

vectorestore = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name='hybrid_test'
)

vector_retriever = vectorestore.as_retriever(
    search_kwargs = {'k':3}
)

print('Vector retriever ready')

bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

print("BM25 retriever ready")

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], 
    weights=[0.5, 0.5]  # Must sum up to 1.0
)

print("Hybrid Search is ready")


def test_query(query, name, retiriever):
    results = retiriever.invoke(query)
    print(f'\n{name}- Query: \"{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f'{i+1}. {preview}')
    return results 

# Test queries to analyse search results 
test_queries = [
    'SKU_7742X specifications',
    'E_CONN_REFUSED error',
    'How do I authenticate',
    'WCAG compliance',
    'router configuration'
]

for query in test_queries:
    print("="*60)
    
    vector_results = test_query(query, "VECTOR", vector_retriever)
    
    bm25_results = test_query(query, "BM25", bm25_retriever)
    
    hybrid_results = test_query(query, "HYBRID", ensemble_retriever)
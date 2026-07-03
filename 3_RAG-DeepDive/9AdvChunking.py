''' Very Important 
Semantic V/S Recursive Chunking
'''
import os
from pathlib import Path 
from dotenv import load_dotenv 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 


env_path = Path(__file__).resolve().parent.parent/ "2_RAG-Fundamentals" / ".env"
load_dotenv(dotenv_path=env_path)

#Creating Embedding and VectoreStor 

embeddings_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

# Sample document with distinct topics
document = '''
# Authentication Guide
## 0Auth2 Authentication
To authenticate with our API, vou need 0Auth2 credentials.
First, obtain a client id and client secret from the developer portal. Make a POST request to /oauth/token with grant_type=client credentials The response contains an access token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE _LIMITED, INVALID REQUEST.
Always check the HTTP status code first, then parse the error body. 

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm. F ree tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, vou receive a 429 status code.
The Retry-After header indicates when to retry.
'''


# Recursive Chunking
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap=50,
    separators=['\n\n','\n','. ', ' ']
)

recursive_chunks = recursive_splitter.split_text(document) 

print(f"Recursive Chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"\n-------- Chunk {i+1}: ({len(chunk)} chars)  -----------")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
 
# Semantic Chunking   
semantic_chunker = SemanticChunker(
    embeddings=embeddings_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90  #split at 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)

print(f"Semantic Chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\n---- Chunk {i+1}: {len(chunk)} char ------")
    print(chunk[:100] +"..." if len(chunk) >100 else chunk)
    
    

def smart_chunker(
    text:str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500
) -> list[str]:
    '''
    Production chunking with semantic as primary, recursive as fallback
    '''
    embeddings_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
    
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings=embeddings_model,
                breakpoint_threshold_type=90,
                breakpoint_threshold_amount='percentile'
            )
            chunks = chunker.split_text(text)
            
            # Validate chunks are not too long 
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                # Fallback to recursive for oversized chunks 
                return _recursive_fallback(text, fallback_chunk_size)
            
            return chunks
        
        except Exception as err:
            print(f'Semantic Chunking failed: {err}, using fallback')
            return _recursive_fallback(text, fallback_chunk_size)
        
    return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text:str, chunk_size:int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap=50
    )
    return splitter.split_text(text)


#Usage 

# Note: True when you want to use semantic chunking for your Use-Case
#       False when you want to use recursive chunking.
chunks = smart_chunker(document, use_semantic=False)
print(f'Create {len(chunks)} semantic chunks')
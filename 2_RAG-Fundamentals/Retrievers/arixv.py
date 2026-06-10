from langchain_community.retrievers import ArxivRetriever

# create the retrivers 
retriever = ArxivRetriever(
    load_max_docs=4,
    load_all_available_meta=True
)

# query arxiv 
docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"Result: {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:500])
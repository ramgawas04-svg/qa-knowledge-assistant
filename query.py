from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("playwright_docs")

while True:

    question = input("\nAsk a question (exit to quit): ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    print("\n=== RAW RESULTS ===")
    print(results)
    print("\n=== RESULTS ===\n")

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for i, doc in enumerate(docs):
        print(f"Result {i+1}")
        print(f"Source: {metas[i]['source']}")
        print(doc)
        print("-" * 50)
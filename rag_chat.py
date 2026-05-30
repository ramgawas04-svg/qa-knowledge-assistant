from sentence_transformers import SentenceTransformer
import chromadb
from ollama import chat

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect Chroma
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("playwright_docs")

while True:

    question = input("\nAsk a question (exit to quit): ")

    if question.lower() == "exit":
        break

    # Create embedding
    query_embedding = model.encode(question).tolist()

    # Retrieve top chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    docs = results["documents"][0]

    # Build context
    context = "\n\n".join(docs)

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n=== ANSWER ===\n")
    print(response.message.content)
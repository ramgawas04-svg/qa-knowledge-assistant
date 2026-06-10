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
    print("\n=== RAW RESULTS ===")
    print(results)
    docs = results["documents"][0]
    distances = results["distances"][0]


    filtered_docs = []
    
    for doc, distance in zip(docs, distances):
        if distance < 0.8:   # experiment with value
            filtered_docs.append(doc)

    if not filtered_docs:
        print("No relevant information found.")
        continue
    # Build context
    context = "\n\n".join(docs)

    prompt = f"""
You are a QA documentation assistant.

Answer ONLY using facts explicitly present in the provided context.

Do not add explanations, assumptions, or external knowledge.

If the answer is not fully present in the context, say:
"The provided documents do not contain additional information."

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
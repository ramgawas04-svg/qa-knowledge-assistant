from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection
try:
    client.delete_collection("playwright_docs")
except:
    pass

collection = client.get_or_create_collection(
    name="playwright_docs"
)

total_chunks = 0

for file_name in os.listdir("docs"):

    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join("docs", file_name)

    print(f"\nProcessing: {file_name}")

    with open(file_path, "r") as f:
        text = f.read()

    # Split on blank lines
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]
    overlap_chunks = []

    for i in range(len(chunks) - 1):
        combined_chunk = chunks[i] + "\n\n" + chunks[i + 1]
        overlap_chunks.append(combined_chunk)
        
    chunks = overlap_chunks

    for chunk_id, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        collection.add(
            ids=[f"{file_name}_{chunk_id}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": file_name,
                "chunk_id": chunk_id
            }]
        )

        print(
            f"Stored chunk {chunk_id} "
            f"from {file_name}"
        )

        total_chunks += 1

print(f"\nTotal chunks stored: {total_chunks}")
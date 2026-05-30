from sentence_transformers import SentenceTransformer
import chromadb

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read document
with open("docs/playwright_guide.txt", "r") as f:
    text = f.read()

# Split by blank lines
parts = [p.strip() for p in text.split("\n\n") if p.strip()]

# Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection if exists
try:
    client.delete_collection("playwright_docs")
except:
    pass

collection = client.get_or_create_collection(
    name="playwright_docs"
)

chunk_count = 0

# Skip first title
i = 1

while i < len(parts) - 1:

    section_name = parts[i]
    content = parts[i + 1]

    embedding = model.encode(content).tolist()

    collection.add(
        ids=[f"chunk_{chunk_count}"],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "source": "playwright_guide.txt",
            "section": section_name,
            "chunk_id": chunk_count
        }]
    )

    print(f"Stored chunk: {section_name}")

    chunk_count += 1
    i += 2

print(f"\nTotal chunks stored: {chunk_count}")
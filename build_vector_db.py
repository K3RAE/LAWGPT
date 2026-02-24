import json
import uuid
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- PATH SETUP ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dev.jsonl")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# ---------------- INIT CHROMA (PERSISTENT) ----------------

client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_or_create_collection(name="legal_cases")

# ---------------- LOAD EMBEDDING MODEL ----------------

print("🔹 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- CHUNK FUNCTION ----------------

def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# ---------------- BUILD VECTOR DATABASE ----------------

print("🔹 Reading dataset and building embeddings...")

MAX_RECORDS = 2000  # Adjust if needed

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):

        if idx >= MAX_RECORDS:
            break

        if idx % 100 == 0:
            print(f"Processing record {idx}")

        record = json.loads(line)
        full_text = " ".join(record["text"])

        for chunk in chunk_text(full_text):

            embedding = embedder.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "case_id": record["id"],
                    "full_text": full_text
                }],
                ids=[str(uuid.uuid4())]
            )

print("✅ Vector database built successfully!")
print("📦 Total documents:", collection.count())
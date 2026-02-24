import os

LEGAL_DOCS_FOLDER = "legal_docs"
CHUNK_SIZE = 300  # number of characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks


def load_documents():
    documents = []

    for file_name in os.listdir(LEGAL_DOCS_FOLDER):
        if file_name.endswith(".txt"):
            file_path = os.path.join(LEGAL_DOCS_FOLDER, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "file_name": file_name,
                "text": text
            })

    return documents


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP

    return chunks


if __name__ == "__main__":
    documents = load_documents()

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "file_name": doc["file_name"],
                "chunk_id": i,
                "text": chunk
            })

    print(f"Total chunks created: {len(all_chunks)}\n")

    for chunk in all_chunks:
        print(f"File: {chunk['file_name']} | Chunk ID: {chunk['chunk_id']}")
        print(chunk["text"])
        print("-" * 50)

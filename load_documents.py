import os

LEGAL_DOCS_FOLDER = "legal_docs"

def load_documents():
    documents = []

    # Loop through all files in legal_docs folder
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


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded {len(docs)} document(s)\n")

    for doc in docs:
        print("File:", doc["file_name"])
        print("Content:")
        print(doc["text"])
        print("-" * 40)

import os

# Folder where user uploads documents
INPUT_FOLDER = "input_docs"

def read_document(file_name):
    file_path = os.path.join(INPUT_FOLDER, file_name)

    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


if __name__ == "__main__":
    # For now, we hardcode the file name
    document_name = "sample_case.txt"

    document_text = read_document(document_name)

    if document_text:
        print("\n===== DOCUMENT CONTENT START =====\n")
        print(document_text)
        print("\n===== DOCUMENT CONTENT END =====")


"""
Script to build the ChromaDB vector index from granular cultural documents.
Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 embedding model.
"""

import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = Path("data/cultural/chroma_db")
DOCUMENTS_PATH = Path("data/cultural/cultural_documents.json")
COLLECTION_NAME = "cultural_knowledge"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index():
    print(f"Loading cultural documents from {DOCUMENTS_PATH}...")
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Documents file not found at {DOCUMENTS_PATH}")

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} cultural documents.")

    print(f"Initializing ChromaDB PersistentClient at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH.resolve()))

    embedding_func = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    # Delete existing collection if it exists to prevent duplicate indexing
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    print(f"Creating collection '{COLLECTION_NAME}' with embedding model '{MODEL_NAME}'...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    formatted_documents = []
    metadatas = []

    for doc in docs:
        ids.append(doc["id"])
        formatted_text = f"{doc['title']}: {doc['content']}"
        formatted_documents.append(formatted_text)
        metadatas.append({
            "category": doc.get("category", ""),
            "title": doc.get("title", "")
        })

    print(f"Indexing {len(ids)} documents into ChromaDB...")
    collection.add(
        ids=ids,
        documents=formatted_documents,
        metadatas=metadatas
    )

    print(f"Successfully indexed {collection.count()} documents into ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()

import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent 
load_dotenv(BASE_DIR / ".env")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # <--- nouveau import

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_DIR = Path("data/processed/index")


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    docs = []
    for item in raw_chunks:
        docs.append(
            Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )
        )
    return docs


def build_index():
    docs = load_chunks()
    print(f"[INDEX] Nb documents/chunks: {len(docs)}")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"[INDEX] FAISS sauvé dans {INDEX_DIR}")


if __name__ == "__main__":
    build_index()

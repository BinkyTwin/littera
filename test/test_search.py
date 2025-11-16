import os
from pathlib import Path
from dotenv import load_dotenv

# ⚠️ Workaround OpenMP (faiss / Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

INDEX_DIR = "data/processed/index"


def test_query(query: str):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=3)
    for d in docs:
        print("----")
        print("Source:", d.metadata.get("file_name"), "page", d.metadata.get("page"))
        print(d.page_content[:400], "...\n")


if __name__ == "__main__":
    test_query("RAG definition and use cases")


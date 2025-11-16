from pathlib import Path
import json

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = Path("data/pdf")
OUT_PATH = Path("data/processed/chunks.json")


def load_pdfs(pdf_dir: Path):
    all_docs = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"[LOAD] {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            d.metadata["file_name"] = pdf_path.name
        all_docs.extend(docs)
    return all_docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    print(f"[CHUNK] {len(docs)} documents (pages) → chunking...")
    chunks = splitter.split_documents(docs)
    print(f"[CHUNK] Total chunks: {len(chunks)}")
    return chunks


def save_chunks(chunks, out_path: Path):
    serializable = []
    for i, c in enumerate(chunks):
        meta = dict(c.metadata) if c.metadata else {}
        meta["chunk_id"] = i
        serializable.append(
            {
                "text": c.page_content,
                "metadata": meta,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {len(chunks)} chunks → {out_path}")


def main():
    docs = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs)
    save_chunks(chunks, OUT_PATH)


if __name__ == "__main__":
    main()

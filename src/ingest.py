from pathlib import Path
import json

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

PDF_DIR = Path("data/pdf")
OUT_PATH = Path("data/processed/chunks.json")


def load_pdfs(pdf_dir: Path) -> list[Document]:
    """Charge tous les PDF du dossier en une liste de Documents LangChain."""
    all_docs: list[Document] = []

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"[LOAD] {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()  # 1 Document par page, avec metadata {"source": "...", "page": ...}
        for d in docs:
            d.metadata["file_name"] = pdf_path.name
        all_docs.extend(docs)

    return all_docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Découpe les Documents en chunks avec LangChain."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # à ajuster si besoin
        chunk_overlap=200,    # 10–20% d’overlap
        separators=["\n\n", "\n", ". ", " ", ""],  # découpe intelligente
    )

    print(f"[CHUNK] {len(docs)} documents (pages) → chunking...")
    chunks = text_splitter.split_documents(docs)
    print(f"[CHUNK] Total chunks: {len(chunks)}")
    return chunks


def save_chunks(chunks: list[Document], out_path: Path) -> None:
    """Sauvegarde les chunks dans un JSON simple (texte + métadonnées)."""
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

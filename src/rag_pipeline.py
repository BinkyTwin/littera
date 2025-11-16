# src/rag_pipeline.py

import os
import sys
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

# Workaround OpenMP (comme dans test_search)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Charge le .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

INDEX_DIR = BASE_DIR / "data" / "processed" / "index"


def load_vectorstore() -> FAISS:
    """Charge l'index FAISS sauvegardé précédemment."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def retrieve_context(
    query: str,
    k: int = 4,
) -> tuple[str, List[Dict]]:
    """
    1) Transforme la question en vecteur
    2) Récupère les k chunks les plus proches
    3) Construit une string 'contexte' + une liste de sources pour les citations
    """
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    # Construire le contexte brut pour le LLM
    context_parts = []
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("file_name", "unknown")
        page = meta.get("page", meta.get("page_num", ""))
        chunk_id = meta.get("chunk_id", i)

        # Ce qui part dans le prompt
        context_parts.append(
            f"[SOURCE {i}] (file={source}, page={page})\n{d.page_content}\n"
        )

        # Ce qu'on garde pour les afficher après
        sources.append(
            {
                "source_id": i,
                "file_name": source,
                "page": page,
                "chunk_id": chunk_id,
                "preview": d.page_content[:200].replace("\n", " "),
            }
        )

    full_context = "\n\n".join(context_parts)
    return full_context, sources


def call_llm(
    question: str,
    context: str,
    model_name: str = "gpt-4o-mini",
    language: str = "fr",
) -> str:
    """
    Appelle le LLM avec un prompt RAG :
    - contexte + question
    - consigne : ne répondre que sur la base du contexte
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
    )

    system_prompt = f"""
Tu es un assistant académique qui aide à faire des revues de littérature.
Tu dois répondre UNIQUEMENT en te basant sur le CONTEXTE fourni.
Si le contexte ne contient pas assez d'information, tu le dis honnêtement.

Règles :
- réponds en {language}
- cite explicitement les [SOURCE X] quand tu t'appuies sur un passage
- ne fais PAS de référence à d'autres documents que ceux du contexte
"""

    user_prompt = f"""
CONTEXTE :

{context}

QUESTION :
{question}

Réponds de manière structurée, en citant les sources entre crochets, par exemple : 
"Selon [SOURCE 1] ...", "D'autres auteurs ([SOURCE 2], [SOURCE 3]) indiquent que ..."
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)
    return response.content


def answer_question(question: str) -> None:
    """Pipeline complet : retrieve → LLM → affichage."""
    print(f"\n=== QUESTION ===\n{question}\n")

    context, sources = retrieve_context(question, k=4)
    answer = call_llm(question, context)

    print("=== RÉPONSE (RAG) ===")
    print(answer)
    print("\n=== SOURCES UTILISÉES ===")
    for s in sources:
        print(
            f"[SOURCE {s['source_id']}] {s['file_name']} (page {s['page']}) "
            f"- preview: {s['preview']}..."
        )


if __name__ == "__main__":
    # Permet de lancer depuis la ligne de commande :
    # python src/rag_pipeline.py "ta question ici"
    if len(sys.argv) < 2:
        print("Usage: python src/rag_pipeline.py \"Votre question ici\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer_question(question)

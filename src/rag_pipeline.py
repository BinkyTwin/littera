# src/rag_pipeline.py
import os
from pathlib import Path

# ⚠️ Workaround OpenMP (FAISS sous Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # pour les embeddings uniquement

# ====== Chargement env & config ======

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Clé OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY n'est pas défini dans le .env")

# Modèle OpenRouter (modifiable à un seul endroit)
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"

# Index FAISS déjà construit
INDEX_DIR = BASE_DIR / "data/processed/index"


# ====== Initialisation clients ======

# Client OpenRouter (LLM)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Embeddings OpenAI (pour FAISS) - nécessite OPENAI_API_KEY dans .env
embeddings = OpenAIEmbeddings()


def load_vectorstore():
    """Charge l'index FAISS existant."""
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # pour FAISS sur disque
    )
    return vectorstore


# ====== Brique RAG ======

def retrieve_relevant_docs(question: str, k: int = 4):
    """Fait la recherche sémantique dans FAISS et renvoie les meilleurs chunks."""
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    return docs


def build_context_from_docs(docs):
    """Construit un gros contexte texte à partir des docs récupérés."""
    context_parts = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page", meta.get("page_num", "?"))
        context_parts.append(
            f"[Source {i+1} | {file_name} | page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)
    return context


def call_llm_with_openrouter(question: str, context: str) -> str:
    """
    Appelle le LLM via OpenRouter (moonshotai/kimi-k2:free) avec un prompt RAG.
    """
    # Tu peux personnaliser ces meta-infos pour le ranking openrouter
    extra_headers = {
        "HTTP-Referer": "https://litteria.local",  # par ex. nom du projet
        "X-Title": "Litteria - Academic RAG",
    }

    system_prompt = (
        "Tu es un assistant académique. "
        "Tu dois répondre uniquement à partir des SOURCES fournies ci-dessous. "
        "Si l'information n'est pas présente ou insuffisante, tu dois dire que tu ne sais pas.\n\n"
        "Pour chaque idée importante, cite la source au format (auteur, année, page si disponible).\n"
        "Ne fabrique pas de références."
    )

    user_content = (
        f"Question de l'utilisateur :\n{question}\n\n"
        f"SOURCES (extraits d'articles) :\n{context}\n\n"
        "Réponds de manière structurée, en français, "
        "et ajoute une section 'Références utilisées' à la fin."
    )

    completion = client.chat.completions.create(
        extra_headers=extra_headers,
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return completion.choices[0].message.content


def answer_question(question: str, k: int = 4) -> str:
    """
    Pipeline complet :
    - retrieve depuis FAISS
    - construire le contexte
    - appeler le LLM
    - renvoyer la réponse finale
    """
    docs = retrieve_relevant_docs(question, k=k)

    if not docs:
        return "Je n'ai trouvé aucune source pertinente pour répondre à cette question."

    context = build_context_from_docs(docs)
    answer = call_llm_with_openrouter(question, context)
    return answer


# ====== Test CLI ======

if __name__ == "__main__":
    q = input("Pose ta question : ")
    print("[QUESTION]", q)
    print()
    response = answer_question(q, k=4)
    print("[RÉPONSE]\n")
    print(response)

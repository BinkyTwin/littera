# src/app.py
import os
from pathlib import Path

# Fix OpenMP (même problème que dans les autres scripts)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv

# On importe les fonctions dont on a besoin depuis ton pipeline RAG
from rag_pipeline import (
    retrieve_relevant_docs,
    build_context_from_docs,
    call_llm_with_openrouter,
)

# ==== Chargement .env ====

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ==== UI Streamlit ====

st.set_page_config(
    page_title="Litteria - Assistant académique",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Litteria – Assistant de veille & rédaction scientifique")
st.write(
    "Pose ta question sur tes articles scientifiques. "
    "Le système va chercher dans ton corpus et répondre **avec les sources**."
)

# Zone de saisie
question = st.text_input("❓ Ta question :", value="", placeholder="Ex : Qu'est-ce que la data governance selon Otto et Khatri ?")
top_k = st.slider("Nombre de passages à récupérer (k)", min_value=2, max_value=8, value=4)

if st.button("Lancer la recherche") and question.strip():
    st.markdown(f"### 🔎 Question\n> {question}")
    with st.spinner("Recherche des passages pertinents dans tes PDFs..."):
        docs = retrieve_relevant_docs(question, k=top_k)

    if not docs:
        st.warning("Aucune source pertinente trouvée dans l'index.")
    else:
        # Afficher les sources brutes
        st.markdown("### 📑 Sources trouvées")
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            file_name = meta.get("file_name", "unknown")
            page = meta.get("page", meta.get("page_num", "?"))
            with st.expander(f"Source {i+1} – {file_name} (page {page})"):
                st.write(doc.page_content)

        # Construire le contexte pour le LLM
        context = build_context_from_docs(docs)

        with st.spinner("Génération de la réponse (LLM via OpenRouter)..."):
            answer = call_llm_with_openrouter(question, context)

        st.markdown("### 🧠 Réponse proposée")
        st.write(answer)

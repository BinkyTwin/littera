# src/app.py
import os
from pathlib import Path
# Fix OpenMP (même problème que dans les autres scripts)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from dotenv import load_dotenv
# On importe les fonctions dont on a besoin depuis ton pipeline RAG
from rag_pipeline import (
    load_vectorstore,
    build_context_from_docs,
    call_llm_with_openrouter,
    embeddings,          # même embeddings que pour l’index de base
)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ==== Chargement .env ====
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env") 

# ==== Gestion de l'index vectoriel en session ====

def get_or_create_vectorstore():
    """
    Charge l'index FAISS de base (créé via ingest.py + build_index.py)
    et le garde en cache dans la session Streamlit.
    """
    if "vectorstore" not in st.session_state:
        try:
            st.session_state["vectorstore"] = load_vectorstore()
        except Exception as e:
            st.warning("Impossible de charger l'index existant. "
                       "Vous pouvez quand même indexer des PDF via l'interface.")
            st.session_state["vectorstore"] = None
    return st.session_state["vectorstore"]


def add_uploaded_pdfs_to_index(uploaded_files):
    """
    Sauvegarde les fichiers uploadés, les ingère (chunking) et
    les ajoute à l'index FAISS déjà présent en session.
    """
    if not uploaded_files:
        return

    # éviter d'indexer deux fois les mêmes fichiers dans la même session
    if "uploaded_filenames" not in st.session_state:
        st.session_state["uploaded_filenames"] = []

    new_files = [f for f in uploaded_files if f.name not in st.session_state["uploaded_filenames"]]
    if not new_files:
        return

    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for f in new_files:
        temp_path = upload_dir / f.name
        # sauvegarde sur disque pour que PyMuPDF puisse lire
        with open(temp_path, "wb") as out:
            out.write(f.read())

        loader = PyMuPDFLoader(str(temp_path))
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["file_name"] = f.name
        docs.extend(pdf_docs)

    # Chunking identique au pipeline offline
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Index spécifique aux fichiers uploadés
    upload_vs = FAISS.from_documents(chunks, embeddings)

    # Fusion avec l'index de base (ou création d’un nouvel index si None)
    base_vs = get_or_create_vectorstore()
    if base_vs is None:
        st.session_state["vectorstore"] = upload_vs
    else:
        base_vs.merge_from(upload_vs)
        st.session_state["vectorstore"] = base_vs

    st.session_state["uploaded_filenames"].extend([f.name for f in new_files])

# ==== Configuration de la page ====
st.set_page_config(
    page_title="Littera - Assistant académique RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==== CSS Custom - Design académique professionnel ====
st.markdown("""
<style>
    /* Import Google Fonts - Police académique */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ==================== FOND & STRUCTURE ==================== */
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #4c1d95 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Container principal avec effet glassmorphism */
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem;
    }
    
    /* ==================== HEADER LITTERA ==================== */
    .littera-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .littera-logo {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }

    .littera-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .littera-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 0.8rem;
    }

    .littera-description {
        color: #94a3b8;
        font-size: 0.95rem;
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Badge RAG */
    .rag-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* ==================== ZONE DE RECHERCHE ==================== */
    .search-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Labels des inputs */
    label, .stTextInput label, .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }
    
    /* Input de recherche */
    .stTextInput > div > div > input {
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.95);
        color: #1e293b;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
        background: white;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* Slider personnalisé */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Bouton de recherche */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 1.05rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.5);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(59, 130, 246, 0.7);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* ==================== QUESTION POSÉE ==================== */
    .question-display {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    .question-display strong {
        color: #60a5fa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .question-text {
        color: #e2e8f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* ==================== SOURCES TROUVÉES ==================== */
    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Caption compteur de sources */
    .stCaption, .element-container div[data-testid="caption"] {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-style: italic;
        margin-bottom: 1rem !important;
    }
    
    /* Expanders (sources) */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: #3b82f6 !important;
        transform: translateX(4px);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 1.5rem;
    }
    
    /* Contenu des sources */
    .source-content {
        background: rgba(15, 23, 42, 0.6);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #3b82f6;
        color: #cbd5e1;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    /* ==================== RÉPONSE GÉNÉRÉE ==================== */
    .response-container {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
    }
    
    .response-container h3 {
        color: #60a5fa !important;
        margin-top: 0 !important;
    }
    
    .response-text {
        color: #e2e8f0;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    /* ==================== WARNING & ALERTS ==================== */
    .stWarning {
        background: rgba(251, 191, 36, 0.15) !important;
        border-left: 4px solid #fbbf24 !important;
        border-radius: 8px;
        color: #fde68a !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* ==================== FOOTER ==================== */
    .littera-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0 0.3rem;
        font-weight: 500;
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563eb, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)

# ==== HEADER LITTERA ====
st.markdown("""
<div class="littera-header">
    <div class="littera-logo">📚</div>
    <h1 class="littera-title">LITTERA</h1>
    <p class="littera-subtitle">Assistant académique basé sur le RAG</p>
    <p class="littera-description">
        Résumés, comparaisons d'auteurs, citations fiables. Littera permet de résumer des articles scientifiques, 
        répondre à des questions complexes, extraire des citations vérifiables, et comparer des auteurs en s'appuyant 
        exclusivement sur les sources chargées.
    </p>
    <div class="rag-badge">🧠 Retrieval-Augmented Generation • No Hallucination Policy</div>
</div>
""", unsafe_allow_html=True)

# ==== UPLOAD DE PDF ====
st.markdown("### 📥 Importer des articles scientifiques (PDF)")
uploaded_files = st.file_uploader(
    "Ajoutez ici vos articles (PDF) pour les intégrer au corpus de Littera.",
    type=["pdf"],
    accept_multiple_files=True,
    help="Les documents uploadés seront indexés pour cette session uniquement.",
)

if uploaded_files:
    if st.button("📚 Indexer les documents uploadés", use_container_width=True):
        add_uploaded_pdfs_to_index(uploaded_files)
        st.success(f"{len(uploaded_files)} document(s) ajouté(s) à l'index pour cette session.")

# ==== ZONE DE RECHERCHE ====
st.markdown('<div class="search-section">', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "🔍 Votre question académique", 
        value="", 
        placeholder="Ex : Quelles sont les principales contributions d'Otto et Khatri sur la data governance ?",
        label_visibility="visible",
        key="question_input"
    )

with col2:
    top_k = st.slider(
        "📊 Sources à analyser", 
        min_value=2, 
        max_value=8, 
        value=4,
        key="top_k_slider",
        help="Nombre de passages pertinents à récupérer dans le corpus"
    )

search_button = st.button("🚀 Lancer la recherche RAG", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==== TRAITEMENT DE LA RECHERCHE ====
if search_button and question.strip():
    
    # Afficher la question
    st.markdown(f"""
    <div class="question-display">
        <strong>🔎 Question posée</strong>
        <div class="question-text">{question}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recherche des documents
    with st.spinner("🔍 Recherche des passages pertinents dans le corpus indexé (FAISS)..."):
        vectorstore = get_or_create_vectorstore()
        if vectorstore is None:
            docs = []
        else:
            docs = vectorstore.similarity_search(question, k=top_k)
    
    if not docs:
        st.warning("⚠️ Aucune source pertinente trouvée dans l'index. Essayez de reformuler votre question ou d'ajouter des documents.")
    else:
        # Afficher les sources
        st.markdown("### 📑 Sources trouvées")
        st.caption(f"*{len(docs)} passage(s) pertinent(s) identifié(s) dans votre corpus*")
        
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            file_name = meta.get("file_name", "unknown")
            page = meta.get("page", meta.get("page_num", "?"))
            
            with st.expander(f"📄 Source {i+1} – {file_name} (page {page})"):
                st.markdown(f"""
                <div class="source-content">
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)
        
        # Construire le contexte pour le LLM
        context = build_context_from_docs(docs)
        
        # Génération de la réponse
        with st.spinner("🧠 Génération de la réponse par le LLM (OpenRouter)..."):
            answer = call_llm_with_openrouter(question, context)
        
        # Afficher la réponse
        st.markdown("""
        <div class="response-container">
            <h3>🤖 Réponse générée</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="response-text">
            {answer}
        </div>
        """, unsafe_allow_html=True)

# ==== FOOTER ====
st.markdown("---")
st.markdown("""
<div class="littera-footer">
    <div style="margin-bottom: 0.5rem;">
        <span class="footer-badge">FAISS Vector Search</span>
        <span class="footer-badge">OpenRouter LLM</span>
        <span class="footer-badge">Sentence Transformers</span>
    </div>
    Projet EMLV • Littera RAG System • 2024-2025
</div>
""", unsafe_allow_html=True)

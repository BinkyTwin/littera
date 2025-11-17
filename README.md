# 📚 LITTERIA

### *Assistant académique basé sur le RAG — Résumés, comparaisons d’auteurs, citations fiables.*

Litteria est un **assistant intelligent dédié à la recherche académique**.
Il permet de **résumer des articles scientifiques**, **répondre à des questions complexes**, **extraire des citations vérifiables**, et **comparer des auteurs** en s’appuyant exclusivement sur les sources chargées par l’utilisateur.

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)** :
➡ extraction de passages pertinents dans des PDF → génération de réponse par un LLM OpenRouter (DeepSeek/Moonshot ou autre) → citations vérifiées → aucune hallucination.

Développé dans le cadre d’un projet EMLV.

---

# 🚀 Fonctionnalités principales

### 🧠 **1. Recherche académique assistée (RAG)**

* Pose une question → Litteria recherche dans les PDF indexés.
* Réponses **structurées**, **en français**, **avec sources obligatoires**.
* Pas de sources → pas de réponse (no hallucination policy).

### 📑 **2. Exploration du corpus**

* Affichage des passages exacts utilisés (file_name + page).
* Inspection du texte chunké depuis vos PDFs.

### 📊 **3. Interface simple (Streamlit)**

* Interface web minimaliste.
* Input question + sliders.
* Réponse + sources dans des panels extensibles.

### 📝 **4. Ingestion intelligente des documents**

* PDFs découpés en chunks (400–800 tokens).
* Métadonnées : auteur, année, page, fichier.
* Indexation FAISS pour une recherche très rapide.

---

# 🏗️ Architecture technique

### 🔍 **Ingestion & Vectorisation**

* Parsing PDF : *PyMuPDF*
* Chunking : *LangChain Text Splitters*
* Embeddings : *OpenAIEmbeddings*
* Stockage : *FAISS* (index vectoriel local)

### 🤖 **LLM / Génération**

* LLM via OpenRouter (DeepSeek, Moonshot, GPT, etc.)
* Client OpenAI configuré en :

```
base_url = "https://openrouter.ai/api/v1"
api_key  = OPENROUTER_API_KEY
```

### 🖥️ **Front**

* *Streamlit*
* Résultats lisibles + inspection des sources

Architecture RAG complète :
**PDF → chunks → embeddings → FAISS → retrieval → LLM (OpenRouter)**

---

# 📦 Installation

### 1. Cloner le repo

```bash
git clone https://github.com/tonrepo/litteria.git
cd litteria
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Créer un fichier `.env`

```env
OPENAI_API_KEY=ta_clé_openai_si_embeddings
OPENROUTER_API_KEY=ta_clé_openrouter
```

### 4. Construire l’index FAISS

Place tes PDFs dans :
`data/raw/`

Puis lance :

```bash
python src/ingest.py
python src/build_index.py
```

### 5. Lancer l’app

```bash
streamlit run src/app.py
```

---

# 🧪 Structure du projet

```
litteria/
│
├── data/
│   ├── raw/              # PDFs déposés ici
│   └── processed/
│       └── index/        # Index FAISS
│
├── src/
│   ├── ingest.py         # Extraction & chunking
│   ├── build_index.py    # Embeddings + FAISS
│   ├── rag_pipeline.py   # RAG complet (retrieval + LLM)
│   └── app.py            # Interface Streamlit
│
├── .env                  # Clés API
└── requirements.txt
```

---

# 🎯 Pourquoi Litteria ?

Parce que les étudiants (et chercheurs) ont besoin d’un assistant qui :

* **ne fabrique pas de citations**
* **ne hallucine pas**
* **explique clairement**
* **travaille à partir de leurs propres sources**
* **facilite la rédaction de mémoires et rapports**

Litteria répond **uniquement** à partir de vos documents → parfait pour la recherche académique.

---

# 📌 Améliorations prévues

* Recherche hybride BM25 + FAISS
* Mode comparaison d’auteurs
* Résumé automatique d’un PDF
* Export Word/BibTeX
* Upload direct depuis l’interface Streamlit
* Visualisation interactive des vecteurs

---

# ❤️ Crédits

Projet développé par Abdelatif Djeddou & Manissa Bouda, dans le cadre du programme EMLV.
Tech powered by **LangChain**, **FAISS**, **Streamlit**, **OpenRouter**.

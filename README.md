# 📄 RAG Document Chatbot (Groq + LangChain + FAISS)

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents
and ask natural-language questions about their content. Answers are grounded in
the actual text of your documents, with source document and page citations.

---

## Overview

The app extracts text from uploaded PDFs, splits it into chunks, embeds those
chunks, and stores the embeddings in a **FAISS** vector database. When you ask
a question, it retrieves the most relevant chunks and passes them as context to
an LLM served by **Groq's** low-latency inference API, which generates a
grounded answer. Conversation history is preserved so you can ask natural
follow-up questions.

```
PDF(s) → pdf_loader → text_splitter → embeddings → vector_store (FAISS)
                                                          │
User question ──► rag_chatbot (retrieve + condense + Groq LLM) ──► Answer + Sources
```

---

## Features

- 📤 Upload **one or more PDF documents** at once
- ✂️ Automatic text extraction and smart chunking (with overlap)
- 🔎 Semantic search over document chunks using FAISS
- ⚡ Fast, high-quality answers via the **Groq API** (e.g. Llama 3.3 70B)
- 💬 **Conversational memory** — follow-up questions are understood in context
- 📚 **Source citations** — every answer shows which file & page it came from
- 🖥️ Clean **Streamlit** web interface, plus an optional CLI (`main.py`)
- 🔁 Persist the vector index to disk and reload it on the next run
- 🗑️ One-click **clear & rebuild** of the vector database
- 🛡️ Graceful error handling: invalid file types, empty/scanned PDFs, missing
  API keys, and network failures all produce clear, actionable messages
  instead of crashes

---

## Technologies Used

| Purpose               | Technology                                             |
|------------------------|--------------------------------------------------------|
| Orchestration           | [LangChain](https://python.langchain.com/)             |
| LLM inference            | [Groq API](https://console.groq.com/) (Llama 3.3 70B)  |
| Embeddings               | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector database          | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| PDF parsing              | [pypdf](https://pypi.org/project/pypdf/)                |
| Web UI                   | [Streamlit](https://streamlit.io/)                      |
| Config / secrets         | `python-dotenv`                                          |

> **Why local embeddings?** Groq serves LLM chat completions only — it does not
> currently offer an embeddings endpoint. Using a local `sentence-transformers`
> model keeps embedding generation free, fast, and independent of an extra
> API key.

---

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com/keys) and generate an API key.

### 5. Configure environment variables
```bash
cp .env.example .env
```
Then edit `.env` and add your key:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Required Dependencies

See [`requirements.txt`](requirements.txt) — key packages:
```
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-groq
langchain-huggingface
langchain-core
faiss-cpu
pypdf
sentence-transformers
python-dotenv
groq
```

---

## How to Run

### Web interface (recommended)
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`):
1. Upload one or more PDFs in the sidebar.
2. Click **Process documents**.
3. Ask questions in the chat box at the bottom.
4. Expand **Sources** under any answer to see which file/page it came from.
5. Use **Clear database** to wipe the index and start fresh, or **Clear chat
   history** to reset the conversation without losing the index.

### Command-line interface
```bash
python main.py path/to/document1.pdf path/to/document2.pdf
```
Then type questions at the `You:` prompt. Type `exit` or `quit` to stop.
Running `python main.py` with no arguments will try to reload a previously
saved index (`faiss_index/`) if one exists.

---

## Sample Questions

Using the included `sample_pdfs/sample.pdf` (or your own documents), try:

- "What is this document about?"
- "Summarize the key points in this PDF."
- "What does the document say about [specific topic]?"
- "Which page discusses [topic]?"
- "Can you list the main points as bullet points?"
- Follow-up: "Can you explain that in simpler terms?" *(tests conversational memory)*
- "Is [something not in the document] mentioned anywhere?" *(tests that the bot says "not found" rather than hallucinating)*

---

## Error Handling

The app is designed to fail gracefully and explain what went wrong, rather than crash:

| Situation                                | Behavior                                                        |
|--------------------------------------------|-------------------------------------------------------------------|
| Non-PDF file uploaded                      | Rejected with a clear "not a PDF file" message                    |
| Corrupted / password-protected PDF          | Reported per-file; other valid files in the same batch still process |
| Scanned PDF with no extractable text        | Clear error explaining OCR isn't supported                        |
| Missing `GROQ_API_KEY`                       | App shows a configuration error with setup instructions instead of crashing |
| Network / Groq API failure                    | Caught and surfaced as a readable chat error message               |
| Empty question submitted                       | Rejected before any API call is made                              |

---

## Notes & Limitations

- Scanned/image-only PDFs are not supported (no OCR) — use text-based PDFs.
- The FAISS index is persisted locally under `faiss_index/` (excluded from
  git via `.gitignore`); delete this folder or use **Clear database** to reset.
- The first run downloads the embedding model from HuggingFace (a few hundred
  MB); subsequent runs use the local cache.

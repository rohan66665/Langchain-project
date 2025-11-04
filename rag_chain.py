# rag_chain.py
# Paste this full file (replace your current rag_chain.py)
# Works with your installed venv packages (langchain_groq, langchain_community FAISS, langchain_huggingface, groq)

import os
import json
from dotenv import load_dotenv
from typing import List

# load installed wrappers
try:
    from langchain_groq import ChatGroq
except Exception:
    # If import fails, show clear error
    raise ImportError("langchain_groq not found. Make sure your venv has langchain-groq / langchain_groq installed.")

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    raise ImportError("langchain_community.vectorstores.FAISS not available. Install the correct langchain-community version.")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    raise ImportError("langchain_huggingface HuggingFaceEmbeddings not available. Install langchain-huggingface.")

try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    # Try alternative import path if needed
    from langchain.prompts import PromptTemplate

# -----------------------
# Config & env
# -----------------------
load_dotenv()  # loads .env in project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
VECTORSTORE_FOLDER = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_FILE = "memory.json"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set. Add it to .env (GROQ_API_KEY=...) and restart.")

# -----------------------
# Initialize LLM (Groq)
# -----------------------
llm = ChatGroq(
    model=MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.0,  # deterministic answers, change if you want creativity
    streaming=False
)

# -----------------------
# Load vectorstore & retriever
# -----------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if not os.path.isdir(VECTORSTORE_FOLDER):
    raise RuntimeError(f"Vectorstore folder not found: {VECTORSTORE_FOLDER}. Run ingest.py first.")

vectorstore = FAISS.load_local(
    folder_path=VECTORSTORE_FOLDER,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# create retriever with reasonable defaults
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Safe helper to get documents from retriever (works across versions)
def retrieve_docs(query: str):
    # prefer high-level API if available
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    # fallback name variations
    if hasattr(retriever, "get_relevant_entries"):
        return retriever.get_relevant_entries(query)
    # last-resort: private method (some versions)
    if hasattr(retriever, "_get_relevant_documents"):
        # some versions require run_manager kw, pass None
        try:
            return retriever._get_relevant_documents(query)
        except TypeError:
            return retriever._get_relevant_documents(query, run_manager=None)
    raise RuntimeError("Retriever object does not expose a known retrieval method.")

# -----------------------
# Simple persistent memory
# -----------------------
def load_memory() -> List[dict]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(mem: List[dict]):
    with open(MEMORY_FILE, "w", encoding="utf8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

memory = load_memory()

# -----------------------
# Prompt template: uses context + conversation history
# -----------------------
template = """You are an assistant. Use ONLY the provided context & conversation to answer the user's question.
Conversation history (most recent last):
{history}

Context (retrieved docs):
{context}

User question:
{question}

Provide a concise answer and mention the sources briefly (if any).
"""

prompt = PromptTemplate.from_template(template)

# -----------------------
# RAG query function
# -----------------------
def rag_query(question: str) -> str:
    # 1) retrieve
    docs = retrieve_docs(question) or []
    context = "\n\n".join([f"Source: {getattr(d, 'metadata', {}).get('source', '')}\n{d.page_content}" for d in docs])

    # 2) build conversation history
    hist_text = ""
    for turn in memory[-10:]:  # keep last 10 for prompt
        hist_text += f"User: {turn.get('user')}\nAssistant: {turn.get('assistant')}\n\n"

    # 3) format prompt
    final_prompt = prompt.format(history=hist_text, context=context or "No context found.", question=question)

    # 4) call LLM
    response = llm.invoke(final_prompt)

    # 5) extract text safely
    text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)

    # 6) save to persistent memory (append)
    memory.append({"user": question, "assistant": text})
    # keep memory length reasonable
    if len(memory) > 200:
        memory[:] = memory[-200:]
    save_memory(memory)

    return text

# -----------------------
# CLI loop
# -----------------------
def main_cli():
    print("✅ Smart RAG Chatbot is ready with memory!")
    print("   Commands: 'clear' to reset memory, 'exit' to quit.\n")
    while True:
        query = input("🧠 Ask (or 'clear'/'exit'): ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "clear":
            memory.clear()
            save_memory(memory)
            print("Memory cleared.")
            continue
        try:
            ans = rag_query(query)
            print("\n💬 Answer:", ans, "\n")
        except Exception as e:
            print("Error during query:", str(e))

# -----------------------
# Optional: function to call from web server
# -----------------------
def answer_for_web(question: str) -> dict:
    """Return JSON-friendly answer (for FastAPI etc.)."""
    try:
        txt = rag_query(question)
        return {"answer": txt, "ok": True}
    except Exception as e:
        return {"answer": str(e), "ok": False}

if __name__ == "__main__":
    main_cli()

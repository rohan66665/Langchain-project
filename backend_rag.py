import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Put it in .env as GROQ_API_KEY=...")

# ── LLM ────────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model=MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
)

# ── Vectorstore / Retriever ────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── Simple in-memory chat history ──────────────────────────────────────────────
_CONV_HISTORY: List[Dict[str, str]] = []  # [{"role":"user"/"assistant","content":"..."}]

def reset_memory():
    _CONV_HISTORY.clear()

def _history_text(last_n: int = 6) -> str:
    # convert last N messages to a readable chat transcript
    msgs = _CONV_HISTORY[-last_n:]
    lines = []
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

# ── Prompt ─────────────────────────────────────────────────────────────────────
template = """You are a helpful assistant. Use the retrieved context to answer.
If the context does not contain the answer, say "I don't know from the documents" and then give a short general answer if you can (but clearly mark it as general).

Conversation so far:
{history}

Retrieved context:
{context}

User question:
{question}

Answer:
"""
PROMPT = PromptTemplate.from_template(template)

# ── Public chat() API ──────────────────────────────────────────────────────────
def chat(question: str) -> Dict[str, Any]:
    # 1) retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join([f"- {d.page_content.strip()}" for d in docs]) if docs else "None"

    # 2) craft prompt with memory
    prompt_text = PROMPT.format(history=_history_text(), context=context, question=question)

    # 3) call LLM
    resp = llm.invoke(prompt_text)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    # 4) update memory
    _CONV_HISTORY.append({"role": "user", "content": question})
    _CONV_HISTORY.append({"role": "assistant", "content": answer})

    # 5) build simple sources
    sources = []
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        sources.append({
            "source": meta.get("source") or meta.get("file") or "document",
            "snippet": d.page_content[:180].replace("\n", " ")
        })

    return {"answer": answer, "sources": sources}

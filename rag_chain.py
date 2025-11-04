import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------
# Setup
# -----------------------
load_dotenv()  # reads .env in project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")
if not MODEL_NAME:
    raise RuntimeError("MODEL_NAME missing in .env")

# LLM (no streaming — avoids older callback import paths)
llm = ChatGroq(
    model=MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.5,
)

# Vector store (created earlier by ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# -----------------------
# Hybrid RAG helpers
# -----------------------
def retrieve_with_scores(query: str, k: int = 4) -> List[Tuple[str, float]]:
    """
    Try to retrieve with scores. Falls back to plain retrieve if method not available.
    Returns list of (text, score_or_distance). Lower distance is better for FAISS L2.
    """
    texts_scores: List[Tuple[str, float]] = []
    try:
        # FAISS usually supports similarity_search_with_score -> (Document, distance)
        docs_scores = vectorstore.similarity_search_with_score(query, k=k)
        for doc, score in docs_scores:
            texts_scores.append((doc.page_content, float(score)))
    except Exception:
        # fallback: no explicit scores, just get docs
        docs = vectorstore.similarity_search(query, k=k)
        for d in docs:
            texts_scores.append((d.page_content, 0.0))
    return texts_scores

def pick_context(texts_scores: List[Tuple[str, float]]) -> str:
    """
    Heuristic: if we have scores (distances), keep those that look 'close'.
    For FAISS L2, smaller = closer. Threshold is empirical; you can tune if needed.
    If we don’t have meaningful scores, we still use the top-k.
    """
    if not texts_scores:
        return ""
    # Separate ones that came with real scores (distance > 0) vs 0.0 fallback
    have_real_scores = any(s > 0 for _, s in texts_scores)
    if have_real_scores:
        # Keep items with distance <= 1.2 (tune if needed)
        filtered = [t for t, s in texts_scores if s <= 1.2]
        if filtered:
            return "\n\n".join(filtered)
        # If nothing passed threshold, treat as no reliable context
        return ""
    else:
        # No scores available — just join top-k
        return "\n\n".join(t for t, _ in texts_scores)

# -----------------------
# Prompt building (with memory)
# -----------------------
SYSTEM_RAG = """You are a helpful AI assistant.

You have two knowledge sources:
1) Retrieved context (if provided) from the user's documents.
2) Your own general knowledge.

RULES:
- If context is present and relevant, use it and cite with: (from docs).
- If context is empty or irrelevant, answer from general knowledge and say: (general).
- Be concise, accurate, and helpful. If unsure, say you’re unsure.
"""

def build_prompt(chat_history: List[Tuple[str, str]], user_query: str, context: str) -> str:
    # format the last few turns of history
    history_str = ""
    for u, a in chat_history[-5:]:
        history_str += f"User: {u}\nAssistant: {a}\n"

    return f"""{SYSTEM_RAG}

Chat history:
{history_str if history_str.strip() else "(none)"}

Retrieved context:
{context if context.strip() else "(none)"}

User question:
{user_query}

Assistant:
"""

# -----------------------
# Main loop with memory
# -----------------------
def answer(chat_history: List[Tuple[str, str]], user_query: str) -> str:
    # 1) retrieve
    texts_scores = retrieve_with_scores(user_query, k=4)
    context = pick_context(texts_scores)

    # 2) build prompt with memory + context
    prompt = build_prompt(chat_history, user_query, context)

    # 3) ask LLM
    result = llm.invoke(prompt)
    reply = result.content if hasattr(result, "content") else str(result)

    # 4) tag source
    if context.strip():
        reply += "\n\n(from docs)"
    else:
        reply += "\n\n(general)"

    return reply

if __name__ == "__main__":
    print("\n✅ Hybrid RAG Chatbot (Docs + General) with Memory is ready!")
    print("   Commands: 'clear' to reset memory, 'exit' to quit.\n")

    memory: List[Tuple[str, str]] = []  # list of (user, assistant)

    while True:
        try:
            q = input("🧠 Ask (or 'clear'/'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break
        if q.lower() == "clear":
            memory.clear()
            print("🧹 Memory cleared.\n")
            continue

        ans = answer(memory, q)
        print("\n💬 Answer:", ans, "\n")
        memory.append((q, ans))

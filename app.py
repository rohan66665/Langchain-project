from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend_rag import chat as rag_chat, reset_memory as rag_reset

# ✅ Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot (Groq API)",
    version="1.0.0",
    description="Conversational RAG Chatbot powered by LangChain + Groq"
)

# ✅ Enable CORS for all origins (safe for Vercel/Render frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input model for POST /chat endpoint
class ChatRequest(BaseModel):
    message: str

# ✅ Health check endpoint for Render
@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG Chatbot is healthy 🟢"}

# ✅ Main chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = rag_chat(req.message)
        return {"reply": response}
    except Exception as e:
        return {"error": str(e)}

# ✅ Reset chat memory endpoint
@app.post("/reset")
def reset():
    try:
        message = rag_reset()
        return {"status": message}
    except Exception as e:
        return {"error": str(e)}

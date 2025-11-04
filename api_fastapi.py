from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware   # <-- ye import upar rakho
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv


# ✅ pehle FastAPI app banao
app = FastAPI(title="Rohan 2.0 AI Backend")

# ✅ ab CORS middleware add karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Verify API key exists
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in environment variables!")

# ✅ Initialize model safely
try:
    llm = ChatGroq(model="llama-3.1-8b-instant")  # <-- define llm yahan
except Exception as e:
    print("Error initializing ChatGroq:", e)
    llm = None


# ✅ Request body model
class ChatRequest(BaseModel):
    text: str


# ✅ API route
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    if not llm:
        return {"error": "Groq model not initialized properly."}
    try:
        user_message = request.text
        response = llm.invoke([HumanMessage(content=user_message)])
        return {"response": response.content}
    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

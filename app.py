from fastapi import FastAPI
from pydantic import BaseModel
from backend_rag import chat, load_document, reset_memory

app = FastAPI()

class Query(BaseModel):
    question: str

class Doc(BaseModel):
    text: str

@app.post("/chat")
def ask_question(data: Query):
    return {"response": chat(data.question)}

@app.post("/upload")
def upload_docs(data: Doc):
    load_document(data.text)
    return {"message": "Document uploaded and processed ✅"}

@app.post("/reset")
def clear_memory():
    return {"message": reset_memory()}

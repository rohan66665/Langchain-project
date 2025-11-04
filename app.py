import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from backend_rag import chat, reset_memory

load_dotenv()

app = FastAPI(title="RAG + Memory Chat")

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatIn(BaseModel):
    message: str
    reset: bool | None = False

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_api(payload: ChatIn):
    try:
        if payload.reset:
            reset_memory()
            return JSONResponse({"answer": "Memory cleared.", "sources": []})
        result = chat(payload.message)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"answer": f"Error: {e}", "sources": []}, status_code=500)

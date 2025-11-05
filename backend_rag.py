import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# ✅ Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please add it in Render → Environment Variables")

# ✅ Initialize embeddings (lightweight, no torch GPU needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Vector database setup
def load_and_index_pdf(pdf_path="docs/sample.pdf"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore

# ✅ Load vectorstore if already created
def get_vectorstore():
    if os.path.exists("vectorstore"):
        return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return load_and_index_pdf()

# ✅ Initialize Groq model
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.3
)

# ✅ Create RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=get_vectorstore().as_retriever()
)

# ✅ Chat function
chat_history = []

def chat(query):
    global chat_history
    if not query.strip():
        return "⚠️ Please enter a question."
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

# ✅ Reset memory
def reset_memory():
    global chat_history
    chat_history = []
    return "🧠 Chat memory cleared!"

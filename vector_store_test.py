# vector_store_test.py

from dotenv import load_dotenv
import os

# Load environment variables (.env se API key load karega)
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- STEP 1: Initialize Embeddings ---
print("🔹 Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- STEP 2: Sample documents ---
documents = [
    "LangChain helps developers build LLM-powered applications.",
    "Groq provides fast inference for AI models.",
    "Rohan 2.0 is learning full LangChain integration.",
]

# --- STEP 3: Split text into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.create_documents(documents)

# --- STEP 4: Create FAISS vector store ---
print("🔹 Creating FAISS vector database...")
vectorstore = FAISS.from_documents(texts, embeddings)

# --- STEP 5: Initialize Groq LLM ---
print("🔹 Connecting to Groq model...")
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# --- STEP 6: Create simple prompt ---
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer based on this context:\n{context}\n\nQuestion: {question}"
)

# --- STEP 7: Run a test query ---
query = "What is LangChain?"
docs = vectorstore.similarity_search(query, k=2)
context = "\n".join([d.page_content for d in docs])

chain_input = {"context": context, "question": query}
result = llm.invoke(prompt.format(**chain_input))

print("\n🤖 Groq Model Response:\n", result.content)

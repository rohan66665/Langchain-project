import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

def load_document(text: str):
    global vector_store
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")

def chat(query: str):
    global vector_store
    if vector_store is None:
        return "⚠ Please upload documents first."

    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="mixtral-8x7b-32768"
    )
    
    prompt = f"Answer based on the context.\n\nContext:\n{context}\n\nUser: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response

def reset_memory():
    global vector_store
    vector_store = None
    if os.path.exists("faiss_index"):
        import shutil
        shutil.rmtree("faiss_index")
    return "Memory cleared ✅"

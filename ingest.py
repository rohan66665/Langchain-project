import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# Load documents from /data folder
def load_documents():
    text_loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = text_loader.load() + pdf_loader.load()
    print(f"✅ Loaded {len(docs)} documents from /data folder")
    return docs

# Split into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# Create embeddings & save vectorstore
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local("vectorstore")
    print("✅ Vectorstore with embeddings saved in /vectorstore")

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)

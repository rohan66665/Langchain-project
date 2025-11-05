import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# ✅ Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# ✅ Embeddings — lightweight (no torch / no GPU / no 700MB download)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ Load & Split Documents
def load_documents(pdf_folder="data"):
    documents = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
            documents.extend(loader.load())
    return documents

def create_vectorstore():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = load_documents()
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# ✅ Load existing vectorstore or create
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = create_vectorstore()

# ✅ Chat Model (Uses OpenAI, works fast, low memory)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# ✅ Conversation Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# ✅ Chat Function for API use
def chat(query):
    response = qa_chain({"question": query})
    return response["answer"]

# ✅ Reset Chat Memory
def reset_memory():
    memory.clear()
    return "✅ Memory cleared!"


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("⚠️ OPENAI_API_KEY not found. Check your .env file!")

# LangChain model setup
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Send message
response = chat.invoke([HumanMessage(content="Rohan 2.0 is learning LangChain, kya bolti duniya?")])

print("🤖 AI Response:")
print(response.content)

# main.py
from rag_chain import ask_groq

if __name__ == "__main__":
    print("🚀 Testing LangChain + Groq connection...")
    user_prompt = "Hello from Rohan 2.0! Is my LangChain + Groq integration working fine?"
    response = ask_groq(user_prompt)
    print("\n✅ Response from Groq model:\n")
    print(response)


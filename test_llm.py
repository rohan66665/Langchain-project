import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# ✅ .env file load karne ke liye
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("⚠️ No OpenAI API key found in .env file.")
else:
    print("✅ OpenAI API key loaded successfully!")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello from Rohan 2.0’s LangChain project!"}
    ]
)

print(response.choices[0].message.content)

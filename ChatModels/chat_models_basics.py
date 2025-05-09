from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

gpt_api = os.getenv("OPENAI_API_KEY")
google_api = os.getenv("GOOGLEAI_API")
os.getenv("HUGGINGFACEHUB_API_TOKEN")


gpt_model = ChatOpenAI(
    model = "gpt-3.5-turbo",
    api_key=gpt_api
)
"""
gpt_result = gpt_model.invoke("hello world")

print(gpt_result.content)"""

gemini_model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key=google_api
)


gemini_result = gemini_model.invoke("hello world")
print(gemini_result.content)

llama_model = ChatOllama(
    model="llama3.2:3b"
)


llama_result = llama_model.invoke("hello world")
print(llama_result.content)


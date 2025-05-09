from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , HumanMessage 
import os
from dotenv import load_dotenv

load_dotenv()

google_api = os.getenv("GOOGLEAI_API")


gemini_model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key=google_api
)

gemini_messages = [
    SystemMessage(content="You are professional python developar , that solve python tasks"),
    HumanMessage(content="write a python code that implement a linked list"),
]

gemini_result = gemini_model.invoke(gemini_messages)
print(gemini_result.content)

llama_model = ChatOllama(
    model="llama3.2:3b"
)

llama_messages = [
    SystemMessage(content="You are professional python developar , that solve python tasks"),
    HumanMessage(content="write a python code that implement a linked list"),
]

llama_result = llama_model.invoke(llama_messages)
print(llama_result.content)



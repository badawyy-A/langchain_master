from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , HumanMessage 
import os
from dotenv import load_dotenv

load_dotenv()

google_api = os.getenv("GOOGLEAI_API")


def interactive_with_gemini(model:str,api_key:str,system_message:str) -> list :
    
    chat_history = [SystemMessage(content=system_message)]

    gemini = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key
    )

    while True:
        query = HumanMessage(content=input("You: "))
        chat_history.append(query)
        if query.content.lower() == "exit":
            break
        
        response = gemini.invoke(chat_history)

        chat_history.append(response)
        print(f"AI: {response.content}")

    return chat_history



def interactive_with_llama(model:str,system_message:str) -> list :
    
    chat_history = [SystemMessage(content=system_message)]

    llama = ChatOllama(
        model=model,
    )

    while True:
        query = HumanMessage(content=input("You: "))
        chat_history.append(query)
        if query.content.lower() == "exit":
            break
        
        response = llama.invoke(chat_history)

        chat_history.append(response)
        print(f"AI: {response.content}")

    return chat_history

gemini_result = interactive_with_gemini(
    model="gemini-2.0-flash",
    api_key=google_api,
    system_message="You are professional python developar , that solve python tasks"
)

llama_result = interactive_with_llama(
    model="llama3.2:3b",
    system_message="You are professional python developar , that solve python tasks"
)

print(llama_result)
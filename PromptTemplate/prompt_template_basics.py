from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api = os.getenv("GOOGLEAI_API")

template = "tell me a {subject} , about {topic}"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke(
    {
        "subject":"story",
        "topic":"egyptain"
    }
)

print(prompt)

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=gemini_api
)

template_result = gemini.invoke(prompt)

print(template_result.content)

chat = [
    ("system", "You are a professional {tech} developer, that writes professional code."),
    ("human", "This code '{user_code}' gives me this error '{user_error}'")
]

messages_template = ChatPromptTemplate.from_messages(chat)

messages = messages_template.invoke(
    {
        "tech":"python",
        "user_code":"print'hello world'",
        "user_error":"syntex error"
    }
)

print(messages)



messages_result = gemini.invoke(messages)

print(messages_result.content)
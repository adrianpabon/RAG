from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Config:
    def __init__(self):
        self.OPENAI_API_KEY = OPENAI_API_KEY
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.DOCUMENTS_DIRECTORY = os.path.join(base_dir, "docs_for_rag")
        self.PERSIST_DIRECTORY = os.path.join(base_dir, "vector_db")
        
    def chat_model(self):
        llm = ChatOpenAI(model="gpt-4o-mini", 
        temperature=0,
        api_key=self.OPENAI_API_KEY)
        return llm
        
config = Config()

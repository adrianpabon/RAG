import os
from dotenv import load_dotenv
from config import config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import PyPDF2

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
)

with open("prompt.txt", "r") as f:
    prompt = f.read()

def procesar_documentos():
    if not os.path.exists(config.DOCUMENTS_DIRECTORY):
        return f"Error: El directorio {config.DOCUMENTS_DIRECTORY} no existe"

    documents = []
    for file in os.listdir(config.DOCUMENTS_DIRECTORY):
        if file.endswith(".pdf"):
            with open(os.path.join(config.DOCUMENTS_DIRECTORY, file), "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                documents.append(Document(
                            page_content=text, 
                            metadata={"source": file,
                                    
                            }))

        print(documents)

    if not documents:
        return f"Error: No se encontraron documentos en el directorio {config.DOCUMENTS_DIRECTORY}"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(
        texts,
        EMBEDDINGS,
        persist_directory=config.PERSIST_DIRECTORY
    )
    return db

def generar_respuesta(pregunta: str):
    
    if not os.path.exists(config.PERSIST_DIRECTORY):
        return f"Error: El directorio {config.PERSIST_DIRECTORY} no existe"

    db = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=EMBEDDINGS)

    retriever = db.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={
        "k": 4,
        "fetch_k": 10  # Busca 10, devuelve los 4 más diversos
    }
)

    consulta = retriever.invoke(pregunta)
    print(consulta)
    print("--------------------------------")

    llm = config.chat_model()

    respuesta = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Contexto:\n{consulta}\n\nPregunta: {pregunta}"}
    ])

    
    return respuesta.content

if __name__ == "__main__":
    #procesar_documentos() # descomentar para procesar documentos
    respuesta = generar_respuesta("Cómo hago para participar en el programa?")
    print(respuesta)



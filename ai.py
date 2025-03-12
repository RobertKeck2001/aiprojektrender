import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Lade Umgebungsvariablen
load_dotenv(".env")
print("Existiert die Datei?", os.path.exists(".env"))
print("AICORE_BASE_URL:", os.getenv("AICORE_BASE_URL"))
print("AICORE_CLIENT_ID:", os.getenv("AICORE_CLIENT_ID"))
print("AICORE_CLIENT_SECRET:", os.getenv("AICORE_CLIENT_SECRET"))

# FastAPI App initialisieren
app = FastAPI()

# CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"]
)

# Modell zur Validierung von Anfragen
class QueryRequest(BaseModel):
    input: str

# Dokumente aus dem Ordner "Rag-Docs" laden
loader = DirectoryLoader("RAG-Docs", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Prüfen, ob Dokumente gefunden wurden
if not documents:
    raise ValueError("Keine Dokumente im Ordner 'Rag-Docs' gefunden!")

print(f"Gefundene Dokumente: {len(documents)}")


# Dokumente in Chunks aufteilen
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Embedding-Modell definieren und Datenbank erstellen
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-3-small')
db = Chroma.from_documents(texts, embedding_model)

# Abfrage-Retriever erstellen
retriever = db.as_retriever()

# Chat LLM initialisieren
chat_llm = ChatOpenAI(proxy_model_name='gpt-4o-mini')

# RetrievalQA erstellen
qa = RetrievalQA.from_llm(llm=chat_llm, retriever=retriever)

# **Root-Route hinzufügen**
@app.get("/")
async def root():
    """Gibt eine einfache Nachricht zurück, wenn die App läuft."""
    return {"message": "App läuft"}

# API-Endpunkt für Anfragen
@app.post("/query")
async def query_llm(request: QueryRequest):
    """Verarbeitet Anfragen von SAP Build Apps"""
    user_input = request.input

    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    response = qa.invoke(user_input)
    return {"response": response['result']}

# Uvicorn-Start (falls direkt ausgeführt)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))  # Render setzt den PORT automatisch
    print(f"Starte die App auf Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)

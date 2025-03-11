import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Lade Umgebungsvariablen
print("Existiert die Datei?", os.path.exists(".env"))
load_dotenv(".env")

print("AICORE_BASE_URL:", os.getenv("AICORE_BASE_URL"))
print("AICORE_CLIENT_ID:", os.getenv("AICORE_CLIENT_ID"))
print("AICORE_CLIENT_SECRET:", os.getenv("AICORE_CLIENT_SECRET"))

# FastAPI App initialisieren
app = FastAPI()

# Modell zur Validierung von Anfragen
class QueryRequest(BaseModel):
    input: str

# Dokumente laden
loader = TextLoader("Geburtstagsliste.txt")
documents = loader.load()

# Dokumente in Chunks aufteilen
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Embedding-Modell definieren und Datenbank erstellen
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-3-small')
db = Chroma.from_documents(texts, embedding_model)

# Abfrage-Retriever erstellen
retriever = db.as_retriever()

# ChatLLM erstellen
chat_llm = ChatOpenAI(proxy_model_name='gpt-4o-mini')

# QA-Instanz erstellen
qa = RetrievalQA.from_llm(llm=chat_llm, retriever=retriever)

# **Hinzufügen der Root-Route**
@app.get("/")
async def root():
    """Gibt eine einfache Nachricht zurück, wenn die App läuft."""
    return {"message": "App läuft"}

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
    # Setze den Port auf den Wert der Umgebungsvariable PORT oder 8000, falls nicht gesetzt
    port = int(os.environ.get('PORT')
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)




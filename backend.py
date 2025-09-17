# backend.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core import get_qa_chain
from functools import lru_cache

# Initialize the FastAPI app
app = FastAPI(
    title="HR Chatbot API",
    description="An API for interacting with the HR Policy RAG Chatbot.",
    version="1.0.0",
)

# Serve static files (e.g., favicon.ico) - create a 'static/' folder for this
app.mount("/static", StaticFiles(directory="static"), name="static")

# This Pydantic model defines the expected structure of the request body
class Query(BaseModel):
    question: str

# Load the RAG chain once when the server starts with caching
qa_chain = get_qa_chain()

@lru_cache(maxsize=50)
def cached_invoke(query_str):
    return qa_chain.invoke({"query": query_str})

@app.post("/query")
def ask_question(query: Query):
    """
    This endpoint receives a question, processes it through the RAG chain,
    and returns the generated answer with sources.
    """
    result = cached_invoke(query.question)
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get('source', 'HR-Policy.pdf'),
            "page": doc.metadata.get('page', 'N/A')
        })
    return {"answer": result["result"], "sources": sources}

@app.get("/query")
def query_get_error():
    """
    Handle accidental GET requests to /query with a helpful message.
    """
    return {"error": "Method Not Allowed. Use POST to send a question to the chatbot."}
# backend.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  # For serving static files (favicon)
from pydantic import BaseModel
from core import get_qa_chain  # Import our main logic

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

# Load the RAG chain once when the server starts
qa_chain = get_qa_chain()

@app.post("/query")
def ask_question(query: Query):
    """
    This endpoint receives a question, processes it through the RAG chain,
    and returns the generated answer.
    """
    result = qa_chain.invoke({"query": query.question})
    return {"answer": result["result"]}

@app.get("/query")
def query_get_error():
    """
    Handle accidental GET requests to /query with a helpful message.
    """
    return {"error": "Method Not Allowed. Use POST to send a question to the chatbot."}
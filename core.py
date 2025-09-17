import os
import functools
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Define the path for the FAISS vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    """
    Creates a FAISS vector database from the HR policy PDF.
    """
    loader = PyPDFLoader('./data/HR-Policy (1).pdf')
    documents = loader.load()
    print("PDF loaded successfully.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    print("Embeddings model loaded.")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved at {DB_FAISS_PATH}")

# The prompt template guides the AI on how to answer questions
prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}
Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def rerank_docs(query, docs, k=2):
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:k]
    return [docs[i] for i in top_indices]

@functools.lru_cache(maxsize=None)
def get_qa_chain():
    """
    Initializes and returns a RetrievalQA chain with re-ranking and caching.
    """
    print("Loading QA chain...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 4})
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile"
        )
        base_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        def custom_invoke(inputs):
            query = inputs["query"]
            docs = base_chain.retriever.get_relevant_documents(query)
            reranked_docs = rerank_docs(query, docs)
            result = base_chain.combine_documents_chain.run(
                context='\n\n'.join([doc.page_content for doc in reranked_docs]),
                question=query
            )
            return {"result": result, "source_documents": reranked_docs}
        chain = type('CustomQA', (), {'invoke': custom_invoke})()
        print("QA chain loaded successfully.")
        return chain
    except Exception as e:
        print(f"Error loading QA chain: {e}")
        raise

if __name__ == "__main__":
    create_vector_db()
    print("--- Vector DB Creation Complete ---")
import os
import functools
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import to fix deprecation
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever, Document
from typing import List

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

    embeddings = HuggingFaceEmbeddings(  # Updated to new package
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

@functools.lru_cache(maxsize=None)
def get_qa_chain():
    """
    Initializes and returns a RetrievalQA chain.
    """
    print("Loading QA chain...")
    try:
        embeddings = HuggingFaceEmbeddings(  # Updated to new package
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        # Get initial FAISS retriever with higher k for re-ranking candidates
        initial_retriever = db.as_retriever(search_kwargs={'k': 5})  # Fetch 5 docs initially
        # BM25 setup: Create corpus from all documents in the vector store
        all_texts = [doc.page_content for doc in db.docstore._dict.values()]
        tokenized_corpus = [text.split() for text in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        # Define rerank function
        def rerank_docs(query: str, docs: List[Document], top_k: int = 2) -> List[Document]:
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:top_k]]
        # Custom retriever class to wrap FAISS + BM25
        class BM25RerankerRetriever(BaseRetriever):
            initial_retriever: BaseRetriever
            bm25: BM25Okapi
            top_k: int = 2
            def _get_relevant_documents(self, query: str) -> List[Document]:
                initial_docs = self.initial_retriever.get_relevant_documents(query)
                return rerank_docs(query, initial_docs, self.top_k)
        # Instantiate custom retriever
        custom_retriever = BM25RerankerRetriever(
            initial_retriever=initial_retriever,
            bm25=bm25,
            top_k=2  # Final top_k for chain
        )
        retriever = custom_retriever  # Now uses re-ranked results
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile"  # Updated to supported model
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("QA chain loaded successfully.")
        return chain
    except Exception as e:
        print(f"Error loading QA chain: {e}")
        raise

if __name__ == "__main__":
    create_vector_db()
    print("--- Vector DB Creation Complete ---")
from fastapi import APIRouter, Form
from app.services.tinyllama_service import tinyllama
from app.services.vector_service import get_vector_store

router = APIRouter()

@router.post("/query/")
async def query_document(question: str = Form(...)):
    # Get the vector store instance
    vector_store = get_vector_store()
    
    # Search for relevant chunks
    top_chunks = vector_store.search(question, top_k=3)
    
    if not top_chunks:
        return {"answer": "No documents have been uploaded yet. Please upload a PDF first."}
    
    # Generate answer using TinyLlama
    answer = tinyllama.ask(question, top_chunks)
    
    return {"answer": answer}
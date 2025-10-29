from fastapi import APIRouter, UploadFile
from app.services.pdf_service import extract_text_from_pdf
from app.services.vector_service import get_vector_store

router = APIRouter()

@router.post("/")
async def upload_pdf(file: UploadFile):
    text = extract_text_from_pdf(file)
    vector_store = get_vector_store()
    vector_store.add_document(file.filename, text)  # Note: add_document method needs to be implemented
    return {"message": f"{file.filename} uploaded and indexed."}
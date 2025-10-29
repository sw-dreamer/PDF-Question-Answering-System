from fastapi import FastAPI
from app.routes import upload, query

app = FastAPI(title="DocuMind TinyLlama")


app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])

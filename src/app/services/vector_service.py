from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class VectorService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.array([])
        self.chunks = []

    def add_document(self, filename: str, text: str, chunk_size: int = 500, overlap: int = 50):
        """문서를 청크로 나누고 임베딩을 추가"""
        from app.utils.chunker import chunk_text
        
        # 텍스트를 청크로 분할
        new_chunks = chunk_text(text, chunk_size, overlap)
        
        # 청크를 임베딩으로 변환
        new_embeddings = self.model.encode(new_chunks)
        
        # 기존 데이터에 추가
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.chunks.extend(new_chunks)
        
        return len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """쿼리와 가장 유사한 청크를 검색"""
        if len(self.chunks) == 0:
            return []
        
        # 쿼리를 임베딩으로 변환
        query_embedding = self.model.encode([query])
        
        # 코사인 유사도 계산
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 상위 k개 인덱스 추출
        top_idx = sims.argsort()[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_idx]


# 전역 벡터 스토어
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = VectorService()
    return vector_store
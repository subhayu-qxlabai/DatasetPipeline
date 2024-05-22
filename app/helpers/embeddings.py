import pickle
import hashlib
from typing import List
from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, String, LargeBinary

from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

Base = declarative_base()

GET_DEFAULT_EMBEDDINGS = lambda: HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={"device": "cpu"},
    show_progress=False,                        
)

class EmbeddingRecord(Base):
    __tablename__ = "embeddings"
    text_hash = Column(String, primary_key=True)
    embedding = Column(LargeBinary)

@dataclass
class Embeddings:
    model: LCEmbeddings = None
    use_cache: bool = True
    database_url: str = "sqlite:///embeddings.db"

    def __post_init__(self):
        if self.model is None:
            self.model = GET_DEFAULT_EMBEDDINGS()
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def hash_text(self, text: str) -> str:
        return hashlib.sha256((self.model.model_name + text).encode('utf-8')).hexdigest()

    def get_embedding_from_db(self, text_hash: str):
        session = self.Session()
        result = session.query(EmbeddingRecord).filter_by(text_hash=text_hash).first()
        session.close()
        if result:
            return pickle.loads(result.embedding)
        return None

    def _save_embedding_to_db(self, text_hash: str, embedding: List[float]):
        session = self.Session()
        binary_embedding = pickle.dumps(embedding)
        record = EmbeddingRecord(text_hash=text_hash, embedding=binary_embedding)
        session.merge(record)  # insert or update
        session.commit()
        session.close()

    def _embed_documents_with_caching(self, texts: List[str]):
        text_hashes = {text: self.hash_text(text) for text in texts}
        
        cached_embeddings = {}
        uncached_texts = []

        for text, text_hash in text_hashes.items():
            embedding = self.get_embedding_from_db(text_hash)
            if embedding is not None:
                cached_embeddings[text] = embedding
            else:
                uncached_texts.append(text)

        generated_embeddings = self.model.embed_documents(uncached_texts)
        
        for text, embedding in zip(uncached_texts, generated_embeddings):
            self._save_embedding_to_db(text_hashes[text], embedding)
            cached_embeddings[text] = embedding

        return [cached_embeddings[text] for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.use_cache:
            return self._embed_documents_with_caching(texts)
        return self.model.embed_documents(texts)

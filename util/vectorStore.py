import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from .embedding import EmbeddingPipeline

class FaissVectorStore:
    """A simple vector store using FAISS for similarity search."""

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2",  
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"Initialized FaissVectorStore with embedding model: {embedding_model}")

    def build_from_document(self, documents: List[Any]):
        """Build the FAISS index from the given documents."""
        print(f"Building vector store from {len(documents)} documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadata = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadata)
        self.save()
        print(f"[Info] Vector store built and saved with {len(chunks)} chunks.")

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Any] = None):
        """Add embeddings and their corresponding metadata to the FAISS index."""
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            print(f"Created new FAISS index with dimension: {dim}")
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)
        print(f"Added {embeddings.shape[0]} embeddings. Total: {self.index.ntotal}")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index to {faiss_path} and metadata to {meta_path}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded FAISS index from {faiss_path} and metadata from {meta_path}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for the most similar chunks given a query embedding."""
        if self.index is None:                                    # Fix 2: guard against uninitialized index
            raise RuntimeError("Index not built or loaded. Call build_from_document() or load() first.")

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:                                         # Fix 3: skip invalid FAISS padding indices
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({'index': idx, 'distance': dist, 'metadata': meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        """Generate embedding for the query text and perform search."""
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
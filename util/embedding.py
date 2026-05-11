from typing import List,  Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from util.data_loader import load_all_documents

class EmbeddingPipeline:
    """Pipeline to handle text splitting and embedding generation."""
    def __init__(self , model_name:str = "all-MiniLM-L6-v2" , chunk_size:int = 1000 , chunk_overlap:int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")
    
    def chunk_documents(self , documents:List[Any])->List[Any]:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        print(f"[Info] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks 
    
    def embed_chunks(self , chunks:List[str])->np.ndarray:
        """Generate embeddings for the given text chunks."""
        texts = [chunk.page_content for chunk in chunks]
        
        print(f"[Info] Generating embeddings for {len(texts)} chunks...")
        embeddings =  self.model.encode(texts, show_progress_bar=True)
        print(f"[Info] Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
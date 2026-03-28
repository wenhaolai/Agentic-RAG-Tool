import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import logging
import faiss
import numpy as np
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSWikiBuilder:
    def __init__(self, config_path: str = None):
        """
        Initialize FAISS Builder with config.
        """
        self.config = load_config(config_path)
        
        # Paths
        self.corpus_path = self.config.get("paths", {}).get("rag_corpus_path")
        
        # Embedding config
        emb_config = self.config.get("rag_data", {}).get("embedding", {})
        self.model_name = emb_config.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.device = emb_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.emb_batch_size = emb_config.get("batch_size", 256)
        
        # FAISS config
        faiss_config = self.config.get("rag_data", {}).get("faiss", {})
        self.index_path = faiss_config.get("index_path", "faiss_index/wiki_en.index")
        self.dimension = faiss_config.get("dimension", 512)
        self.index_type = faiss_config.get("index_type", "Flat")
        
        # Initialize resources
        self._init_model()
        self._init_index()

    def _init_model(self):
        """Initialize SentenceTransformer model."""
        logger.info(f"Initializing embedding model: {self.model_name} on {self.device}...")
        try:
            # 开启 FP16 半精度加速 (仅限 CUDA)
            model_kwargs = {"torch_dtype": torch.float16} if "cuda" in self.device else {}
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                model_kwargs=model_kwargs
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _init_index(self):
        """Initialize FAISS index."""
        logger.info(f"Initializing FAISS index type: {self.index_type} with dim {self.dimension}...")
        
        # Use Inner Product (IP) for cosine similarity (ensure vectors are normalized)
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Fallback or other types (e.g., HNSW)
            # Example: HNSW
            # self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            # Default to Flat if unknown
            logger.warning(f"Unknown index type '{self.index_type}', defaulting to Flat (IP).")
            self.index = faiss.IndexFlatIP(self.dimension)
            
        # If using GPU for FAISS (optional, usually CPU index is fine for <10M vectors if RAM allows)
        # For simplicity and stability on varying hardware, we use CPU index here.
        # Generating embeddings is on GPU (if available).

    def build(self):
        """Execute the build process."""
        logger.info(f"Loading dataset from {self.corpus_path}...")
        try:
            ds = load_from_disk(self.corpus_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

        if 'train' in ds:
            data = ds['train']
        else:
            data = ds[list(ds.keys())[0]]
            
        total_docs = len(data)
        logger.info(f"Dataset loaded. Total documents: {total_docs}")

        # Parameters for chunking based on character length
        chunk_size = 512
        chunk_overlap = 100
        
        # We need to manage memory, so processed in chunks
        # Usually dataset iteration is fast, bottleneck is encoding.
        
        logger.info("Starting chunking, embedding and indexing...")
        
        # 批量大小 (embedding batch size trigger)
        process_batch_size = 10000 
        texts_buffer = []
        
        for row in tqdm(data, desc="Indexing"):
            title = row.get('title') or ""
            text = row.get('text') or ""
            
            if not text:
                texts_buffer.append(title)
            else:
                start = 0
                while start < len(text):
                    chunk = text[start:start + chunk_size]
                    # Prepare texts: Title + " " + Chunk
                    texts_buffer.append(title + " " + chunk)
                    start += (chunk_size - chunk_overlap)
                    
            if len(texts_buffer) >= process_batch_size:
                # Encode
                # normalize_embeddings=True for Cosine Similarity user with IndexFlatIP
                embeddings = self.model.encode(
                    texts_buffer, 
                    batch_size=self.emb_batch_size, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                print(embeddings.shape)
                
                # Add to index
                self.index.add(embeddings)
                
                # Explicitly clear memory just in case
                del embeddings
                texts_buffer = []
                
        # Process remaining chunks
        if len(texts_buffer) > 0:
            embeddings = self.model.encode(
                texts_buffer, 
                batch_size=self.emb_batch_size, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.index.add(embeddings)
            del embeddings
            texts_buffer = []

        logger.info(f"Indexing complete. Total vectors in index: {self.index.ntotal}")

        # Save index
        self._save_index()

    def _save_index(self):
        """Save the FAISS index to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        logger.info(f"Saving index to {self.index_path}...")
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info("Index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

if __name__ == "__main__":
    builder = FAISSWikiBuilder(config_path="config.yaml")
    builder.build()

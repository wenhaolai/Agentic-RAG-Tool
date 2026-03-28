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
import json

from src.utils.config_loader import load_config

def build_chunks():
    chunk_size = 512
    overlap_size = 100

    process_batch_size = 10000 
    texts_buffer = []

    ds = load_from_disk("/root/autodl-tmp/wikipedia")
    if 'train' in ds:
        data = ds['train']
    else:
        data = ds[list(ds.keys())[0]]

    total_docs = len(data)
    print(f"Dataset loaded. Total documents: {total_docs}")

    metadata_buffer = []
    global_chunk_id = 0

    for row in tqdm(data, desc="Indexing"):
        title = row.get('title') or ""
        text = row.get('text') or ""

        if not text:
            texts_buffer.append(title)
            metadata_buffer.append({
                "title": title,
                "text": title,
                "chunk_id": global_chunk_id
            })
            global_chunk_id += 1
        else:
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size]
                texts_buffer.append(chunk)
                metadata_buffer.append({
                    "title": title,
                    "text": chunk,
                    "chunk_id": global_chunk_id
                })
                global_chunk_id += 1
                start += chunk_size - overlap_size
    
    metadata_path = "/root/autodl-tmp/faiss_index/wiki_zh_metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for metadata in metadata_buffer:
            f.write(f"{json.dumps(metadata, ensure_ascii=False)}\n")
    print(f"Metadata saved to {metadata_path}")

if __name__=="__main__":
    build_chunks()
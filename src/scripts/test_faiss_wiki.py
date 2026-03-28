import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class FAISSSearcher:
    def __init__(self, index_path, metadata_path, model_name="BAAI/bge-small-zh-v1.5"):
        # 1. 加载相同的 Embedding 模型
        print("正在加载 Embedding 模型...")
        self.model = SentenceTransformer(model_name)
        
        # 2. 加载之前保存的 FAISS 索引
        print(f"正在加载 FAISS 索引: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"索引加载成功，库中共有 {self.index.ntotal} 条向量。")

        # 3. 加载元数据
        print(f"正在加载元数据: {metadata_path}")
        self.metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"元数据加载成功，共 {len(self.metadata)} 条记录。")

    def search(self, query, top_k=5):
        # 3. 对用户的 Query 进行向量化
        # 注意：BGE 模型在做检索任务时，官方强烈建议在 Query 前加特定的指令前缀
        instruction = "Represent this sentence for searching relevant passages: "
        full_query = instruction + query
        
        # 将 Query 转为向量
        # 注意：因为构建时使用了 IndexFlatIP 和 normalize_embeddings=True
        # 所以查询向量也必须进行归一化，才能正确计算余弦相似度
        query_vector = self.model.encode(
            [full_query],
            # query, 
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        print(query_vector.shape)
        
        # 4. 在 FAISS 中执行搜索
        # distances: 返回的相似度得分 (因为是归一化内积，值在 -1 到 1 之间，越接近 1 越相似)
        # indices: 返回的最相似向量的内部整数 ID
        distances, indices = self.index.search(query_vector, top_k)
        
        return distances[0], indices[0]

# ================= 使用示例 =================
if __name__ == "__main__":
    # 假设你的索引文件保存在这里
    index_file = "/root/autodl-tmp/faiss_index/wiki_zh.index" 
    metadata_path = "/root/autodl-tmp/faiss_index/wiki_zh_metadata.jsonl"
    searcher = FAISSSearcher(index_file, metadata_path)
    
    user_question = "热射病是什么？"
    print(f"\n用户问题: {user_question}")
    
    # 查找最相关的 5 个结果
    scores, ids = searcher.search(user_question, top_k=5)
    
    print("\n--- 检索结果 ---")
    for rank, (score, doc_id) in enumerate(zip(scores, ids)):
        print(f"Rank {rank+1}: 相似度得分 = {score:.4f}, FAISS ID = {doc_id}")
        print(f"元数据: {searcher.metadata[doc_id]}")
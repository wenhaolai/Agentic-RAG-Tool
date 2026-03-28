import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import sys
import logging
from typing import List, Dict
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

# Add project root to sys path if not present (to find src module)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSWikiSearcher:
    def __init__(self, config_path: str = None):
        """
        初始化 FAISS 搜索器
        :param config_path: 配置文件路径
        """
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config()
            
        # 读取配置
        rag_data = self.config.get("rag_data", {})
        faiss_config = rag_data.get("faiss", {})
        emb_config = rag_data.get("embedding", {})
        retrieval_config = rag_data.get("retrieval", {})

        # 模型及检索参数
        self.index_path = faiss_config.get("index_path", "faiss_index/wiki_zh.index")
        self.model_name = emb_config.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.device = emb_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.default_top_k = retrieval_config.get("top_k", 5)
        self.score_threshold = retrieval_config.get("score_threshold", 0.0)
        self.metadata_path = faiss_config.get("metadata_path", "faiss_index/wiki_zh_metadata.jsonl")

        # 语料库路径，用于重建文本映射
        self.corpus_path = self.config.get("paths", {}).get("rag_corpus_path")

        # 初始化核心组件
        self._init_model()
        self._init_index()
        self._init_metadata()

    def _init_model(self):
        """加载 Embedding 模型"""
        logger.info(f"正在加载 Embedding 模型: {self.model_name} on {self.device}")
        try:
            model_kwargs = {"torch_dtype": torch.float16} if "cuda" in self.device else {}
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                model_kwargs=model_kwargs
            )
        except Exception as e:
            logger.error(f"加载 Embedding 模型失败: {e}")
            raise

    def _init_index(self):
        """加载 FAISS 索引"""
        logger.info(f"正在加载 FAISS 索引: {self.index_path}")
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"索引加载成功，库中共有 {self.index.ntotal} 条向量。")
        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {e}")
            raise

    def _init_metadata(self):
        """加载元数据"""
        logger.info(f"正在加载元数据: {self.metadata_path}")
        self.metadata = []
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line))
            logger.info(f"元数据加载成功，共 {len(self.metadata)} 条记录。")
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            raise


    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        执行搜索
        :param query: 搜索关键词
        :param top_k: 返回结果数量，默认使用配置值
        :return: 结果列表 [{"text":..., "faiss_score":...}]
        """
        if top_k is None:
            top_k = self.default_top_k

        # 针对基于检索任务优化的模型，加上强烈建议的前缀指令
        instruction = "为这个句子生成表示以用于检索相关文章："
        full_query = instruction + query

        try:
            # Query 向量化
            query_vector = self.model.encode(
                [full_query], 
                convert_to_numpy=True,
                normalize_embeddings=True 
            )
            
            # FAISS 查找
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            # distances[0] 包含余弦相似度的打分, indices[0] 包含整数 ID
            for score, doc_id in zip(distances[0], indices[0]):
                # 跳过未找到的情况或者得分低于阈值的情况
                if doc_id == -1 or score < self.score_threshold:
                    continue
                    
                text_content = self.metadata[doc_id].get("text", "文本丢失")
                
                results.append({
                    "text": text_content,
                    "faiss_score": float(score)  # 转为标准 float 方便 json 序列化
                })
            
            return results

        except Exception as e:
            logger.error(f"FAISS 搜索执行失败: {e}")
            return []

    def format_results(self, results: List[Dict]) -> str:
        """
        格式化输出结果，便于打印
        """
        if not results:
            return "未找到相关结果。"
            
        output = []
        for i, res in enumerate(results):
            text_preview = res['text'][:200].replace('\n', ' ') + "..."
            doc_str = (
                f"[{i+1}] (Score: {res['faiss_score']:.4f})\n"
                f"    Text: {text_preview}\n"
            )
            output.append(doc_str)
            
        return "\n".join(output)

if __name__ == "__main__":
    # 测试代码
    try:
        searcher = FAISSWikiSearcher()
        test_query = "台湾在哪个省？"
        
        print(f"\n正在搜索: '{test_query}' ...")
        results = searcher.search(test_query)
        
        print(f"找到 {len(results)} 条结果:\n")
        print(searcher.format_results(results))
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        pass
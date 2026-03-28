import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

from src.retrieval.es_wiki_search import ESWikiSearcher
from src.retrieval.faiss_wiki_search import FAISSWikiSearcher

class HybridSearcher:
    def __init__(self, config_path: str = None):
        self.es_searcher = ESWikiSearcher(config_path)
        self.faiss_searcher = FAISSWikiSearcher(config_path)

        self.default_top_k = 20
        self.rerank_model = SentenceTransformer("BAAI/bge-reranker-large")

    def search(self, query: str, top_k: int = None):
        """
        执行混合搜索
        :param query: 搜索关键词
        :param top_k: 初始召回结果数量，默认使用配置值
        :return: 经过精排后的最好的 3 个文本列表，只有文本内容
        """
        if top_k is None:
            top_k = self.default_top_k

        try:
            # 1. 多路独立召回
            es_results = self.es_searcher.search(query, top_k)
            # FAISS 内部已经处理了 instruction，这里直接传原始 query 即可
            faiss_results = self.faiss_searcher.search(query, top_k)
            
            # 2. 合并去重构建候选池 (Candidate Pool)
            unique_texts = {}
            
            for res in es_results:
                text = res.get("text")
                if text and text not in unique_texts:
                    unique_texts[text] = True
                    
            for res in faiss_results:
                text = res.get("text")
                if text and text not in unique_texts:
                    unique_texts[text] = True
            
            passages = list(unique_texts.keys())
            
            # 若没有任何召回结果，则直接返回
            if not passages:
                return []
                
            # 3. 使用 Reranker 进行打分重排
            instruction = "为这个句子生成表示以用于检索相关文章："
            # 处理 Query
            q_embeddings = self.rerank_model.encode([instruction + query], normalize_embeddings=True)
            # 处理所有候选段落
            p_embeddings = self.rerank_model.encode(passages, normalize_embeddings=True)
            
            # 计算点积得出每一对之间的 Score (1, dim) @ (dim, num_passages) -> (1, num_passages)
            scores = (q_embeddings @ p_embeddings.T)[0]
            
            # 4. 根据 Score 进行排序并提取前三个
            scored_passages = list(zip(scores, passages))
            # 按照分数降序排列
            scored_passages.sort(key=lambda x: x[0], reverse=True)
            
            # 提取最好的 3 个结果（如果总数不满3个则提全量）
            best_3_results = [item[1] for item in scored_passages[:3]]
            
            return best_3_results

        except Exception as e:
            print(f"搜索过程中发生错误: {e}")
            return []

if __name__=="__main__":
    searcher = HybridSearcher("/root/Agentic-RAG-Tool/config.yaml")
    print("searcher is prepared")

    query = "什么是人工智能？"
    results = searcher.search(query)
    print("Search results:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result}")
    
    

import os
import sys
import logging
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch

# Add project root to sys path if not present (to find src module)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESWikiSearcher:
    def __init__(self, config_path: str = None):
        """
        初始化 ES 搜索器
        :param config_path: 配置文件路径
        """
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config()
        
        # 读取配置
        rag_data = self.config.get("rag_data", {})
        es_config = rag_data.get("elasticsearch", {})
        retrieval_config = rag_data.get("retrieval", {})

        self.host = es_config.get("host", "http://localhost:9200")
        self.index_name = es_config.get("index_name", "wiki_en")
        self.username = es_config.get("username")
        self.password = es_config.get("password")
        
        # 默认检索参数
        self.default_top_k = retrieval_config.get("top_k", 5)
        self.score_threshold = retrieval_config.get("score_threshold", 0.0)

        # 初始化客户端
        basic_auth = None
        if self.username and self.password:
            basic_auth = (self.username, self.password)
            
        try:
            self.es = Elasticsearch(self.host, basic_auth=basic_auth)
            if self.es.ping():
                logger.info(f"成功连接到 ES 服务器: {self.host}")
            else:
                logger.warning(f"连接 ES 服务器失败: {self.host}")
        except Exception as e:
            logger.error(f"ES 初始化异常: {e}")
            raise

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        执行搜索
        :param query: 搜索关键词
        :param top_k: 返回结果数量，默认使用配置值
        :return: 结果列表 [{"id":..., "title":..., "text":..., "score":...}]
        """
        if top_k is None:
            top_k = self.default_top_k

        # 构建查询 DSL
        # 使用 multi_match 在 title 和 text 中搜索，增加 title 的权重
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "text"], # 标题权重 x2
                    "type": "best_fields"
                }
            },
            "size": top_k,
            "_source": ["id", "title", "url", "text"] # 只返回需要的字段
        }

        try:
            response = self.es.search(index=self.index_name, body=body)
            hits = response['hits']['hits']
            results = []
            
            for hit in hits:
                score = hit['_score']
                if score < self.score_threshold:
                    continue
                    
                source = hit['_source']
                results.append({
                    "id": source.get("id"),
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "text": source.get("text"),
                    "score": score
                })
            
            return results

        except Exception as e:
            logger.error(f"搜索执行失败: {e}")
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
                f"[{i+1}] Title: {res['title']} (Score: {res['score']:.4f})\n"
                f"    URL: {res['url']}\n"
                f"    Text: {text_preview}\n"
            )
            output.append(doc_str)
            
        return "\n".join(output)

if __name__ == "__main__":
    # 测试代码
    try:
        searcher = ESWikiSearcher()
        test_query = "Python programming"
        
        print(f"正在搜索: '{test_query}' ...")
        results = searcher.search(test_query)
        
        print(f"找到 {len(results)} 条结果:\n")
        print(searcher.format_results(results))
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")

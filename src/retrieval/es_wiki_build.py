import os
import sys
import logging
from elasticsearch import Elasticsearch, helpers
from datasets import load_from_disk
from tqdm import tqdm

from src.utils.config_loader import load_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESWikiBuilder:
    def __init__(self, host="http://localhost:9200", basic_auth=None):
        """
        初始化 ES 客户端
        :param host: ES 服务地址
        :param basic_auth: (username, password) 元组，如果需要鉴权
        """
        self.es = Elasticsearch(host, basic_auth=basic_auth)
        if self.es.ping():
            logger.info(f"成功连接到 Elasticsearch: {host}")
        else:
            logger.error(f"无法连接到 Elasticsearch: {host}")
            raise ConnectionError(f"Cannot connect to Elasticsearch at {host}")

    def create_index(self, index_name="wiki_en"):
        """
        创建索引并定义 Mapping
        注意：针对英文数据集，使用 standard 分词器即可
        """
        # 定义索引 Mapping
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",  # 英文推荐使用 standard
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "standard"   # 英文推荐使用 standard
                    }
                }
            }
        }

        # 检查索引是否存在
        if self.es.indices.exists(index=index_name):
            logger.info(f"索引 {index_name} 已存在，正在删除...")
            self.es.indices.delete(index=index_name)
        
        try:
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"索引 {index_name} 创建成功。")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")

    def _generate_actions(self, dataset, index_name):
        """生成批量写入的动作"""
        for item in dataset:
            yield {
                "_index": index_name,
                "_source": {
                    "id": item.get("id"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "text": item.get("text")
                }
            }

    def build(self, corpus_path, index_name="wiki_zh", batch_size=2000):
        """
        执行构建流程：加载数据 -> 创建索引 -> 批量写入
        """
        logger.info(f"正在从 {corpus_path} 加载数据集...")
        try:
            ds = load_from_disk(corpus_path)
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            return

        # 通常数据集有 train/test split，这里默认取第一个或者名为 'train' 的 split
        if 'train' in ds:
            data = ds['train']
        else:
            data = ds[list(ds.keys())[0]]
            
        logger.info(f"数据集加载完成，共 {len(data)} 条数据。")

        # 1. 创建索引
        self.create_index(index_name)

        # 2. 批量写入
        logger.info(f"开始写入数据到索引 {index_name}...")
        try:
            # 使用 tqdm 显示进度
            success, failed = helpers.bulk(
                self.es, 
                self._generate_actions(data, index_name),
                chunk_size=batch_size,
                stats_only=True,
                raise_on_error=False
            )
            logger.info(f"写入完成: 成功 {success} 条，失败 {failed} 条。")
        except Exception as e:
            logger.error(f"写入过程中发生错误: {e}")

def main():
    # 1. 加载配置
    config = load_config()
    paths = config.get("paths", {})
    rag_corpus_path = paths.get("rag_corpus_path")
    
    if not rag_corpus_path:
        logger.error("在 config.yaml 中未找到 rag_corpus_path 配置。")
        return

    # 2. ES 配置
    rag_config = config.get("rag_data", {})
    es_config = rag_config.get("elasticsearch", {})
    
    es_host = es_config.get("host", os.getenv("ES_HOST", "http://localhost:9200"))
    es_username = es_config.get("username")
    es_password = es_config.get("password")
    index_name = es_config.get("index_name", "wiki_zh")
    
    basic_auth = None
    if es_username and es_password:
        basic_auth = (es_username, es_password)

    # 3. 开始构建
    builder = ESWikiBuilder(host=es_host, basic_auth=basic_auth)
    builder.build(corpus_path=rag_corpus_path, index_name=index_name)

if __name__ == "__main__":
    main()

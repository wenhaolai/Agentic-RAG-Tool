import os

# 设置缓存目录到大数据盘 /root/autodl-tmp/cache
# 必须在导入 datasets 之前设置
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/cache/huggingface/datasets'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import yaml
from datasets import load_dataset
from src.utils.config_loader import load_config

def download_rag_data():
    # 加载config.yaml
    config = load_config()
    rag_config = config.get("rag_data", {})
    hf_config = rag_config.get("huggingface_dataset", {})
    
    # 获取参数
    repo_id = hf_config.get("repo_id", "wikimedia/wikipedia")
    subset = hf_config.get("subset", "20231101.en") # 默认为中文

    # 从根配置 paths 中获取保存路径 (rag_corpus_path 现在指向目录)
    paths_config = config.get("paths", {})
    save_dir = paths_config.get("rag_corpus_path", "./data/corpus")

    print(f"Saving dataset to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载并加载数据
    print(f"Loading dataset {repo_id} (subset: {subset})...")
    ds = load_dataset(repo_id, subset)
    
    # 保存数据 (使用 save_to_disk 保存为 Arrow 格式，方便后续读取)
    print(f"Saving dataset to {save_dir}...")
    ds.save_to_disk(save_dir)
    print(f"Saved successfully to {save_dir}")

if __name__ == "__main__":
    download_rag_data()

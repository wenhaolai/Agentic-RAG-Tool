import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import yaml
from huggingface_hub import snapshot_download
from src.utils.config_loader import load_config

def download_rag_data():
    # 加载config.yaml
    config = load_config()
    rag_config = config.get("rag_data", {})
    hf_config = rag_config.get("huggingface_dataset", {})
    
    # 获取参数
    repo_id = hf_config.get("repo_id", "wikimedia/wikipedia")
    subset = hf_config.get("subset", "20231101.zh") # 默认为中文

    # 从根配置 paths 中获取保存路径
    paths_config = config.get("paths", {})
    output_path = paths_config.get("rag_corpus_path", "./data/corpus/wiki_zh.jsonl")
    local_dir = os.path.dirname(output_path)
    
    os.makedirs(local_dir, exist_ok=True)
    
    # 下载数据 (直接下载 parquet 文件)
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.parquet"],
        local_dir_use_symlinks=False
    )
    print(f"Saved to {local_dir}")

if __name__ == "__main__":
    download_rag_data()

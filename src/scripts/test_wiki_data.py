import sys
import os
from datasets import load_from_disk
from src.utils.config_loader import load_config

def main():
    # 1. 加载配置
    config = load_config()
    paths_config = config.get("paths", {})
    rag_corpus_path = paths_config.get("rag_corpus_path")

    if not rag_corpus_path or not os.path.exists(rag_corpus_path):
        print(f"Error: 找不到数据路径: {rag_corpus_path}")
        return

    print(f"正在从本地加载数据集: {rag_corpus_path}")

    # 2. 从磁盘加载数据
    try:
        ds = load_from_disk(rag_corpus_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("\n" + "="*50)
    print("数据集概览")
    print("="*50)
    print(ds)
    
    # 3. 检查 split (通常会有 'train')
    split_name = 'train'
    if split_name not in ds:
        # 如果没有 train，尝试获取第一个 split
        split_name = list(ds.keys())[1]
    
    data = ds[split_name]
    
    print("\n" + "="*50)
    print(f"分析 split: '{split_name}' (总共 {len(data)} 条数据)")
    print("="*50)

    # 4. 查看列名 (features)
    print(f"数据列名 (Features): {data.column_names}")
    
    # 5. 打印几条样例数据
    print("\n" + "="*50)
    print("样例数据展示 (前 3 条)")
    print("="*50)
    
    # 只取前3条
    samples = data.select(range(10))
    
    for i, example in enumerate(samples):
        print(f"\n[Sample {i+1}]")
        for key, value in example.items():
            # 为了防止内容过长，截取前 200 个字符
            if isinstance(value, str) and len(value) > 200:
                display_value = value[:200] + "... (truncated)"
            else:
                display_value = value
            print(f"  - {key}: {display_value}")

    # 6. 简单的长度统计
    # 只是简单统计一下 text 字段的长度分布
    if 'text' in data.column_names:
        print("\n" + "="*50)
        print("简单统计: 'text' 字段长度")
        print("="*50)
        # 随机抽样 1000 条进行估算
        sample_size = min(1000, len(data))
        text_lengths = [len(x['text']) for x in data.select(range(sample_size))]
        avg_len = sum(text_lengths) / len(text_lengths)
        print(f"基于前 {sample_size} 条数据的估算:")
        print(f"  - 平均长度: {avg_len:.1f} 字符")
        print(f"  - 最大长度: {max(text_lengths)} 字符")
        print(f"  - 最小长度: {min(text_lengths)} 字符")

if __name__ == "__main__":
    main()
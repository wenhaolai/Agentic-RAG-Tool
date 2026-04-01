import os
import asyncio
import json
from typing import Any

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client

from src.utils.config_loader import load_config
from src.models.model import AgenticRAGModel
from src.data.prompt import build_system_tools

def main():
    config = load_config()

    # == 1. 获取模型配置与加载模型 ==
    model_config = config.get("models")
    generation_config = model_config.get("generation")
    base_model_path = generation_config.get("local_path")
    device = generation_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {base_model_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    if device != "cuda" and base_model.device.type != device:
        base_model.to(device)

    print("Model loaded successfully!")
    
    # == 2. 初始化 MCP Client 用于实时调用 ==
    # 只需要在推理时保持一个 MCP Client
    mcp_config = {"transport": "http://127.0.0.1:8000/mcp"}
    try:
        mcp_client = Client(**mcp_config)
    except Exception as e:
        print(f"Failed to create MCP Client: {e}")
        mcp_client = None

    # == 3. 构建包装了 MCP 的 AgenticRAGModel ==
    model = AgenticRAGModel(base_model, tokenizer, mcp_client=mcp_client)

    # == 4. 组装支持中文检索的一键 Prompt == 
    query = "什么是人工智能"
    
    # 现在直接调用 build_system_tools 即可自动走内部通信渲染出全量 System Prompt
    system_prompt = build_system_tools(mcp_server_url=mcp_config["transport"])
    prompt = {"text": f"{system_prompt}\n\nUser: {query}"}

    inputs = tokenizer(prompt["text"], return_tensors="pt").to(device)
    inputs_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print("\nStarting generation...")
    # == 5. 执行伴随思维打断的 RAG 生成 ==
    output_ids = model.generate(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            max_length_for_gather=10000,
            do_sample=False,
            temperature=0.8,
    )

    output_ids = output_ids[0][len(inputs_ids[0]) :]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    
    print("\n=== 最终输出 ===")
    print(outputs)

if __name__ == "__main__":
    main()
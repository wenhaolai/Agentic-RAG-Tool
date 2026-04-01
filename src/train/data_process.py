import argparse
import asyncio
import json
import os
from typing import Any

import datasets
from fastmcp import Client
from fastmcp.tools import Tool
from rich import pretty

#from verl.utils.hdfs_io import copy, makedirs

import pandas as pd
from src.utils.config_loader import load_config
from src.data.prompt import SYSTEM_PROMPT_TOOLS_BACKTRACK_EN

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--mcp_server_url", default="http://127.0.0.1:8000/mcp", help="The URL of the MCP server.")
    parser.add_argument("--judge_model", default=None, help="The model to use for judging the responses.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/ugreen_function_call",
        help="The save directory for the preprocessed dataset.",
    )

    arg = parser.parse_args()

    # 准备MCP格式的调用工具
    from src.data.prompt import get_mcp_tools, convert_to_openai_tools
    mcp_config = {"transport": arg.mcp_server_url}
    loop = asyncio.get_event_loop()
    mcp_tools = loop.run_until_complete(get_mcp_tools(mcp_cfg=mcp_config))
    openai_tools = convert_to_openai_tools(mcp_tools)

    print(json.dumps(openai_tools))

    data_source = "/root/Agentic-RAG-Tool/data/train.parquet"
    ds = datasets.load_dataset("parquet", data_files=data_source)
    ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds= ds["train"]
    test_ds = ds["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TOOLS_BACKTRACK_EN
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "tool",
                "reward_model": {"style": "rule"},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "tools": json.dumps(openai_tools["tools"], ensure_ascii=False),
                    "judge_model": arg.judge_model,
                    "need_tools_kwargs": False,
                    "interaction_kwargs": {
                        "query": question,
                    },
                },
            }
            return data
        return process_fn
    
    train_dataset = train_ds.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_ds.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = "/root/autodl-tmp/grpo_data"
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    
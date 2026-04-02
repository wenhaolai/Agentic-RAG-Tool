import re
import json
from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall

@ToolParser.register("qwen_rag_parser")
class QwenRAGToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        # 精确匹配 Prompt 规定的格式: <search> [工具名]: {"参数": "值"} </search>
        self.tool_call_regex = re.compile(r"<search>\s*\[(.*?)\]:\s*(\{.*?\})\s*</search>", re.DOTALL)

    async def extract_tool_calls(self, text: str):
        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                name = match[0].strip()       # 提取工具名，如 "wiki_rag"
                args_str = match[1].strip()   # 提取 JSON 参数字符串
                
                # 校验 JSON 格式是否合法
                json.loads(args_str)
                
                # 封装成标准 FunctionCall，VERL 的 MCPBaseTool 会接管它并发起网络请求
                function_calls.append(FunctionCall(name=name, arguments=args_str))
            except Exception as e:
                pass
        return function_calls
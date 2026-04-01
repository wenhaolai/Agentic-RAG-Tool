import json
import asyncio
from typing import Any
from fastmcp import Client
from fastmcp.tools import Tool

async def get_mcp_tools(mcp_cfg: dict) -> list[Tool]:
    """Get tools from MCP server."""
    client = Client(**mcp_cfg)
    async with client:
        tools = await client.list_tools()
    return tools

def convert_to_openai_tools(tools: list[Tool]) -> dict[str, list[dict[str, Any]]]:
    functions = []
    for tool in tools:
        function = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": tool.inputSchema or {},
            },
        }
        functions.append(function)
    return {"tools": functions}

TOOL_DESC = """工具名称: {name}
功能描述: {description}
参数定义: {parameters}
（当你需要使用此工具时，请确保参数格式符合上述 JSON 规范。）"""


SYSTEM_PROMPT_TOOLS_BACKTRACK_ZH = """
当用户提出问题时，助手应积极解决。助手可以进行思考、搜索、反思，然后给出最终答案。请使用以下结构化标签来组织推理和搜索步骤。精确的格式很重要——请遵循以下规则。

标签（语义角色）
1. <reasoning> ... </reasoning>
   - 用于记录助手的内部推理、逐步分析或中间想法，以解释助手是如何得出结论的。
2. <search> ... </search>
   - 当助手必须执行外部或不确定的信息检索时使用。
   - 内容必须遵循这个精确的格式，指定你想要调用的工具名称（例如 wiki_rag）以及 JSON 格式的输入参数：
     "<search> [wiki_rag]: {{\"input\": \"关键词\"}} </search>"
   - 发送搜索标签后，系统/工具将返回被 "<observation> ... </observation>" 块包裹的结果。
3. <backtrack> ... </backtrack>
   - 当先前的推理或结论需要纠正或修改时使用。需要解释改变了什么以及为什么。
4. <summary> ... </summary>
   - 用于对先前的上下文或结论提供简明的回顾总结。
5. <answer> ... </answer>
   - 提供给用户问题的最终答案。
   - 此标签必须且只能出现一次，并且必须放置在回复的最末尾。

可用工具列表：
{tool_descs}

严格规则与格式约束
1. 只有 <answer> 标签必须且只能出现一次——并且它必须只出现在助手回复的最后。
2. 所有其他标签（<reasoning>、<search>、<backtrack>、<summary>）可以根据需要多次、以任意顺序出现。
3. 保持标签拼写和尖括号标点符号的精确无误。标签区分大小写。
4. 使用 <search> 时，坚持所需的语法：使用字面上的前缀 "[工具名称]" 加上 JSON 格式的参数串。请勿偏离此格式。
5. 如果使用了 <search> 标签，系统后续会提供一个 "<observation> ... </observation>"。请预料到此情况，并将该观察结果纳入后续的推理或最终的答案中。
6. 保持推理清晰聚焦——如果合适，较长的内部思维链可以拆分到多个 <reasoning> 块中。

行为指南
- 保持简洁、诚实且具有帮助性。
- 你可以多次调用可用的工具来帮助你完成任务。
- 当你进行回溯（backtrack）时，清晰地说明你改变了什么以及为什么。
- 最终的 <answer> 应该是一个清晰独立的回复，用户无需看到中间过程的标签也能阅读（如果为了提高清晰度，允许在其中包含推理意图的简短摘要）。
- 避免将仅限内部控制的信号或非人类可读的标记泄露到结构化标签之外。
"""


def build_system_tools(mcp_server_url: str = "http://127.0.0.1:8000/mcp", sys_prompt=SYSTEM_PROMPT_TOOLS_BACKTRACK_ZH):
   """
   一键构建系统提示词。此函数将自动向指定的 MCP Server 发起请求，抓取当前的 Tool 列表并嵌入提示词中。
   """
   mcp_config = {"transport": mcp_server_url}
   try:
      # Use asyncio.run for a clean isolated loop
      mcp_tools_raw = asyncio.run(get_mcp_tools(mcp_cfg=mcp_config))
      openai_tools = convert_to_openai_tools(mcp_tools_raw)
   except Exception as e:
      print(f"Warning: Failed to fetch tools from MCP server ({e}). Using empty tool list.")
      openai_tools = {"tools": []}
        
   if not openai_tools or "tools" not in openai_tools:
      return sys_prompt.format(tool_descs="无可用工具。")

   tool_descs_list = []
    
   for tool in openai_tools["tools"]:
      func = tool.get("function", {})
      name = func.get("name", "unknown_tool")
      desc = func.get("description", "No description provided.")
      params = func.get("parameters", {})
        
      # 将参数字典准换为 JSON 字符串显示
      params_str = json.dumps(params, ensure_ascii=False)
        
      tool_descs_list.append(
         TOOL_DESC.format(name=name, description=desc, parameters=params_str)
      )

   tool_descs_final = "\n\n".join(tool_descs_list)
   sys_prompt_tools = sys_prompt.format(tool_descs=tool_descs_final)
   print("构建系统提示词完成")
   print("==================")
   print(sys_prompt_tools)
   print("==================")
   return sys_prompt_tools
    
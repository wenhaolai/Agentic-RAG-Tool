from fastmcp import FastMCP
from typing import Annotated, Any, Literal
from pydantic import Field

from src.retrieval.es_wiki_search import ESWikiSearcher
from src.retrieval.hybrid_search import HybridSearcher

# Initialize the FastMCP server
mcp = FastMCP()

# Initialize the searcher once globally when the server starts
print("正在初始化全局 HybridSearcher 实例...")
searcher = HybridSearcher()
print("全局 HybridSearcher 实例加载完成。")

@mcp.tool(name="wiki_rag")
def wiki_rag(
    input: Annotated[
        str,
        Field(
            default=...,
            description="""The name of an entity or query keyword to search for in the wiki."""
        )
]):
    """
    Searches the wiki for information about the specified entity or query keyword.
    """
    try:
        # Use the global instance instead of recreating it
        results = searcher.search(input)
        print(results)
        return results
    except Exception as e:
        print(f"Error occurred while searching wiki: {e}")
        return {"error": str(e)}
    

if __name__=="__main__":
    mcp.run(host="0.0.0.0", port=8000, path="/mcp", transport="streamable-http")
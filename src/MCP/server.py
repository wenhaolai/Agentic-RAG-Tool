from fastmcp import FastMCP
from typing import Annotated, Any, Literal
from pydantic import Field

from src.retrieval.es_wiki_search import ESWikiSearcher

# Initialize the FastMCP server
mcp = FastMCP()

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
        searcher = ESWikiSearcher()
        return searcher.format_results(searcher.search(input))
    except Exception as e:
        print(f"Error occurred while searching wiki: {e}")
        return {"error": str(e)}
    

if __name__=="__main__":
    mcp.run(host="0.0.0.0", port=8000, path="/mcp", transport="streamable-http")
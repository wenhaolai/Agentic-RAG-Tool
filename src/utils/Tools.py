from src.retrieval.es_wiki_search import ESWikiSearcher
import logging

# Configure logger
logger = logging.getLogger(__name__)

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
        # Initialize searcher once to reuse connection
        try:
            self.searcher = ESWikiSearcher()
        except Exception as e:
            logger.error(f"Failed to initialize ESWikiSearcher: {e}")
            self.searcher = None

    def _tools(self):
        tools = [
            {
                "name_for_human": "WikiPedia",
                "name_for_model": "Wiki_RAG",
                "description_for_model": "Using this tool, you can search for wikipedia knowledge. Please combine the retrieved knowledge to assist you in answering.",
                "parameters": [
                    {
                        "name": "input",
                        "description": "A normalized name of an entity or query keyword.",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            }
        ]
        return tools

    def Wiki_RAG(self, input: str) -> str:
        """
        Tool implementation for Wiki_RAG.
        Returns formatted search results or error message.
        """
        if not self.searcher:
            return "Error: Search service is not available (initialization failed)."

        try:
            results = self.searcher.search(input)
            # print("==================")
            # print(results)
            # print("==================")
            return self.searcher.format_results(results)
        except Exception as e:
            logger.error(f"Wiki_RAG tool execution failed: {e}")
            return f"Error executing search: {str(e)}"
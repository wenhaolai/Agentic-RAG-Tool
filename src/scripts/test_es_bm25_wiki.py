from elasticsearch import Elasticsearch
from elasticsearch import exceptions

def test_search(query_text):
    try:
        # Connect to Elasticsearch
        es = Elasticsearch(["http://localhost:9200"])
        index_name = "wiki_en"
        
        # Simple ping check
        if not es.ping():
            print("Error: Could not connect to Elasticsearch at localhost:9200")
            return

        if not es.indices.exists(index=index_name):
            print(f"Error: Index '{index_name}' does not exist.")
            return

        print(f"Searching for: '{query_text}' in index '{index_name}'...")

        # BM25 search query
        # We search both 'title' and 'text' fields, boosting 'title' slightly.
        body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "text"], # Boost search relevance for title
                    "type": "best_fields"
                }
            },
            "_source": ["id", "title", "url", "text"], # Retrieve fields
            "size": 3
        }

        # Execute search
        response = es.search(index=index_name, body=body)
        
        hits = response['hits']['hits']
        total_hits = response['hits']['total']['value']
        
        print(f"Found {total_hits} matches. Showing top {len(hits)}:")
        print("-" * 60)
        
        if not hits:
            print("No results found.")
        
        for i, hit in enumerate(hits):
            score = hit['_score']
            source = hit['_source']
            title = source.get('title', 'No Title')
            url = source.get('url', 'No URL')
            # Truncate text for cleaner display
            text = source.get('text', '')[:300].replace('\n', ' ') + "..." 
            
            print(f"Rank {i+1} | Score: {score:.4f}")
            print(f"Title: {title}")
            print(f"URL:   {url}")
            print(f"Text:  {text}")
            print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test queries
    test_search("Python (programming language)")



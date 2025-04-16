# search_builder.py
from typing import List, Dict

def bm25_pipeline(keywords: List[str], k: int = 300) -> List[Dict]:
    """
    Build the same boosted compound $search you verified in Atlas Playground.
    """
    query = " ".join(keywords)

    return [
        {
            "$search": {
                "index": "default_bm25",
                "compound": {
                    "should": [
                        { "text": { "query": query, "path": "skills",
                                    "score": { "boost": { "value": 5 } } } },
                        { "text": { "query": query, "path": "projects.title",
                                    "score": { "boost": { "value": 3 } } } },
                        { "text": { "query": query, "path": "experience.title" } },
                        { "text": { "query": query, "path": "summary" } },
                        { "text": { "query": query,
                                    "path": "name",
                                    "score": { "boost": { "value": 8 } } } }
                    ]
                }
            }
        },
        { "$limit": k },
        {
            "$project": {
                "_id": 1, "name": 1, "skills": 1,
                "score": { "$meta": "searchScore" }
            }
        }
    ]

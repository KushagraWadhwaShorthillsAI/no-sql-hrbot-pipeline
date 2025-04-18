from typing import List, Dict

def bm25_pipeline(keywords: list[str], fields: list[str] = None, boost: int = 2, min_match: int = 2, top_k :int = 200) -> dict:
    if not fields:
        fields = [
            "name", "location", "summary", "skills",
            "experience.title", "experience.description", "experience.company",
            "projects.title", "projects.description",
            "education.degree", "education.institution", "education.year",
            "certifications.title", "certifications.issuer"
        ]

    should_clauses = []
    for kw in keywords:
        should_clauses.append({
            "text": {
                "query": kw,
                "path": fields,
                "score": { "boost": { "value": boost } }
            }
        })

    return [
        {
            "$search": {
                "index": "bm25_static",
                "compound": {
                    "should": should_clauses,
                    "minimumShouldMatch": min(min_match, len(should_clauses))
                }
            }
        },
        {
            "$limit": top_k
        }
    ]
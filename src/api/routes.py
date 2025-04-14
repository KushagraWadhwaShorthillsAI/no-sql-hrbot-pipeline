from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr
from src.core.pipeline import NoSQLQueryPipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=3, max_length=300)
    use_gemini: bool = False

@app.post("/query")
async def run_query(request: QueryRequest):
    try:
        pipeline = NoSQLQueryPipeline(use_gemini=request.use_gemini)
        result = await pipeline.run(request.query)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "query": request.query,
            "answer": result["answer"],
            "relevant_resumes": result["reranked_resumes"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ResearchRequest, ResearchResponse
from agents.orchestrator import run_research
from dotenv import load_dotenv
import uvicorn
import time

load_dotenv()

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="AI system that searches ArXiv,retrieves papers,and synthesizes research summaries using LangGraph Multi-Agent pipeline.",
    version="1.0.0"
)

# CORS(Cross-Origin Resource Sharing):
# allow browsers to access this API from another origin(different domain,port,or protocol)
# without this,browsers block frontend->backend requests. 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # "*"= allow requests from any frontend(development only).    
    allow_methods=["*"], # allow all HTTP methods(GET,POST,PUT,DELETE,etc.)
    allow_headers=["*"] # allow all request headers(content-Type,authorisation,etc.)
)

# server running check route 
@app.get("/")
def root():
    return {
        "message" : "Multi-Agent Research Assistant API",
        "status"  : "running",
        "docs"    : "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status"   : "healthy",
        "timestamp": time.time()
    }



@app.post("/research")
def research(request: ResearchRequest):
    """
    Main research endpoint.
    Args:
        request: ResearchRequest with query string
    Returns:
        ResearchResponse with findings,sources,score
    """

    # empty querycheck
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # query very short check
    if len(request.query.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Query too short — please be more specific"
        )


    print(f"NEW REQUEST RECEIVED")
    print(f"Query: {request.query}")


    try:
        # langGraph orchestrator call
        start_time = time.time()
        result = run_research(request.query)
        end_time = time.time()

        time_taken = round(end_time - start_time, 2)
        print(f"\nRequest completed in {time_taken} seconds")

        return {
            "query"        : request.query,
            "final_answer" : result,
            "papers_found" : result.get("papers_analyzed", 0),
            "iterations"   : result.get("iterations_taken", 0),
            "time_taken"   : time_taken
        }

    # prints this when research fails internally 
    except Exception as e:
        print(f"Error in research: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )


# server run 
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True # to auto start on code change
    )
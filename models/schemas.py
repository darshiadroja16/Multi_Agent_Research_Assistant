from typing import TypedDict, List, Optional
from pydantic import BaseModel
# TypedDict is used for data validation. data validation means wrong type of daqta nhi pass kr skte fileds me agar normally python use karoge to error nhi dega wrong type of data pass krne pe. 
# 

# researchstate is the state which stores all the results of node's call result
# reserachstate is the shared state between whole system or graph it passes the dictionary in graph
# every node updates the state 
class ResearchState(TypedDict):
    query: str
    papers: List[dict]
    rag_results: List[dict]
    summary: dict
    critique: dict
    iteration: int
    final_answer: dict



#request from frontend 
class ResearchRequest(BaseModel):
    query: str
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Compare transformer and LSTM for NLP tasks"
            }
        }


# response to frontend 
class ResearchResponse(BaseModel):
    query: str
    final_answer: dict
    papers_found: int
    iterations: int
    time_taken : float

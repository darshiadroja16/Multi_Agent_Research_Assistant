import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.graph import StateGraph,END      
from models.schemas import ResearchState  
from tools.vector_store import (add_papers_to_store,search_similar_chunks,clear_collection)
from tools.llm_tool import (generate_summary,critique_summary)
import arxiv                   
import time
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================================================================
# Nodes For the Graph
# ==========================================================================================
# Node is a specialised worker or agent.
# every Node takes data from state, does the work and adds new data to state
# initially state is empty then search node start to fill it by returning the data 

# 1st Node : Search Node
def search_node(state:ResearchState) -> dict:
    """
    Input  : state["query"]
    Output : state["papers"]
    """

    query =state["query"]
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=6,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers =[]

        for result in client.results(search):
            papers.append({
                "title"    : result.title,
                "abstract" : result.summary,
                "authors"  : [str(a) for a in result.authors[:3]],
                "url"      : result.entry_id,
                "published": str(result.published.date()),
            })

        logger.info(f"\nPapers found:{len(papers)}")
        for i,p in enumerate(papers):
            logger.info(f"[{i+1}] {p['title'][:55]}...")

    except Exception as e:
        logger.error(f"ArXiv search error:{e}")
        papers =[]

    return {"papers": papers}

# 2nd Node : Rag Node (store + relevant chunks retrieve)
def rag_node(state: ResearchState) -> dict:
    """
    Input  : state["papers"] + state["query"]
    Output : state["rag_results"]
    """
    
    papers = state["papers"]
    query  = state["query"]
    session_id = state.get("session_id", "default_collection")

    if not papers:
        logger.warning("no papers to index")
        return {"rag_results": []}
    
    collection_name = f"research_{session_id}"
    clear_collection(collection_name)

    logger.info(f"Indexing {len(papers)} papers in collection {collection_name}...")
    add_papers_to_store(papers, collection_name)

    print(f"\nretrieving relevant chunks....")
    rag_results = search_similar_chunks(
        query,
        n_results= 5,
        collection_name= collection_name
    )

    print(f"\nchunks retrived: {len(rag_results)}")
    for i,r in enumerate(rag_results):
        print(f"[{i+1}] ({r['similarity']}) {r['content'][:55]}...")
    
    return {"rag_results": rag_results}


# 3rd Node : Summarizer Node
def summarizer_node(state:ResearchState) ->dict:
    """
    Input  : state["query"] + state["rag_results"]
    Output : state["summary"]
    """

    query = state["query"]
    rag_results =state["rag_results"]
    previous_summary = state.get("summary", {})
    critique = state.get("critique", {})

    if not rag_results:
        logger.warning("No RAG results — returning default summary")
        return {
            "summary": {
                "key_findings"           : ["No relevant papers found"],
                "methodology_comparison" : "N/A",
                "research_gaps"          : ["Insufficient data"],
                "recommendations"        : ["Try a different query"],
                "confidence_score"       : 1
            }
        }
    
    logger.info(f"Chunks available: {len(rag_results)}")
    logger.info("Calling Groq LLM for summary...")

    summary = generate_summary(
        query, 
        rag_results, 
        previous_summary=previous_summary if previous_summary else None, 
        critique=critique if critique else None
    )

    logger.info(f"\nSummary generated!")
    logger.info(f"  Key findings  : {len(summary.get('key_findings', []))}")
    logger.info(f"  Research gaps : {len(summary.get('research_gaps', []))}")
    logger.info(f"  Confidence    : {summary.get('confidence_score', 0)}/10")

    return {"summary": summary}



# 4th Node : Critic Node
def critic_node(state:ResearchState) -> dict:
    """
    Input  : state["query"] + state["summary"]
    Output : state["critique"] + state["iteration"]
    """

    query = state["query"]
    summary = state["summary"]
    current_iteration = state["iteration"]

    logger.info("Evaluating summary quality...")

    critique = critique_summary(query, summary)

    score          = critique.get("score", 0)
    needs_revision = critique.get("needs_revision", False)

    logger.info(f"\n  Score : {score}/10")
    logger.info(f"Needs Revision : {needs_revision}")

    if critique.get("issues"):
        logger.info(f"  Issues:")
        for issue in critique["issues"]:
            logger.info(f" -> {issue}")

    if current_iteration >= 2:
        print("\nMax iterations reached — accepting as is")
        critique["needs_revision"] = False

    return {
        "critique" : critique,
        "iteration": current_iteration + 1
    }

# 5th Edge : Conditional Edge
def should_revise(state: ResearchState) -> str:
    """
    Return:
      "revise" → summarize node pe wapas jao
      "finish" → format node pe jao
    """

    score = state["critique"].get("score", 10)
    needs_revision = state["critique"].get("needs_revision", False)
    iteration = state["iteration"]

    if needs_revision and iteration < 3:
        logger.info(f"\n↩ Score{score}/10—Looping back to summarizer.....")
        return "revise"
    else:
        logger.info(f"\nScore {score}/10—Moving to format...")
        return "finish"
    

# 6th Node : Formate Node
def format_node(state: ResearchState) -> dict:
    """
    Input  : full state
    Output : state["final_answer"]
    """

    summary  = state["summary"]
    papers   = state["papers"]
    critique = state["critique"]
    session_id = state.get("session_id", "default_collection")
    
    # Cleanup vector store
    collection_name = f"research_{session_id}"
    clear_collection(collection_name)

    final_answer = {

        "key_findings": summary.get(
            "key_findings", []
        ),
        "methodology_comparison": summary.get(
            "methodology_comparison", ""
        ),
        "research_gaps": summary.get(
            "research_gaps", []
        ),
        "recommendations": summary.get(
            "recommendations", []
        ),
        
        "quality_score"   : critique.get("score", 0),
        "iterations_taken": state["iteration"],
        "papers_analyzed" : len(papers),

        "sources": [
            {
                "title"    : p["title"],
                "url"      : p["url"],
                "published": p["published"],
                "authors"  : p["authors"]
            }
            for p in papers[:5]
        ]
    }

    logger.info(f"Quality Score : {final_answer['quality_score']}/10")
    logger.info(f"Papers Used   : {final_answer['papers_analyzed']}")
    logger.info(f"Iterations    : {final_answer['iterations_taken']}")

    return {"final_answer": final_answer}



# ==============================================================================
# Graph Buliding 
# ==============================================================================

def build_graph():
    '''
    START -> search -> rag ->summarize -> critic 
                               ^             |
                               | -- revise <-| (score < 7)
                                      |
                                      |-->  finish -> formate -> END
    '''
    graph = StateGraph(ResearchState) # graph initialization 

    # adding nodes 
    graph.add_node("search",search_node)
    graph.add_node("rag",rag_node)
    graph.add_node("summarize",summarizer_node)
    graph.add_node("critic",critic_node)
    graph.add_node("format",format_node)

    # Entry point
    graph.set_entry_point("search")

    # connecting edges
    # Edges are the connecting bridge between nodes to pass the message from one state to another 
    graph.add_edge("search","rag")
    graph.add_edge("rag","summarize")
    graph.add_edge("summarize","critic")

    # conditional edge 
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise" : "summarize",
            "finish" : "format"
        }
    )

    graph.add_edge("format",END)

    compiled =graph.compile()
    print("Graph compiled successfully")
    return compiled


# The final function which will be called in main.py 
def run_research(query: str) -> dict:
    """
    Query lo, graph run karo, final answer return karo.
    Args:
        query : research question string
    Returns:
        final_answer dict with all findings + sources
    """

    logger.info(f"NEW RESEARCH QUERY")
    logger.info(f"Query: {query}")

    # build the graph
    graph = build_graph()

    # initial state (only query will be filled)
    initial_state: ResearchState = {
        "session_id"  : uuid.uuid4().hex,
        "query"       : query,
        "papers"      : [],
        "rag_results" : [],
        "summary"     : {},
        "critique"    : {},
        "iteration"   : 0,
        "final_answer": {}
    }

    # to invoke graph
    final_state = graph.invoke(initial_state)

    return final_state["final_answer"]


# just for Test
if __name__ == "__main__":

    result = run_research(
        "Compare transformer and LSTM for NLP tasks"
    )

    print("FINAL RESULTS")
    print("-" * 50)

    print(f"\nQuality Score   : {result['quality_score']}/10")
    print(f"Papers Analyzed : {result['papers_analyzed']}")
    print(f"Iterations      : {result['iterations_taken']}")

    print(f"\nKEY FINDINGS:")
    for i, f in enumerate(result['key_findings']):
        print(f"{i+1}. {f}")

    print(f"\nRESEARCH GAPS:")
    for g in result['research_gaps']:
        print(f"-> {g}")

    print(f"\nRECOMMENDATIONS:")
    for r in result['recommendations']:
        print(f"-> {r}")

    print(f"\nSOURCES:")
    for s in result['sources']:
        print(f"•{s['title'][:55]}...")
        print(f"{s['url']}")
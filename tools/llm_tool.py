from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.3-70b-versatile"

SUMMARIZER_SYSTEM = """You are an expert research scientist
specializing in analyzing and synthesizing academic papers in NLP,
machine learning, and artificial intelligence.

STRICT RULES:
1. Answer ONLY from the provided context
2. Never add information not present in context
3. If something is unclear, explicitly state it
4. Always respond in valid JSON format
5. Be precise and technical"""


CRITIC_SYSTEM = """You are a strict research quality evaluator
with expertise in academic paper analysis.

Your job is to evaluate research summaries for:
- Accuracy: Do findings match the source context?
- Completeness: Are all important aspects covered?
- Clarity: Is the summary easy to understand?
- Gap identification: Are research gaps properly noted?

Always respond in valid JSON format."""


def generate_summary(
    query: str,
    context_chunks: list[dict]
) -> dict:
    
    # Returns: dict with key_findings, methodology, gaps, recommendations
    # to provide context it joins all the chunks retrived for query that is passed in the function args
    # so it can be passed directly to the llm as context retrived from chromadb
    context = "\n\n---\n\n".join([
        f"Source: {c['metadata'].get('title', 'Unknown')}\n"
        f"Relevance: {c['similarity']}\n"
        f"Content: {c['content']}"
        for c in context_chunks
    ])

    user_prompt = f"""Analyze the following research paper excerpts
and synthesize a comprehensive research summary.

Research Query: {query}

Paper Context:
{context}

Think step by step:
1. What are the key technical findings across papers?
2. How do the methodologies differ or complement?
3. What gaps or limitations are mentioned?
4. What would you recommend based on findings?

Respond in this exact JSON format:
{{
  "key_findings": [
    "specific finding 1",
    "specific finding 2",
    "specific finding 3"
  ],
  "methodology_comparison": "detailed comparison string",
  "research_gaps": [
    "gap 1",
    "gap 2"
  ],
  "recommendations": [
    "recommendation 1",
    "recommendation 2"
  ],
  "confidence_score": <number 1-10>
}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content
    return json.loads(result)


def critique_summary(
    query: str,
    summary: dict
) -> dict:
    # Returns: dict with score, issues, needs_revision
    user_prompt = f"""Evaluate this research summary for quality.

Original Research Query: {query}

Summary to Evaluate:
{json.dumps(summary, indent=2)} # return dictionary ko readable json string me convert krna 

Evaluate strictly based on:
1. Are findings specific and accurate?
2. Is methodology comparison meaningful?
3. Are research gaps clearly identified?
4. Are recommendations actionable?

Respond in this exact JSON format:
{{
  "score": <number 1-10>,
  "accuracy_score": <number 1-10>,
  "completeness_score": <number 1-10>,
  "clarity_score": <number 1-10>,
  "issues": [
    "specific issue 1 if any",
    "specific issue 2 if any"
  ],
  "needs_revision": <true if score < 7 else false>,
  "improvement_suggestions": [
    "specific suggestion 1",
    "specific suggestion 2"
  ]
}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=800,
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content
    return json.loads(result)


if __name__ == "__main__":

    # fake context chunks—vector_store format 
    test_chunks = [
        {
            "content"   : """Transformer architecture uses multi-head
            self-attention achieving 28.4 BLEU on English-German translation.
            Training required 3.5 days on 8 P100 GPUs.""",
            "metadata"  : {"title": "Attention is All You Need"},
            "similarity": 0.92
        },
        {
            "content"   : """LSTM uses gating mechanisms to solve vanishing
            gradient. Cannot parallelize training unlike transformers.
            Effective for sequential tasks but slower to train.""",
            "metadata"  : {"title": "LSTM Networks"},
            "similarity": 0.85
        },
    ]

    query = "Compare transformer and LSTM for NLP tasks"

    # test summarizer........
    print("\n--- Testing generate_summary() ---")
    summary = generate_summary(query, test_chunks)
    print(f"Key Findings: {len(summary['key_findings'])} found")
    print(f"Confidence  : {summary['confidence_score']}/10")
    for f in summary['key_findings']:
        print(f"  → {f}")

    # test critic.........
    print("\n--- Testing critique_summary() ---")
    critique = critique_summary(query, summary)
    print(f"Score          : {critique['score']}/10")
    print(f"Needs Revision : {critique['needs_revision']}")
    print(f"Issues         : {critique.get('issues', [])}")

    print("\nllm_tool.py working correctly!")
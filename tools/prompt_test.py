from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key =os.getenv("GROQ_API_KEY"))

def llm_call(user_prompt, system_prompt, json_mode=False):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content

# Sample context — Day 2 ke papers 
SAMPLE_CONTEXT = """
Paper 1: Attention is All You Need (2017)
- Transformer architecture uses only attention mechanisms
- Achieved 28.4 BLEU on English-German translation
- Training: 3.5 days on 8 P100 GPUs
- Encoder: 6 layers, each with multi-head attention + FFN
- Eliminated recurrence completely

Paper 2: BERT (2018)
- Bidirectional transformer encoder
- Pre-training: Masked Language Modeling + Next Sentence Prediction
- BERT-base: 12 layers, 768 hidden, 12 heads
- Fine-tuned BERT: 80.5 on GLUE benchmark
- Pre-train once, fine-tune for any task

Paper 3: LSTM Networks
- Solve vanishing gradient via gating mechanisms
- Three gates: forget, input, output
- Sequential processing — cannot parallelize
- Good for time series, speech recognition
- Replaced by transformers for most NLP tasks
"""

SAMPLE_QUERY = "Compare transformer and LSTM architectures for NLP tasks"


print("Test 1 - Baseline")

baseline = llm_call(
    system_prompt="You are an assistant.",
    user_prompt=f"Summarize findings about: {SAMPLE_QUERY}"
)
print(baseline)


print("\n" + "=" * 60)
print("Test 2 - Role + grounding")

role_grounded = llm_call(
    system_prompt="""You are an expert research scientist
specializing in NLP and deep learning architectures.
Your job is to synthesize research findings accurately.
STRICT RULE: Use ONLY information from the provided context.
Never add outside knowledge.""",

    user_prompt=f"""Context from research papers:
{SAMPLE_CONTEXT}

Research Question: {SAMPLE_QUERY}

Provide a detailed comparison based strictly on the context."""
)

print(role_grounded)


print("\n" + "=" * 60)
print("TEST 3 — Role + Grounding + Chain Of Thought")

cot_response = llm_call(
    system_prompt="""You are an expert research scientist in NLP.
Answer ONLY from provided context. Never use outside knowledge.""",

    user_prompt=f"""Context:
{SAMPLE_CONTEXT}

Question: {SAMPLE_QUERY}

Think step by step:
Step 1: Identify architectural differences
Step 2: Compare performance metrics from context
Step 3: Identify use case suitability
Step 4: Summarize key takeaways

Provide your analysis following these steps."""
)
print(cot_response)


print("\n" + "=" * 60)
print("TEST 4 — Production Summarizer Prompt(JSON)")

SUMMARIZER_SYSTEM = """You are an expert research scientist
specializing in analyzing and synthesizing academic papers.

STRICT RULES:
1. Answer ONLY from the provided context
2. Never add information not present in context
3. If something is unclear, say so explicitly
4. Always respond in valid JSON format"""

SUMMARIZER_USER = f"""Analyze the following research paper excerpts and
synthesize a comprehensive research summary.

Research Query: {SAMPLE_QUERY}

Paper Context:
{SAMPLE_CONTEXT}

Think step by step:
1. What are the key technical findings?
2. How do the methodologies differ?
3. What gaps or limitations exist?
4. What do you recommend?

Respond in this exact JSON format:
{{
  "key_findings": [
    "finding 1",
    "finding 2",
    "finding 3"
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

summary_json = llm_call(
    system_prompt=SUMMARIZER_SYSTEM,
    user_prompt=SUMMARIZER_USER,
    json_mode=True
)

parsed_summary = json.loads(summary_json)
print("\nParsed Summary:")
print(f"\nKey Findings:")
for f in parsed_summary['key_findings']:
    print(f"  → {f}")
print(f"\nMethodology: {parsed_summary['methodology_comparison'][:100]}...")
print(f"\nResearch Gaps:")
for g in parsed_summary['research_gaps']:
    print(f"  → {g}")
print(f"\nConfidence Score: {parsed_summary['confidence_score']}/10")



print("\n" + "=" * 60)
print("TEST 5 — Production Critic Prompt(JSON)")

CRITIC_SYSTEM = """You are a strict research quality evaluator.
Your job is to evaluate research summaries for accuracy,
completeness, and clarity.
Always respond in valid JSON format."""

CRITIC_USER = f"""Evaluate this research summary for quality.

Original Query: {SAMPLE_QUERY}

Summary to Evaluate:
{summary_json}

Evaluate based on:
1. Accuracy — findings match the research context?
2. Completeness — all important aspects covered?
3. Clarity — easy to understand?
4. Gaps identified properly?

Respond in this exact JSON format:
{{
  "score": <number 1-10>,
  "accuracy_score": <number 1-10>,
  "completeness_score": <number 1-10>,
  "clarity_score": <number 1-10>,
  "issues": [
    "issue 1 if any",
    "issue 2 if any"
  ],
  "needs_revision": <true or false>,
  "improvement_suggestions": [
    "suggestion 1",
    "suggestion 2"
  ]
}}

needs_revision should be true if score < 7"""

critic_json = llm_call(
    system_prompt=CRITIC_SYSTEM,
    user_prompt=CRITIC_USER,
    json_mode=True
)

parsed_critic = json.loads(critic_json)
print(f"\nCritic Evaluation:")
print(f"  Overall Score    : {parsed_critic['score']}/10")
print(f"  Accuracy         : {parsed_critic['accuracy_score']}/10")
print(f"  Completeness     : {parsed_critic['completeness_score']}/10")
print(f"  Clarity          : {parsed_critic['clarity_score']}/10")
print(f"  Needs Revision   : {parsed_critic['needs_revision']}")
print(f"  Issues:")
for issue in parsed_critic.get('issues', []):
    print(f"    → {issue}")
print(f"  Suggestions:")
for sug in parsed_critic.get('improvement_suggestions', []):
    print(f"    → {sug}")


    
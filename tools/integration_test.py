from embedder import get_embeddings_batch
from vector_store import (
    add_papers_to_store,
    search_similar_chunks,
    clear_collection
)
from llm_tool import generate_summary, critique_summary

print("All Tools Together")

# test papers
papers = [
    {
        "title"   : "Attention is All You Need",
        "abstract": """Transformer uses self-attention mechanisms.
        Achieved 28.4 BLEU English-German. Training 3.5 days
        on 8 P100 GPUs. Encoder-decoder with 6 layers each.
        Multi-head attention with 8 heads in base model.""",
        "url"     : "https://arxiv.org/abs/1706.03762",
        "authors" : ["Vaswani et al."]
    },
    {
        "title"   : "BERT Pre-training",
        "abstract": """BERT uses bidirectional transformer encoder.
        Masked language modeling pre-training on 15% tokens.
        BERT-large: 24 layers, 1024 hidden, 16 heads.
        Fine-tuned BERT: 80.5 GLUE, 93.2 SQuAD F1.""",
        "url"     : "https://arxiv.org/abs/1810.04805",
        "authors" : ["Devlin et al."]
    },
]

query = "How do transformer models handle attention?"

# step 1 — Store
print("\n[Step 1] Storing papers..")
clear_collection("integration_test")
add_papers_to_store(papers, "integration_test")

# step 2 — Retrieve
print("\n[Step 2] Retrieving relevant chunks....")
chunks = search_similar_chunks(
    query,
    n_results=4,
    collection_name="integration_test"
)
print(f"Found {len(chunks)} chunks")


# step 3 — Summarize
print("\n[Step 3] Generating summary...")
summary = generate_summary(query, chunks)
print(f"Key findings: {len(summary['key_findings'])}")
print(f"Confidence  : {summary['confidence_score']}/10")

for f in summary['key_findings']:
    print(f"- {f}")

# s 6tep 4 — Critique
print("\n[Step 4] Critiquing summary...")
critique = critique_summary(query, summary)
print(f"Score          : {critique['score']}/10")
print(f"Needs Revision : {critique['needs_revision']}")

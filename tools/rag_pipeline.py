from google import genai
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os
import time
import json

load_dotenv() # to read .env file

# API connections
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# store data
chroma_client = chromadb.PersistentClient(path="./chroma_db") 

#create or open collection
collection = chroma_client.get_or_create_collection(name="rag_pipeline_test",metadata={"hnsw:space":"cosine"})

# Helper functions

# to get the embeddings 
def get_embedding(text):
    time.sleep(0.3)  # Rate limit to avoid too many request at a time
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

# chunking the documents
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        boundary = text.rfind('.', start, end)
        if boundary != -1 and boundary > start + chunk_size // 2:
            chunk = text[start:boundary + 1].strip()
            next_start = boundary + 1
        else:
            chunk = text[start:end].strip()
            next_start = end
        if chunk:
            chunks.append(chunk)
        start = next_start - overlap
    return chunks


# Documents load -> chunk -> embed -> store in chromadb 
def index_documents(documents):
    print("PHASE 1 — INDEXING")
    print("Indexing documents...")

    all_chunks = []
    all_ids = []
    chunk_counter = 0

    for doc_idx, doc_text in enumerate(documents):
        chunks = chunk_text(doc_text, chunk_size=400, overlap=50)

        print(f"\nDocument {doc_idx + 1}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk[:60]}...")
            all_chunks.append(chunk)
            all_ids.append(f"doc{doc_idx}_chunk{chunk_counter}")
            chunk_counter += 1

    print(f"\nTotal chunks: {len(all_chunks)}")

    embeddings = []
    for i, chunk in enumerate(all_chunks):
        emb = get_embedding(chunk)
        embeddings.append(emb)
        print(f"  [{i+1}/{len(all_chunks)}] embedded ✓")

    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids
    )

    print(f"\nIndexing complete!")

# Query -> embed -> chromadb search -> top chunks
def retrieve_chunks(query, n_results=4):

    print("PHASE 2 — RETRIEVAL")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results['documents'][0]


# context chunks + Query -> Groq LLM -> Answer
def generate_answer(query, context_chunks):
    print("PHASE 3 — GENERATION")

    context = "\n\n---\n\n".join([
        f"Chunk {i+1}:\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])
    prompt = f"""You are a research assistant analyzing academic papers.

STRICT RULE: Answer ONLY using information from the Context below.
If the answer is not in the context, respond with exactly:
"This information was not found in the provided research documents."

Context:
{context}

Question: {query}

Answer (from context only):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Answer strictly from provided context only. Never use outside knowledge."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=600
    )

    answer = response.choices[0].message.content
    print(f"Answer: {answer}\n")
    return answer


# Complete RAG pipeline
# Query → Retrieve → Generate → Answer
    
def rag_query(query):
    
    print(f"RAG QUERY: {query}")

    chunks = retrieve_chunks(query, n_results=3)
    answer = generate_answer(query, chunks)
    
    return answer



research_documents = [
    """
    The Transformer architecture was introduced in the paper Attention is All
    You Need by Vaswani et al in 2017. The model relies entirely on attention
    mechanisms dispensing with recurrence and convolutions completely.
    The encoder has 6 identical layers each with multi-head self-attention
    and position-wise feed-forward network with residual connections.
    The decoder also has 6 layers and adds cross-attention over encoder output.
    Training used Adam optimizer with beta1 0.9 and beta2 0.98.
    The big transformer model achieved 28.4 BLEU on English to German translation
    and 41.0 BLEU on English to French using WMT 2014 dataset.
    Training took 3.5 days on 8 NVIDIA P100 GPUs.
    """,

    """
    BERT stands for Bidirectional Encoder Representations from Transformers.
    It was introduced by Devlin et al from Google in 2018.
    BERT pre-trains deep bidirectional representations by conditioning jointly
    on both left and right context in all layers unlike previous models.
    Pre-training uses two tasks: Masked Language Modeling where 15 percent
    of tokens are masked and model predicts them, and Next Sentence Prediction.
    BERT base model has 12 transformer layers 768 hidden size and 12 attention heads.
    BERT large has 24 layers 1024 hidden size and 16 attention heads.
    Fine-tuned BERT achieved 80.5 on GLUE benchmark and 93.2 F1 on SQuAD 1.1.
    BERT showed that pre-training then fine-tuning paradigm works extremely well.
    """,

    """
    GPT or Generative Pre-trained Transformer uses a unidirectional transformer
    decoder architecture pre-trained on next token prediction objective.
    Unlike BERT which is bidirectional GPT only attends to previous tokens
    making it naturally suited for text generation tasks.
    GPT-3 released by OpenAI in 2020 has 175 billion parameters trained on
    570 GB of text data including Common Crawl Books and Wikipedia.
    GPT-3 demonstrated remarkable few-shot learning capabilities achieving
    strong results on many NLP tasks without any fine-tuning or gradient updates.
    The model uses 96 attention layers 96 heads and embedding dimension of 12288.
    Context window of GPT-3 is 2048 tokens.
    """,

    """
    LSTM or Long Short-Term Memory networks were introduced by Hochreiter and
    Schmidhuber in 1997 to solve the vanishing gradient problem in vanilla RNNs.
    LSTMs use a cell state and three gating mechanisms to control information flow.
    The forget gate decides what information to discard from the cell state
    using a sigmoid function applied to concatenation of previous hidden state
    and current input. Output is between 0 and 1 where 0 means forget completely.
    The input gate decides what new information to store in the cell state.
    The output gate controls what part of cell state goes to hidden state.
    LSTMs were state of the art for sequential tasks like machine translation
    speech recognition and language modeling before transformers replaced them.
    Bidirectional LSTMs process sequence in both directions for better context.
    """,
]

print("Checking if already indexed...")
if collection.count() == 0:
    index_documents(research_documents)
else:
    print(f"Already indexed! {collection.count()} chunks found. Skipping indexing.")

print("TESTING RAG PIPELINE")
rag_query("How does the transformer use attention mechanism?")
rag_query("What is the difference between BERT and GPT architecture?")
rag_query("How does LSTM handle the vanishing gradient problem?")
rag_query("What was the training time for the transformer model?")
rag_query("What is the weather in Mumbai today?")

print("\n--- testing with 1 chunk ---")
chunks_1 = retrieve_chunks("What is GPT-3's training data?", n_results=1)
generate_answer("What is GPT-3's training data?", chunks_1)

print("\n--- testing with 4 chunks ---")
chunks_4 = retrieve_chunks("What is GPT-3's training data?", n_results=4)
generate_answer("What is GPT-3's training data?", chunks_4)

print("\n--- 1 chunk ---")
chunks_1 = retrieve_chunks("Compare BERT and GPT architecture differences", n_results=1)
generate_answer("Compare BERT and GPT architecture differences", chunks_1)

print("\n--- 4 chunks ---")
chunks_4 = retrieve_chunks("Compare BERT and GPT architecture differences", n_results=4)
generate_answer("Compare BERT and GPT architecture differences", chunks_4)
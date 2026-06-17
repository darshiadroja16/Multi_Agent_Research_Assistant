from google import genai
from dotenv import load_dotenv
import os
import numpy as np

# ================================================
# SETUP — API key load karo, client banao
# ================================================
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ================================================
# PART 1 — Ek sentence embed karo
# ================================================

sentence = "Transformers use self-attention mechanism"

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentence
)

vector = result.embeddings[0].values

print("=" * 55)
print("PART 1 — SINGLE SENTENCE EMBEDDING")
print("=" * 55)
print(f"Sentence   : {sentence}")
print(f"Vector size: {len(vector)} numbers")
print(f"First 5    : {vector[:5]}")
print(f"Last 5     : {vector[-5:]}")


# ================================================
# PART 2 — Multiple sentences embed karo
# ================================================

sentences = [
    "Transformers use self-attention to process sequences",
    "BERT is a bidirectional transformer model",
    "GPT uses autoregressive decoding for generation",
    "LSTM uses gates to control information flow",
    "Random forests combine multiple decision trees",
    "Cricket is played between two teams of eleven",
]

embeddings = []
for s in sentences:
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=s
    )
    embeddings.append(res.embeddings[0].values)

print("\n" + "=" * 55)
print("PART 2 — MULTIPLE EMBEDDINGS")
print("=" * 55)
print(f"Total sentences embedded: {len(embeddings)}")
print(f"Each vector size        : {len(embeddings[0])}")
print("\nSentences stored as vectors:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s[:50]}")


# ================================================
# PART 3 — Cosine similarity manually
# ================================================

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    return dot / (mag1 * mag2)

print("\n" + "=" * 55)
print("PART 3 — COSINE SIMILARITY")
print("=" * 55)

score1 = cosine_similarity(embeddings[0], embeddings[1])
print(f"\nSentence 1: '{sentences[0][:40]}...'")
print(f"Sentence 2: '{sentences[1][:40]}...'")
print(f"Similarity: {score1:.4f}  ← HIGH expect karo")

score2 = cosine_similarity(embeddings[0], embeddings[3])
print(f"\nSentence 1: '{sentences[0][:40]}...'")
print(f"Sentence 4: '{sentences[3][:40]}...'")
print(f"Similarity: {score2:.4f}  ← MEDIUM expect karo")

score3 = cosine_similarity(embeddings[0], embeddings[5])
print(f"\nSentence 1: '{sentences[0][:40]}...'")
print(f"Sentence 6: '{sentences[5][:40]}...'")
print(f"Similarity: {score3:.4f}  ← LOW expect karo")

print("\nAll pairwise similarities with Sentence 1:")
for i, s in enumerate(sentences):
    score = cosine_similarity(embeddings[0], embeddings[i])
    bar = "█" * int(score * 20)
    print(f"  [{i}] {score:.4f} {bar} {s[:35]}")
from google import genai
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ================================================
# ChromaDB Setup
# ================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)

# ================================================
# Helper Functions
# ================================================

def get_embedding(text):
    """Ek text ka embedding lo Gemini se"""
    result = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def add_documents(documents, ids):
    """Documents ChromaDB mein store karo"""
    print(f"\nStoring {len(documents)} documents...")

    embeddings = []
    for i, doc in enumerate(documents):
        emb = get_embedding(doc)
        embeddings.append(emb)
        print(f"  Embedded [{i+1}/{len(documents)}]: {doc[:45]}...")

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Stored! Total in DB: {collection.count()}")

def search(query, n_results=3):
    """Query se similar documents dhundo"""
    print(f"\n{'='*55}")
    print(f"QUERY: '{query}'")
    print(f"{'='*55}")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results['documents'][0]
    distances = results['distances'][0]

    for i, (doc, dist) in enumerate(zip(docs, distances)):
        similarity = 1 - dist
        bar = "█" * int(similarity * 20)
        print(f"\nRank {i+1}")
        print(f"Similarity : {similarity:.4f} {bar}")
        print(f"Document   : {doc}")

    return results

# ================================================
# Test Karo
# ================================================

documents = [
    "Transformers use self-attention to process sequences in parallel",
    "BERT is a bidirectional transformer pre-trained on masked language modeling",
    "GPT uses autoregressive transformer decoder for text generation",
    "LSTM uses forget gate, input gate and output gate for sequential processing",
    "Random forests build multiple decision trees and merge their predictions",
    "Convolutional neural networks use filters to detect spatial features in images",
    "Attention mechanism allows model to focus on relevant parts of the input sequence",
    "Word2Vec learns word embeddings by predicting surrounding words in context",
    "Gradient descent minimizes loss function by updating weights iteratively",
    "Cricket is a bat and ball sport played between two teams of eleven players",
    "Stock market prices fluctuate based on supply demand and investor sentiment",
]

ids = [f"doc_{i}" for i in range(len(documents))]

add_documents(documents, ids)

search("transformer architecture and attention")
search("sequential data and memory")
search("sports and outdoor games")
search("image recognition deep learning")


# ================================================
# Interactive Search — Apni query daalo
# ================================================

print("\n" + "=" * 55)
print("INTERACTIVE SEARCH")
print("Type 'q' to quit")
print("=" * 55)

while True:
    query = input("\n🔍 Apni query daalo: ").strip()

    if query.lower() == 'q':
        print("Band kar rahe hain...")
        break

    if not query:
        print("Kuch toh likho!")
        continue

    search(query, n_results=3)
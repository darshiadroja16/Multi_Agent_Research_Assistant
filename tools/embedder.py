from google import genai
from dotenv import load_dotenv
import os
import time 

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Embedding_model = "gemini-embedding-001"

#--------------------- This functions will be used directly in other files for embedding -------------------------- 


# for single string
# used for query embedding 
def get_embedding(text: str ) ->list[float]:
    time.sleep(0.3)
    result = client.models.embed_content(model =Embedding_model,contents=text)
    return result.embeddings[0].values


# for list of string 
# input -> list of string  and output -> list of vectors(list of lists)
# used for embedding multiple papers 
def get_embeddings_batch(texts : list[str]) -> list[list[float]]:
    embeddings= []
    for i, text in enumerate(texts):
        embedding = get_embedding(text)
        embeddings.append(embedding)
        print(f"  Embedded [{i+1}/{len(texts)}]: {text[:50]}...")

    return embeddings

# This function will run just for this file means to print output of this file it will be useful  
if __name__ == "__main__":
    import numpy as np

    def cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / mag

    test_sentences = [
        "Transformers use self-attention mechanism",
        "BERT is based on transformer architecture",
        "Cricket is a bat and ball sport",
    ]

    print("Testing embedder.py...")
    embeddings = get_embeddings_batch(test_sentences)

    print(f"\nVector size: {len(embeddings[0])}")

    s1 = cosine_similarity(embeddings[0], embeddings[1])
    s2 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\nTransformer vs BERT similarity : {s1:.4f} (HIGH expected)")
    print(f"Transformer vs Cricket similarity: {s2:.4f} (LOW expected)")
    print("\n✅ embedder.py working correctly!")
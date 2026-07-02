
# # This part of file is for practice and understanding 

# from google import genai
# from dotenv import load_dotenv # to load api key from .env file to main memory for the use 
# import os #it provides python os module which helps in interacting with the operating system
# import numpy as np #math calculation


# load_dotenv() # to read .env file
# client = genai.Client(api_key = os.getenv("GEMINI_API_KEY")) #client is the connection built with the gemini api(gemini server) no we dont need to request again for future use (login is done) 


# # For one sentence

# sentence = "Transformers use self-attention mechanism"
# result = client.models.embed_content(model = "models/gemini-embedding-001",contents = sentence) #it stores the number of fields of vector of sentence 
# vector =result.embeddings[0].values #result stores the vector of the sentence in list form and [0] is used to access the first element of the 

# print(f"Sentence :{sentence}")
# print(f"vector:{len(vector)} no.")
# print(f"first 5values : {vector[:5]}")


# # For multiple sentences

# sentences =[
#     "Transformers use self-attention to process sequences",
#     "BERT is a bidirectional transformer model",
#     "GPT uses autoregressive decoding for generation",
#     "LSTM uses gates to control information flow",
#     "Random forests combine multiple decision trees",
#     "Cricket is played between two teams of eleven",
# ]
# embeddings = []
# for s in sentences:
#     result = client.models.embed_content(model = "models/gemini-embedding-001",contents= s)
#     embeddings.append(result.embeddings[0].values)

# print(f"Multiple sentences:{sentences}")
# print(f"Total sentences embedded:{len(embeddings)}")
# print(f"each vector lenght:{len(embeddings[0])}")
# for i ,s in enumerate (sentences):
#     print(f"[{i}]{s[:50]}")


# # Cosine similarity 

# def cosine_similarity(vec1,vec2):
#     dot = np.dot(vec1,vec2) 
#     mag1 = np.linalg.norm(vec1) # magnitude of vector 1
#     mag2 = np.linalg.norm(vec2) # magnitude of vector 2
#     return dot / (mag1*mag2)

# print(f"cosine similarity btw two sentence vectors")
# print(f"sentence 1 :'{sentences[0][:40]}...'")
# print(f"sentence 2 :'{sentences[1][:40]}...'")
# print(f"similarity:{cosine_similarity(embeddings[0],embeddings[1])}")


# print(f"sentence 1:'{sentences[0][:40]}...'")
# print(f"sentence 4:'{sentences[3][:40]}...'")   
# print(f"similarity:{cosine_similarity(embeddings[0],embeddings[3])}")   


# print(f"sentence 1:'{sentences[0][:40]}...'")
# print(f"sentence 6:'{sentences[5][:40]}...'")   
# print(f"similarity:{cosine_similarity(embeddings[0],embeddings[5])}")


# print(f"all the pairwise cosine similarities with sentence 1:")
# for i ,s in enumerate(sentences):
#     score = cosine_similarity(embeddings[0],embeddings[i])
#     bar = "#" * int (score*20)
#     print(f"[{i}] {score:.4f}{bar}{s[:35]}")



#--------------------------------------- Production file ------------------------------------------------------

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
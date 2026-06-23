
# This file is just for understanding purpose ,it will not be used in the project production 


from google import genai
from dotenv import load_dotenv # to load api key from .env file to main memory for the use 
import os #it provides python os module which helps in interacting with the operating system
import numpy as np #math calculation


load_dotenv() # to read .env file
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY")) #client is the connection built with the gemini api(gemini server) no we dont need to request again for future use (login is done) 


# For one sentence

sentence = "Transformers use self-attention mechanism"
result = client.models.embed_content(model = "models/gemini-embedding-001",contents = sentence) #it stores the number of fields of vector of sentence 
vector =result.embeddings[0].values #result stores the vector of the sentence in list form and [0] is used to access the first element of the 

print(f"Sentence :{sentence}")
print(f"vector:{len(vector)} no.")
print(f"first 5values : {vector[:5]}")


# For multiple sentences

sentences =[
    "Transformers use self-attention to process sequences",
    "BERT is a bidirectional transformer model",
    "GPT uses autoregressive decoding for generation",
    "LSTM uses gates to control information flow",
    "Random forests combine multiple decision trees",
    "Cricket is played between two teams of eleven",
]
embeddings = []
for s in sentences:
    result = client.models.embed_content(model = "models/gemini-embedding-001",contents= s)
    embeddings.append(result.embeddings[0].values)

print(f"Multiple sentences:{sentences}")
print(f"Total sentences embedded:{len(embeddings)}")
print(f"each vector lenght:{len(embeddings[0])}")
for i ,s in enumerate (sentences):
    print(f"[{i}]{s[:50]}")


# Cosine similarity 

def cosine_similarity(vec1,vec2):
    dot = np.dot(vec1,vec2) 
    mag1 = np.linalg.norm(vec1) # magnitude of vector 1
    mag2 = np.linalg.norm(vec2) # magnitude of vector 2
    return dot / (mag1*mag2)

print(f"cosine similarity btw two sentence vectors")
print(f"sentence 1 :'{sentences[0][:40]}...'")
print(f"sentence 2 :'{sentences[1][:40]}...'")
print(f"similarity:{cosine_similarity(embeddings[0],embeddings[1])}")


print(f"sentence 1:'{sentences[0][:40]}...'")
print(f"sentence 4:'{sentences[3][:40]}...'")   
print(f"similarity:{cosine_similarity(embeddings[0],embeddings[3])}")   


print(f"sentence 1:'{sentences[0][:40]}...'")
print(f"sentence 6:'{sentences[5][:40]}...'")   
print(f"similarity:{cosine_similarity(embeddings[0],embeddings[5])}")


print(f"all the pairwise cosine similarities with sentence 1:")
for i ,s in enumerate(sentences):
    score = cosine_similarity(embeddings[0],embeddings[i])
    bar = "#" * int (score*20)
    print(f"[{i}] {score:.4f}{bar}{s[:35]}")
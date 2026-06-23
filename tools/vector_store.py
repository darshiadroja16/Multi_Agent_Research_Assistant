from google import genai
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()
client_genai = genai.Client(api_key = os.getenv("GEMINI_API_KEY")) #read the evn file and get the api key from it and store it in the main memory for future use (login is done)


#Chroma db stores documents ,their embeddings and their unique ids in the local machine storage.
# chroma db create

chroma_client = chromadb.PersistentClient(path="./chroma_db") #it is used to store data in the local mmachine storage by creating folder name chroma_db in the hard drive which means if the machines shutsdown it will not let data get deleted 
collection = chroma_client.get_or_create_collection(name="research_papers",metadata={"hnsw:space":"cosine"}) # to create new collection in the chromadb collection or to open the existing collection  

# helper functions

def get_embedding(text):
    """ Ek text ka embedding lo gemini se """
    result = client_genai.models.embed_content(model ="gemini-embedding-001",contents = text)
    return result.embeddings[0].values

def add_documents(documents,ids):
    """documents  ko embedding me convert krke chromadb me store krta hai"""
    print(f"storing {len(documents)} documents...") #how many documents are going to get stored in the chromadb collection
    embeddings=[]
    for i,doc in enumerate(documents): #to store embeddings of single single document in the list and then store it in the chromadb collection
        emd = get_embedding(doc) #embedding of current document is stored in the emd variable
        embeddings.append(emd)
        print(f"embedded [{i+1}/{len(documents)}]: {doc[:45]}...")

    collection.upsert(documents=documents,embeddings=embeddings,ids=ids) #upsert is used to update or insert the data in the collection if the data is already present it will update it otherwise it will insert the new data in the collection 
    print(f"stored! total in db {collection.count()}") 

def search_documents(query,n_results=3):
    """query se similar documents dhundo"""
    print(f"Query: '{query}'")

    query_embedding= get_embedding(query) #user query embeddings 
    results=collection.query(query_embeddings=[query_embedding],n_results=n_results)

    docs=results['documents'][0] # top matching documents 
    distances=results['distances'][0] # top matching documents's distances from the query embedding

    for i,(doc,dist) in enumerate(zip(docs,distances)): #to print result documents and their similarity with the query
        similarity=1-dist  #convert distance to similarity 
        bar="."*int(similarity*20) # for visualization of similarity
        print(f"rank {i+1}")
        print(f"similarity:{similarity:.4f}{bar}")
        print(f"document : {doc}")

    return results

documents=[
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

search_documents("transformer architecture and attention")
search_documents("sequential data and memory")
search_documents("sports and outdoor games")
search_documents("image recognition deep learning")
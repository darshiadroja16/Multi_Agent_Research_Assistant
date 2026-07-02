# ------------------------- This part of file is for learning -----------------------------------------

# from google import genai
# from dotenv import load_dotenv
# import chromadb
# import os


# # Chunking Logic

# def chunk_text(text,chunk_size=400,overlap=50):
#     """
#     Text ko meaningful chunks mein todna.

#     chunk_size : characters per chunk
#     overlap    : kitne chars dono chunks mein common honge
#                  (context loss avoid karne ke liye)
#     """
#     chunks =[]
#     start =0

#     while start < len(text):
#         end = start + chunk_size

#         # for last chunk
#         if end >= len(text):
#             chunk = text[start:].strip()
#             if chunk:
#                 chunks.append(chunk)
#             break
        
#         # try to find a natural stopping point (.) instead of cutting in the middle
#         # search backwards from 'end' to 'start' for the last period (.)
#         boundary = text.rfind('.', start, end)

#         if boundary != -1 and boundary > start + (chunk_size // 2):
#             # period was found and it is after the halfway point of the chunk.
#             # cut the text at the period so the sentence remains complete.
#             chunk = text[start:boundary + 1].strip()
#             # next chunk will start after this period.
#             next_start = boundary + 1
#         else:
#         # no suitable period found.
#         # cut the chunk at the fixed chunk size.
#             chunk = text[start:end].strip()
#             # Next chunk starts from the fixed-size end position.
#             next_start = end

#         # store the chunk only if it is not empty.
#         if chunk:
#             chunks.append(chunk)

#         # move the starting position for the next chunk.
#         # subtract overlap so that some text is repeated in the next chunk.
#         # this preserves context between consecutive chunks.
#         start = next_start - overlap

#     return chunks



# # Embedding and Vector store



# load_dotenv()
# client_genai = genai.Client(api_key = os.getenv("GEMINI_API_KEY")) #read the evn file and get the api key from it and store it in the main memory for future use (login is done)


# #Chroma db stores documents ,their embeddings and their unique ids in the local machine storage.
# # chroma db create

# chroma_client = chromadb.PersistentClient(path="./chroma_db") #it is used to store data in the local mmachine storage by creating folder name chroma_db in the hard drive which means if the machines shutsdown it will not let data get deleted 
# collection = chroma_client.get_or_create_collection(name="research_papers",metadata={"hnsw:space":"cosine"}) # to create new collection in the chromadb collection or to open the existing collection  

# # helper functions

# def get_embedding(text):
#     """ Ek text ka embedding lo gemini se """
#     result = client_genai.models.embed_content(model ="gemini-embedding-001",contents = text)
#     return result.embeddings[0].values

# def add_documents(documents,ids):
#     """documents  ko embedding me convert krke chromadb me store krta hai"""
#     print(f"storing {len(documents)} documents...") #how many documents are going to get stored in the chromadb collection
#     embeddings=[]
#     for i,doc in enumerate(documents): #to store embeddings of single single document in the list and then store it in the chromadb collection
#         emd = get_embedding(doc) #embedding of current document is stored in the emd variable
#         embeddings.append(emd)
#         print(f"embedded [{i+1}/{len(documents)}]: {doc[:45]}...")

#     collection.upsert(documents=documents,embeddings=embeddings,ids=ids) #upsert is used to update or insert the data in the collection if the data is already present it will update it otherwise it will insert the new data in the collection 
#     print(f"stored! total in db {collection.count()}") 

# def search_documents(query,n_results=3):
#     """query se similar documents dhundo"""
#     print(f"Query: '{query}'")

#     query_embedding= get_embedding(query) #user query embeddings 
#     results=collection.query(query_embeddings=[query_embedding],n_results=n_results)

#     docs=results['documents'][0] # top matching documents 
#     distances=results['distances'][0] # top matching documents's distances from the query embedding

#     for i,(doc,dist) in enumerate(zip(docs,distances)): #to print result documents and their similarity with the query
#         similarity=1-dist  #convert distance to similarity 
#         bar="."*int(similarity*20) # for visualization of similarity
#         print(f"rank {i+1}")
#         print(f"similarity:{similarity:.4f}{bar}")
#         print(f"document : {doc}")

#     return results

# documents=[
#     "Transformers use self-attention to process sequences in parallel",
#     "BERT is a bidirectional transformer pre-trained on masked language modeling",
#     "GPT uses autoregressive transformer decoder for text generation",
#     "LSTM uses forget gate, input gate and output gate for sequential processing",
#     "Random forests build multiple decision trees and merge their predictions",
#     "Convolutional neural networks use filters to detect spatial features in images",
#     "Attention mechanism allows model to focus on relevant parts of the input sequence",
#     "Word2Vec learns word embeddings by predicting surrounding words in context",
#     "Gradient descent minimizes loss function by updating weights iteratively",
#     "Cricket is a bat and ball sport played between two teams of eleven players",
#     "Stock market prices fluctuate based on supply demand and investor sentiment",
# ]

# ids = [f"doc_{i}" for i in range(len(documents))]

# add_documents(documents, ids)

# search_documents("transformer architecture and attention")
# search_documents("sequential data and memory")
# search_documents("sports and outdoor games")
# search_documents("image recognition deep learning")


# # Chunking experiment

# if __name__ =="__main__":
    
#     sample = """
#     The Transformer architecture introduced self-attention mechanisms
#     that allow models to process sequences in parallel. Unlike RNNs,
#     transformers have no recurrence which makes training much faster.
#     Multi-head attention allows the model to jointly attend to information
#     from different representation subspaces at different positions.
#     The encoder consists of 6 identical layers. Each layer has two
#     sub-layers: multi-head attention and feed-forward network.
#     Residual connections and layer normalization are applied after each.
#     The decoder also has 6 layers but adds a third sub-layer for
#     cross-attention over the encoder output. Training used Adam optimizer.
#     """

#     print("CHUNKING EXPERIMENT")

#     for size in [150, 300, 400]:
#         chunks = chunk_text(sample, chunk_size=size, overlap=50)
#         print(f"Chunk size {size} → {len(chunks)} chunks:")
#         for i, c in enumerate(chunks):
#             print(f"  [{i+1}] ({len(c)} chars): {c[:65]}...")



#----------------------------- Production File --------------------------------------------


import chromadb
from dotenv import load_dotenv
from embedder import get_embedding, get_embeddings_batch
import os

load_dotenv()

# Persistent client — data disk pe save hoga
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# chromaDB collection exist then use that or create new 
def get_collection(name: str = "research_papers"):

    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


# chunk the text (papers) into meaningful chunks
def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> list[str]:
    """
    Text ko meaningful chunks mein todo.

    Args:
        text       : input text
        chunk_size : characters per chunk
        overlap    : overlap between consecutive chunks

    Returns:
        list of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # natural boundary—period dhundo
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

# to store papers in chromadb
# ArXiv does NOT return one big document containing multiple papers. It returns a list of paper objects.
# paper objects are :
# {
#     "title": "Attention Is All You Need",
#     "abstract": "...",
#     "url": "..."
# }
def add_papers_to_store(
    papers: list[dict],
    collection_name: str = "research_papers"
) -> int:
    """
    ArXiv papers ko ChromaDB mein store karo.

    Args:
        papers: list of dicts with keys:
                title, abstract, url, authors
        collection_name: ChromaDB collection naam

    Returns:
        total chunks stored
    """
    collection = get_collection(collection_name)

    all_chunks = []
    all_ids = []
    all_metadatas = []
    chunk_counter = collection.count() #existing count se start

    print(f"\nIndexing {len(papers)} papers...")

    for paper in papers:
        # title + abstract combine 
        full_text = f"Title: {paper['title']}\n\n{paper['abstract']}"

        # chunk karo
        chunks = chunk_text(full_text)

        for chunk in chunks:
            chunk_id = f"chunk_{chunk_counter}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "title" : paper.get('title', ''),
                "url"   : paper.get('url', ''),
                "source": "arxiv"
            })
            chunk_counter += 1

    print(f"Total chunks to embed: {len(all_chunks)}")

    # embeddings generate karo
    embeddings = get_embeddings_batch(all_chunks)

    # chromaDB mein store karo
    # upsert = add if not exists,update if exists
    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"Stored! Total in DB:{collection.count()}")
    return len(all_chunks)


def search_similar_chunks(
    query: str,
    n_results: int = 4,
    collection_name: str = "research_papers"
) -> list[dict]:
    """
    Query ke liye most relevant chunks fetch karo.

    Args:
        query           : search query string
        n_results       : kitne chunks chahiye
        collection_name : ChromaDB collection naam

    Returns:
        list of dicts with keys: content, metadata, similarity
    """
    collection = get_collection(collection_name)

    # empty collection check
    if collection.count() == 0:
        print("Warning: Collection is empty!")
        return []

    # query embed karo
    query_embedding = get_embedding(query)

    # search karo
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count())
    )

    # clean format mein return karo
    chunks = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        chunks.append({
            "content"   : doc,
            "metadata"  : meta,
            "similarity": round(1 - dist, 4)
        })

    return chunks


def clear_collection(collection_name: str = "research_papers"):
    """Collection delete karo — fresh start ke liye"""
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' cleared!")
    except Exception as e:
        print(f"Collection not found: {e}")


if __name__ == "__main__":

    print("Testing vector_store.py...")

    # test papers
    test_papers = [
        {
            "title"   : "Attention is All You Need",
            "abstract": """The dominant sequence transduction models are based
            on complex recurrent or convolutional neural networks. We propose
            the Transformer, based solely on attention mechanisms. The model
            achieves 28.4 BLEU on WMT 2014 English-to-German translation.""",
            "url"     : "https://arxiv.org/abs/1706.03762",
            "authors" : ["Vaswani et al."]
        },
        {
            "title"   : "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": """We introduce BERT, designed to pre-train deep
            bidirectional representations by jointly conditioning on both
            left and right context. BERT achieved 80.5 on GLUE benchmark
            and 93.2 F1 on SQuAD 1.1.""",
            "url"     : "https://arxiv.org/abs/1810.04805",
            "authors" : ["Devlin et al."]
        },
    ]

    # fresh start
    clear_collection("test_collection")

    # store karo
    add_papers_to_store(test_papers, "test_collection")

    # search karo
    print("\nSearching: 'attention mechanism transformer'")
    results = search_similar_chunks(
        "attention mechanism transformer",
        n_results=3,
        collection_name="test_collection"
    )

    for i, r in enumerate(results):
        print(f"\nRank {i+1} | Similarity: {r['similarity']}")
        print(f"Title : {r['metadata']['title']}")
        print(f"Text  : {r['content'][:80]}...")

    print("\nvector_store.py working correctly!")
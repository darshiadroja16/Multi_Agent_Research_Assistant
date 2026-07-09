import chromadb
from dotenv import load_dotenv
try:
    from tools.embedder import get_embedding, get_embeddings_batch
except ModuleNotFoundError:
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
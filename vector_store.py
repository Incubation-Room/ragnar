import os
import logging

import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings

import numpy as np

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_vector_store(vector_store, directory_path="faiss_index"):
    """
    Saves the FAISS vector store and associated document store to a directory.

    Parameters:
    - vector_store (FAISS): The FAISS vector store to save.
    - directory_path (str): Path to the directory where the store will be saved.
    """
    os.makedirs(directory_path, exist_ok=True)
    vector_store.save_local(directory_path)
    logger.info(f"Vector store saved to {directory_path}")

def load_vector_store(directory_path="faiss_index", embedding_function=None):
    """
    Loads a FAISS vector store and associated document store from a directory.

    Parameters:
    - directory_path (str): Path to the directory containing the saved vector store.
    - embedding_function (HuggingFaceEmbeddings, optional): Embedding function to use with the vector store.

    Returns:
    - FAISS: The loaded vector store.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"No vector store found at {directory_path}")
    
    if embedding_function is None:
        raise ValueError("An embedding function must be provided to load the vector store.")
    
    # Correctly pass the embedding function as the 'embeddings' argument
    vector_store = FAISS.load_local(directory_path, embeddings=embedding_function, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from {directory_path}")
    return vector_store



def create_vector_store(chunks, model_name="all-MiniLM-L6-v2", save_path=".vector_store"):
    """
    Creates or loads a FAISS vector store using HuggingFaceEmbeddings.

    Parameters:
    - chunks (List[Document]): List of document chunks to embed.
    - model_name (str): Sentence embedding model to use.
    - save_path (str, optional): Path to save the vector store (if created).

    Returns:
    - FAISS: A vector store ready for use.
    """
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    # If the vector store exists, load it
    if save_path and os.path.exists(save_path):
        return load_vector_store(save_path, embedding_function=embedding_function)

    # Otherwise, create the vector store
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not valid_chunks:
        raise ValueError("No valid documents found after filtering.")

    texts = [chunk.page_content for chunk in valid_chunks]
    embeddings = np.array(embedding_function.embed_documents(texts))
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(valid_chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(valid_chunks))}

    vector_store = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_function,
    )

    # Save the vector store if a save path is provided
    if save_path:
        save_vector_store(vector_store, save_path)

    return vector_store

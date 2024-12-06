import os
import logging
import json


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


def vector_store_exists(directory_path="faiss_index"):
    """
    Vérifie si un vector store existe déjà dans le chemin donné.
    
    Args:
        directory_path (str): Le chemin du vector store.
    
    Returns:
        bool: True si le vector store existe, sinon False.
    """
    return os.path.exists(directory_path) and os.path.isdir(directory_path)



def save_vector_store(vector_store, model_name, directory_path="faiss_index"):
    """
    Saves the FAISS vector store and associated document store to a directory,
    along with metadata like the model name.

    Parameters:
    - vector_store (FAISS): The FAISS vector store to save.
    - model_name (str): The name of the embedding model used.
    - directory_path (str): Path to the directory where the store will be saved.
    """
    os.makedirs(directory_path, exist_ok=True)
    vector_store.save_local(directory_path)

    # Save metadata
    metadata_path = os.path.join(directory_path, "metadata.json")
    metadata = {"model_name": model_name}
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)
    
    logger.info(f"Vector store and metadata saved to {directory_path}")


def load_vector_store(directory_path="faiss_index"):
    """
    Loads a FAISS vector store and associated document store from a directory,
    ensuring the correct model is used based on saved metadata.

    Parameters:
    - directory_path (str): Path to the directory containing the saved vector store.

    Returns:
    - FAISS: The loaded vector store.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"No vector store found at {directory_path}")

    # Load metadata
    metadata_path = os.path.join(directory_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata file found at {metadata_path}")
    
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("Model name not found in metadata.")

    print(f"Using model '{model_name}' to load vector store...")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    # Load the vector store
    vector_store = FAISS.load_local(directory_path, embeddings=embedding_function, allow_dangerous_deserialization=True)
    logger.info(f"Vector store loaded from {directory_path} with model '{model_name}'")
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
        return load_vector_store(directory_path=save_path)

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
        save_vector_store(vector_store, model_name, save_path)

    return vector_store

from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_chunkers import StatisticalChunker
from langchain_huggingface import HuggingFaceEmbeddings
from semantic_router.encoders import HuggingFaceEncoder

from langchain.schema import Document

def split_documents(documents, chunk_size=500, chunk_overlap=50, semantic_chunking=True):
    """
    Divise les documents en segments (chunks) pour une analyse plus fine.

    Parameters:
    - documents (List[Document]): Liste d'objets Document à diviser.
    - chunk_size (int): Taille maximale de chaque chunk (en caractères).
    - chunk_overlap (int): Nombre de caractères de chevauchement entre les chunks.

    Returns:
    - List[Document]: Liste de nouveaux objets Document segmentés.
    """
    # Initialisation du text splitter avec les paramètres fournis
    if semantic_chunking:
        # Initialize the semantic chunker
        encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = StatisticalChunker(encoder=encoder)

        # Extract text from Document objects
        texts = [doc.page_content for doc in documents]

        # Perform semantic chunking
        chunked_texts = text_splitter(texts)  # List of lists of Chunks

        # Convert Chunk objects to Document objects
        chunked_documents = []
        for doc, chunks in zip(documents, chunked_texts):
            for chunk in chunks:
                if hasattr(chunk, "splits") and isinstance(chunk.splits, list):
                    chunked_documents.append(
                        Document(
                            page_content="".join(chunk.splits),
                            metadata=doc.metadata
                        )
                    )
        return chunked_documents
    else:
        # Use default character-based splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

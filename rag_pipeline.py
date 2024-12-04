from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import numpy as np
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import subprocess
import requests
import json
import os
import io


# Fonction pour interroger Ollama

def ollama_query(prompt, model="llama3.2", api_url="http://localhost:11434/api/generate"):
    """
    Interroge Ollama via une API REST locale.
    
    Parameters:
    - prompt: La question ou la commande à exécuter.
    - model: Le modèle Ollama à utiliser (par défaut : "llama3.2").
    - api_url: L'URL de l'API Ollama (par défaut : "http://localhost:11434/api/generate").
    
    Returns:
    - La réponse générée par Ollama.
    """
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt
    }
    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()  # Lève une exception si le statut HTTP est une erreur
        text = response.text.strip()
        lines = text.split("\n")
        tokens = list(map(lambda line: json.loads(line)["response"], lines))
        formated = "".join(tokens)
        answer = formated.strip()
        return answer
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erreur lors de la requête à Ollama : {e}")


class Document:
    def __init__(self, page_content, metadata):
        """
        Représente un document avec son contenu et ses métadonnées.

        Parameters:
        - page_content (str): Contenu textuel du document.
        - metadata (dict): Métadonnées associées au document.
        """
        self.page_content = page_content
        self.metadata = metadata


def load_documents(source, is_directory=False):
    """
    Charge les documents PDF depuis un dossier ou un fichier unique.

    Parameters:
    - source (str | List[UploadedFile]): Chemin du dossier, chemin d'un fichier unique, ou liste de fichiers uploadés.
    - is_directory (bool): Indique si `source` est un dossier.

    Returns:
    - List[Document]: Liste d'objets Document contenant le texte extrait et les métadonnées.
    """
    # Extensions supportées et leurs extracteurs associés
    supported_extensions = {
        "pdf": extract_content_from_pdf,
    }

    documents = []

    if is_directory:
        # Charger depuis un dossier
        for root, _, files in os.walk(source):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_extension = file_name.split(".")[-1].lower()

                if file_extension in supported_extensions:
                    try:
                        extractor = supported_extensions[file_extension]
                        content = extractor(file_path)
                        documents.append(Document(page_content=content["text"], metadata={"source": file_path}))
                    except Exception as e:
                        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
                else:
                    print(f"Type de fichier non pris en charge : {file_path}")
    else:
        # Vérifiez si source est une chaîne (fichier unique)
        if isinstance(source, str):
            file_path = source
            file_extension = file_path.split(".")[-1].lower()

            if file_extension in supported_extensions:
                try:
                    extractor = supported_extensions[file_extension]
                    content = extractor(file_path)
                    documents.append(Document(page_content=content["text"], metadata={"source": file_path}))
                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {file_path}: {e}")
            else:
                print(f"Type de fichier non pris en charge : {file_path}")

        # Si c'est une liste (fichiers uploadés)
        elif isinstance(source, list):
            for uploaded_file in source:
                file_name = uploaded_file.name.lower()
                file_extension = file_name.split(".")[-1]

                if file_extension in supported_extensions:
                    try:
                        extractor = supported_extensions[file_extension]
                        content = extractor(uploaded_file)
                        documents.append(Document(page_content=content["text"], metadata={"source": file_name}))
                    except Exception as e:
                        print(f"Erreur lors du traitement du fichier {file_name}: {e}")
                else:
                    print(f"Type de fichier non pris en charge : {file_name}")

    return documents

def extract_content_from_pdf(file_path):
    """
    Extrait le contenu d'un fichier PDF, incluant :
    - Texte extrait des pages PDF.
    - Texte extrait des images dans le PDF via OCR.
    - Informations des métadonnées, si disponibles.

    Parameters:
    - file_path (str): Chemin vers le fichier PDF.

    Returns:
    - dict: Contient le texte, les métadonnées, et les erreurs rencontrées.
    """
    content = {"text": "", "ocr_text": "", "metadata": {}, "errors": []}

    try:
        # Charger le PDF avec PyMuPDF (fitz)
        doc = fitz.open(file_path)

        # Extraire le texte des pages
        text_content = ""
        for page in doc:
            text_content += page.get_text()

        content["text"] = text_content.strip()

        # Extraire les images et appliquer l'OCR
        image_texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image)
                    image_texts.append(ocr_text)
                except Exception as e:
                    error_msg = f"Erreur lors de l'extraction d'une image à la page {page_num + 1}: {e}"
                    content["errors"].append(error_msg)

        content["ocr_text"] = "\n".join(image_texts)

        # Extraire les métadonnées
        content["metadata"] = doc.metadata or {}

    except Exception as e:
        content["errors"].append(f"Erreur lors de l'extraction du PDF : {e}")

    return content


def split_documents(documents, chunk_size=500, chunk_overlap=50):
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    try:
        return text_splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la division des documents : {e}")


def create_vector_store(chunks):
    """
    Crée une base vectorielle FAISS en utilisant HuggingFaceEmbeddings.

    Parameters:
    - chunks (List[Document]): Liste des segments de documents, chaque segment ayant un attribut `page_content`.

    Returns:
    - FAISS: Une base vectorielle prête à l'emploi pour la récupération d'information.
    """
    model_name = "all-MiniLM-L6-v2"  # Modèle compact et rapide

    try:
        # Créer une fonction d'embedding compatible
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # Filtrer les documents valides (non vides)
        valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not valid_chunks:
            raise ValueError("Aucun document valide trouvé après filtrage.")

        # Extraire le contenu des segments pour les embeddings
        texts = [chunk.page_content for chunk in valid_chunks]

        # Générer les embeddings et les convertir en tableau NumPy
        embeddings = np.array(embedding_function.embed_documents(texts))

        # Vérifier que le nombre d'embeddings correspond au nombre de documents valides
        if len(embeddings) != len(valid_chunks):
            raise ValueError("Le nombre d'embeddings ne correspond pas au nombre de documents valides.")

        print(f"Nombre de documents initiaux : {len(chunks)}")
        print(f"Nombre de documents valides : {len(valid_chunks)}")

        # Créer un index FAISS
        dimension = embeddings.shape[1]  # Déduire la dimension des embeddings
        index = faiss.IndexFlatL2(dimension)  # Index avec la distance L2
        index.add(embeddings)  # Ajouter les embeddings à l'index

        # Associer l'index à un docstore en mémoire
        docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(valid_chunks)})
        index_to_docstore_id = {i: str(i) for i in range(len(valid_chunks))}

        # Afficher le mapping après l'initialisation
        print(f"Mapping index_to_docstore_id : {index_to_docstore_id}")

        # Vérifiez la synchronisation entre le docstore et FAISS
        for i in range(len(valid_chunks)):
            if index_to_docstore_id[i] not in docstore._dict:
                raise ValueError(f"Problème de synchronisation : ID {i} non trouvé dans le docstore.")

        # Construire la base vectorielle FAISS
        vector_store = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=embedding_function,
        )
        print(f"Documents dans le docstore : {len(docstore._dict)}")
        
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création de la base vectorielle FAISS : {e}")
      
def create_retrieval_qa_chain(vector_store):

    """
    Crée une chaîne de récupération et de génération de réponses en utilisant un store vectoriel FAISS.

    Parameters:
    - vector_store (FAISS): La base vectorielle utilisée pour la récupération des documents pertinents.

    Returns:
    - tuple: 
        - retriever (Callable): Fonction pour récupérer les documents les plus pertinents.
        - generate_answer (Callable): Fonction pour générer une réponse basée sur une question et un contexte.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def generate_answer(query, context):
        """
        Génère une réponse en interrogeant Ollama avec un prompt contenant le contexte et la question.

        Parameters:
        - query (str): La question posée par l'utilisateur.
        - context (str): Le contexte fourni par les documents récupérés.

        Returns:
        - str: La réponse générée.
        """
        prompt = f"""
        Context: {context}
        Question: {query}
        Answer:
        """
        return ollama_query(prompt)

    return retriever, generate_answer

# pdf_path = "/Users/sebastienstagno/ICAM/Machine Learning/Projet/Data/Base_Learning/TD/TD Regression.pdf"
# documents = load_documents(pdf_path, is_directory=False)

# chunks = split_documents(documents)
# vector_store = create_vector_store(chunks)

# retriever, generate_answer = create_retrieval_qa_chain(vector_store)

# query = "Quels sont les sujets du TD ?"

# try:
#     print("Lancement de la recherche dans FAISS...")
#     context_docs = retriever.invoke(query)

#     if not context_docs:
#         print("Aucun document pertinent trouvé.")
#     else:
#         print(f"Documents récupérés : {len(context_docs)}")
#         context = "\n".join([doc.page_content for doc in context_docs])
#         print(f"Contexte récupéré : {context[:200]}...")
#         answer = generate_answer(query, context)
#         print(f"Réponse générée : {answer}")
# except ValueError as e:
#     print(f"Erreur lors de la recherche : {e}")
# except RuntimeError as e:
#     print(f"Erreur système : {e}")

# retriever, generate_answer = create_retrieval_qa_chain(vector_store)
# query = "Quels sont les sujets du TD ?"
# context_docs = retriever.get_relevant_documents(query)
# context = "\n".join([doc.page_content for doc in context_docs])
# answer = generate_answer(query, context)
# print(answer)
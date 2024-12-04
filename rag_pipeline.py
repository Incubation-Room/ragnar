from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
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
    Charge les documents PDF depuis un dossier ou des fichiers uploadés (drag-and-drop).

    Parameters:
    - source (str | List[UploadedFile]): Chemin du dossier ou liste de fichiers uploadés.
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
        # Charger depuis des fichiers uploadés (drag-and-drop)
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
    Crée une base vectorielle FAISS en utilisant SentenceTransformers.

    Parameters:
    - chunks (List[Document]): Liste des segments de documents, chaque segment ayant un attribut `page_content`.

    Returns:
    - FAISS: Une base vectorielle prête à l'emploi pour la récupération d'information.
    """
    model_name = "all-MiniLM-L6-v2"  # Modèle compact et rapide

    try:
        # Charger le modèle SentenceTransformer
        embedding_model = SentenceTransformer(model_name)

        # Extraire le contenu des segments pour les embeddings
        texts = [chunk.page_content for chunk in chunks]

        # Générer les embeddings
        embeddings = embedding_model.encode(texts)

        # Créer une base vectorielle FAISS
        vector_store = FAISS(embeddings, chunks)

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
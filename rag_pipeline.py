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

# Définir le contexte initial
DEFAULT_CONTEXT = """
Ce système est conçu pour aider une association à gérer ses activités. 
Il fournit des réponses pertinentes basées sur les documents fournis. Veuillez vous assurer que vos réponses sont concises, utiles et alignées avec cet objectif.
"""

# Permettre la personnalisation du contexte
def get_initial_prompt(user_context=None):
    """
    Génère le prompt initial pour définir le contexte d'utilisation.
    :param user_context: Contexte personnalisé, ou utilise le contexte par défaut.
    :return: Prompt formaté pour inclure le contexte.
    """
    return user_context if user_context else DEFAULT_CONTEXT

def normalize_path(path):
    """
    Normalise un chemin pour s'assurer qu'il est compatible avec les systèmes de fichiers.
    """
    if path.startswith(("'", '"')) and path.endswith(("'", '"')):
        path = path[1:-1]  # Supprime les guillemets simples ou doubles entourant le chemin
    return os.path.abspath(path)  # Renvoie le chemin absolu normalisé

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
                        documents.append(
                            Document(
                                page_content=content["text"],
                                metadata={
                                    "source": file_path,
                                    "title": content["metadata"].get("title", "Titre non défini"),
                                    "date": content["metadata"].get("date", "Date non définie"),
                                },
                            )
                        )
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
                    documents.append(
                        Document(
                            page_content=content["text"],
                            metadata={
                                "source": file_path,
                                "title": content["metadata"].get("title", "Titre non défini"),
                                "date": content["metadata"].get("date", "Date non définie"),
                            },
                        )
                    )
                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {file_path}: {e}")
            else:
                print(f"Type de fichier non pris en charge : {file_path}")

    return documents

def extract_content_from_pdf(file_path):
    """
    Extrait le contenu d'un fichier PDF, incluant :
    - Texte extrait des pages PDF.
    - Texte extrait des images dans le PDF via OCR.
    - Informations des métadonnées, y compris le titre et la date.

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

        # Extraire les métadonnées générales
        metadata = doc.metadata or {}
        content["metadata"] = metadata

        # Tenter d'extraire le titre et la date depuis les métadonnées
        title = metadata.get("title", None)
        creation_date = metadata.get("creationDate", None)

        # Si le titre n'est pas dans les métadonnées, essayer de l'inférer depuis le texte
        if not title and text_content:
            first_line = text_content.split("\n")[0].strip()
            if len(first_line) > 5:  # Longueur minimale pour éviter des titres peu informatifs
                title = first_line

        # Formater la date si elle est présente dans les métadonnées
        if creation_date:
            creation_date = fitz.Document.convert_date(creation_date)  # Conversion automatique de date
            creation_date = creation_date.strftime("%Y-%m-%d")  # Format ISO8601

        # Ajout des valeurs calculées aux métadonnées
        content["metadata"]["title"] = title or "Titre non défini"
        content["metadata"]["date"] = creation_date or "Date non définie"

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


def create_vector_store(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Crée une base vectorielle FAISS en utilisant HuggingFaceEmbeddings.

    Parameters:
    - chunks (List[Document]): Liste des segments de documents, chaque segment ayant un attribut `page_content`.
    - model_name (str): Modèle de sentence embeddings à utiliser (par défaut : "all-MiniLM-L6-v2").

    Returns:
    - FAISS: Une base vectorielle prête à l'emploi pour la récupération d'information.
    """
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

        # Créer un index FAISS
        dimension = embeddings.shape[1]  # Déduire la dimension des embeddings
        index = faiss.IndexFlatL2(dimension)  # Index avec la distance L2
        index.add(embeddings)  # Ajouter les embeddings à l'index

        # Associer l'index à un docstore en mémoire
        docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(valid_chunks)})
        index_to_docstore_id = {i: str(i) for i in range(len(valid_chunks))}

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
        
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création de la base vectorielle FAISS : {e}")

def determine_optimal_k(documents, question, max_k=20, min_k=3):
    """
    Détermine dynamiquement le nombre optimal de chunks (k) à prendre en compte.

    Parameters:
    - documents (List[Document]): Liste des documents ou chunks disponibles.
    - question (str): La question posée par l'utilisateur.
    - max_k (int): Nombre maximal de chunks à prendre en compte.
    - min_k (int): Nombre minimal de chunks à prendre en compte.

    Returns:
    - int: Nombre optimal de chunks à utiliser.
    """
    # Gérer les cas où `documents` ou `question` est None
    if documents is None or len(documents) == 0:
        print("Aucun document fourni ou liste vide.")
        return min_k  # Retourne une valeur minimale par défaut

    if question is None or not isinstance(question, str) or len(question.strip()) == 0:
        print("Question non valide ou vide.")
        question_length = 0
    else:
        question_length = len(question.split())

    # Ajuster k en fonction de la taille des documents
    if len(documents) < min_k:
        return len(documents)
    elif len(documents) <= max_k:
        return len(documents)
    else:
        if question_length <= 10:  # Question courte et spécifique
            return min_k
        elif question_length <= 25:  # Question moyenne
            return min(min_k + 2, max_k)
        else:  # Question longue ou exploratoire
            return max_k
        
def build_context_from_docs(context_docs):
    """
    Builds a context string from the retrieved documents, including metadata.

    Args:
        context_docs (list): List of documents retrieved, each with `page_content` and `metadata` attributes.

    Returns:
        str: A formatted string combining document metadata and content.
    """
    return "\n".join(
        [
            f"Metadata: {doc.metadata}\nContent: {doc.page_content}" 
            for doc in context_docs
        ]
    )


def create_retrieval_qa_chain(vector_store, initial_context=None, search_type="similarity", k=None, question=None):
    """
    Crée une chaîne de récupération et de génération de réponses en utilisant un store vectoriel FAISS.
    Ajuste dynamiquement le nombre de chunks (k).

    Parameters:
    - vector_store (FAISS): La base vectorielle utilisée pour la récupération des documents pertinents.
    - initial_context (str, optional): Contexte initial pour guider les réponses générées.
    - search_type (str): Type de recherche utilisé (par défaut : "similarity").
    - k (int): Nombre de documents à récupérer (par défaut : 5).

    Returns:
    - tuple: 
        - retriever (Callable): Fonction pour récupérer les documents les plus pertinents.
        - generate_answer (Callable): Fonction pour générer une réponse basée sur une question et un contexte.
    """
    # Récupérer le contexte initial ou utiliser celui par défaut
    initial_context = get_initial_prompt(initial_context)

    # Déterminer dynamiquement le nombre de chunks si `k` n'est pas défini
    if k is None:
        if question:
            k = determine_optimal_k(vector_store.docstore._dict.values(), question)
        else:
            k = 5  # Valeur par défaut si aucune question n'est fournie

    # Assurez-vous que k est un entier positif
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Le paramètre 'k' doit être un entier positif. Valeur reçue : {k}")

    # Configurer le retriever avec les paramètres spécifiés
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    def generate_answer(query, context):
        """
        Génère une réponse en interrogeant Ollama avec un prompt contenant le contexte initial, le contexte des documents, et la question.

        Parameters:
        - query (str): La question posée par l'utilisateur.
        - context (str): Le contexte fourni par les documents récupérés.

        Returns:
        - str: La réponse générée.
        """
        prompt = f"""
        Voici les fichiers qui ont été retrouvés d'après la requête: {context}
        Utilise leur contenu pour répondre à cette question: {query}
        Answer:
        """
        try:
            return ollama_query(prompt)
        except RuntimeError as e:
            raise RuntimeError(f"Error generating answer: {e}")

    return retriever, generate_answer
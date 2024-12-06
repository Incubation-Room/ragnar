
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import numpy as np
from extract_pdf import extract_content_from_pdf
import os

from ollama_query import ollama_query # Fonction pour interroger Ollama
from chunking import split_documents

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

# class Document:
#     def __init__(self, page_content, metadata):
#         """
#         Représente un document avec son contenu et ses métadonnées.

#         Parameters:
#         - page_content (str): Contenu textuel du document.
#         - metadata (dict): Métadonnées associées au document.
#         """
#         self.page_content = page_content
#         self.metadata = metadata


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
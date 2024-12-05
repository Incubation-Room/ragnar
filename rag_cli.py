"""
Module pour charger, traiter et interroger des documents à l'aide d'un modèle d'apprentissage automatique.

Ce module permet à l'utilisateur de :
1. Choisir un modèle d'indexation multilingue pour les embeddings.
2. Charger des documents depuis un fichier ou un dossier.
3. Diviser les documents en chunks et créer une base vectorielle.
4. Interroger le système RAG pour obtenir des réponses basées sur les documents chargés.

Fonctions principales :
- print_model_options: Affiche les modèles disponibles.
- select_model: Permet à l'utilisateur de sélectionner un modèle pour les embeddings.
- get_source_path: Demande un chemin de fichier ou dossier, avec option de chemin par défaut.
- check_path_type: Vérifie si le chemin fourni est un fichier ou un dossier.
- handle_documents: Charge et divise les documents en chunks.
- create_vector_store_from_chunks: Crée la base vectorielle à partir des chunks.
- run_interactive_query: Permet à l'utilisateur de poser des questions et d'obtenir des réponses.
- main: Fonction principale qui coordonne le processus.
"""


import os
from pathlib import Path
from rag_pipeline import normalize_path, load_documents, split_documents, create_retrieval_qa_chain, create_vector_store

def print_model_options():
    """
    Affiche les options de modèles disponibles pour l'utilisateur.
    """
    print("Modèles disponibles :")
    print("1. all-MiniLM-L6-v2 (Rapide, multilingue)")
    print("2. paraphrase-multilingual-mpnet-base-v2 (Multilingue, haute précision)")
    print("3. sentence-transformers/LaBSE (Spécifique multilingue)")
    print("4. dangvantuan/sentence-camembert-large (Français, expérimental)")


def select_model():
    """
    Permet à l'utilisateur de sélectionner un modèle et retourne le nom du modèle sélectionné.
    
    Returns:
        str: Le nom du modèle sélectionné.
    """
    model_choice = input("Sélectionnez un modèle (1-4) : ").strip()
    model_mapping = {
        "1": "all-MiniLM-L6-v2",
        "2": "paraphrase-multilingual-mpnet-base-v2",
        "3": "sentence-transformers/LaBSE",
        "4": "dangvantuan/sentence-camembert-large"
    }
    return model_mapping.get(model_choice, "all-MiniLM-L6-v2")


def get_source_path():
    """
    Demande à l'utilisateur de fournir un chemin de fichier ou de dossier. Si aucun chemin n'est fourni,
    un chemin par défaut est utilisé.
    
    Returns:
        Path: Le chemin normalisé.
    """
    source_path = input("Entrez le chemin du fichier ou du dossier : ").strip()
    if not source_path:
        current_folder = Path(os.getcwd())  # Répertoire courant
        default_folder = current_folder / "dev_data" / "archive_Ca_MR"
        source_path = str(default_folder)

    return normalize_path(source_path)


def check_path_type(source_path):
    """
    Vérifie si le chemin fourni est un fichier ou un dossier.
    
    Args:
        source_path (str): Le chemin à vérifier.
    
    Returns:
        bool: True si c'est un dossier, False si c'est un fichier.
    
    Raises:
        ValueError: Si le chemin n'est ni un fichier ni un dossier valide.
    """
    if os.path.isfile(source_path):
        return False
    elif os.path.isdir(source_path):
        return True
    else:
        raise ValueError("Le chemin fourni n'est ni un fichier ni un dossier valide.")


def handle_documents(source_path, is_directory):
    """
    Charge les documents depuis le chemin donné, puis les divise en chunks.
    
    Args:
        source_path (str): Le chemin des fichiers ou dossiers à traiter.
        is_directory (bool): Indique si le chemin est un dossier ou non.
    
    Returns:
        list: La liste des chunks générés à partir des documents.
    """
    documents = load_documents(source_path, is_directory=is_directory)
    if not documents:
        raise ValueError("Aucun document valide chargé.")
    
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Aucun chunk valide généré à partir des documents.")
    
    return chunks


def create_vector_store_from_chunks(chunks):
    """
    Crée la base vectorielle à partir des chunks donnés.
    
    Args:
        chunks (list): La liste des chunks de documents.
    
    Returns:
        object: La base vectorielle FAISS.
    
    Raises:
        RuntimeError: Si la création de la base vectorielle échoue.
    """
    try:
        return create_vector_store(chunks)
    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de la création de la base vectorielle FAISS : {e}")


def run_interactive_query(retriever, generate_answer):
    """
    Permet à l'utilisateur de poser des questions de manière interactive
    et d'obtenir des réponses basées sur les documents récupérés.
    
    Args:
        retriever (object): L'objet de récupération de documents.
        generate_answer (function): La fonction pour générer une réponse.
    """
    print("\nLe système est prêt. Vous pouvez poser vos questions.")
    print("Tapez 'exit' pour mettre fin au test.\n")

    while True:
        query = input("Entrez votre question : ").strip()
        if query.lower() == "exit":
            print("Fin du test. Merci d'avoir utilisé le système.")
            break
        if not query:
            print("La question ne peut pas être vide. Veuillez réessayer.")
            continue

        try:
            print("\nLancement de la recherche dans FAISS...")
            context_docs = retriever.invoke(query)

            if not context_docs:
                print("Aucun document pertinent trouvé.")
                continue

            print(f"Documents récupérés : {len(context_docs)}")
            context = "\n".join([doc.page_content for doc in context_docs])
            print(f"Contexte récupéré : {context[:200]}...")  # Limité à 200 caractères pour l'affichage
            answer = generate_answer(query, context)
            print(f"Réponse générée : {answer}\n")

        except ValueError as e:
            print(f"Erreur lors de la recherche : {e}")
        except RuntimeError as e:
            print(f"Erreur système : {e}")


def main():
    """
    Fonction principale pour charger des documents, créer une base vectorielle,
    interroger le système RAG, et permettre de poser plusieurs questions.
    Permet de traiter un fichier unique ou un dossier.
    """
    print_model_options()

    model_name = select_model()
    print(f"Modèle sélectionné : {model_name}\n")
    
    source_path = get_source_path()

    try:
        is_directory = check_path_type(source_path)
    except ValueError as e:
        print(e)
        return

    try:
        chunks = handle_documents(source_path, is_directory)
    except ValueError as e:
        print(e)
        return

    try:
        vector_store = create_vector_store_from_chunks(chunks)
    except RuntimeError as e:
        print(e)
        return

    retriever, generate_answer = create_retrieval_qa_chain(vector_store)

    run_interactive_query(retriever, generate_answer)


if __name__ == '__main__':
    main()

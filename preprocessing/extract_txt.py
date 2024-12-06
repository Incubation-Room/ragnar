import os
import time


def extract_content_from_txt(file_path):
    """
    Extrait le contenu d'un fichier texte (.txt), incluant :
    - Texte brut.
    - Métadonnées (nom de fichier, date de modification).
    - Gestion des erreurs.

    Parameters:
    - file_path (str): Chemin vers le fichier texte.

    Returns:
    - dict: Contient le texte, les métadonnées, et les erreurs rencontrées.
    """
    content = {"text": "", "metadata": {}, "errors": []}

    try:
        # Charger le texte du fichier
        with open(file_path, "r", encoding="utf-8") as f:
            content["text"] = f.read().strip()

        # Extraire les métadonnées du fichier
        metadata = {}
        metadata["filename"] = os.path.basename(file_path)
        metadata["filepath"] = file_path
        metadata["size_bytes"] = os.path.getsize(file_path)
        metadata["last_modified"] = time.ctime(os.path.getmtime(file_path))
        metadata["creation_date"] = time.ctime(os.path.getctime(file_path))

        content["metadata"] = metadata

    except Exception as e:
        content["errors"].append(f"Erreur lors de l'extraction du fichier texte : {e}")

    return content

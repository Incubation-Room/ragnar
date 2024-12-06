
import os

from langchain.schema import Document

from .extract_pdf import extract_content_from_pdf
from .extract_txt import extract_content_from_txt



supported_extensions = {
    "pdf": extract_content_from_pdf,
    "txt": extract_content_from_txt,
    "json": extract_content_from_txt,
    "md": extract_content_from_txt,
    "log": extract_content_from_txt,  # Extraction depuis fichiers log (texte brut)
    "ini": extract_content_from_txt,  # Extraction depuis fichiers ini (texte brut)
    "yml": extract_content_from_txt,  # Extraction depuis YAML
    "yaml": extract_content_from_txt,  # Extraction depuis YAML
}


def load_documents(source, is_directory=False):
    """
    Charge les documents depuis un dossier ou un fichier unique.

    Parameters:
    - source (str | List[UploadedFile]): Chemin du dossier, chemin d'un fichier unique, ou liste de fichiers uploadés.
    - is_directory (bool): Indique si `source` est un dossier.

    Returns:
    - List[Document]: Liste d'objets Document contenant le texte extrait et les métadonnées.
    """
    # Extensions supportées et leurs extracteurs associés


    documents = []

    def process_file(file_path):
        """
        Traite un fichier donné : extrait le contenu et crée un objet Document.

        Parameters:
        - file_path (str): Chemin du fichier à traiter.

        Returns:
        - Document | None: L'objet Document créé, ou None en cas d'erreur ou si l'extension n'est pas supportée.
        """
        file_extension = file_path.split(".")[-1].lower()
        if file_extension in supported_extensions:
            try:
                extractor = supported_extensions[file_extension]
                content = extractor(file_path)
                return Document(
                    page_content=content["text"],
                    metadata={
                        "source": file_path,
                        "title": content["metadata"].get("title", "Titre non défini"),
                        "date": content["metadata"].get("date", "Date non définie"),
                    },
                )
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        else:
            print(f"Type de fichier non pris en charge : {file_path}")
        return None

    if is_directory:
        # Charger depuis un dossier
        for root, _, files in os.walk(source):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                document = process_file(file_path)
                if document:
                    documents.append(document)
    else:
        # Charger depuis un fichier unique
        if isinstance(source, str):
            document = process_file(source)
            if document:
                documents.append(document)

    return documents


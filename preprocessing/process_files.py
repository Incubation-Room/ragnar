
import os
import tempfile

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


def save_uploaded_file(uploaded_file):
    """
    Sauvegarde un fichier téléchargé dans un répertoire temporaire et retourne son chemin.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())  # Sauvegarde le contenu du fichier téléchargé
        tmp_file_path = tmp_file.name
    return tmp_file_path


def load_documents(source, is_directory=False):
    """
    Charge les documents depuis un dossier ou un fichier unique.

    Parameters:
    - source (str | List[UploadedFile]): Chemin du dossier, chemin d'un fichier unique, ou liste de fichiers uploadés.
    - is_directory (bool): Indique si `source` est un dossier.

    Returns:
    - List[Document]: Liste d'objets Document contenant le texte extrait et les métadonnées.
    """
    # Liste pour stocker les documents extraits
    documents = []

    def process_file(file_path_or_obj, is_uploaded_file=False):
        """
        Traite un fichier donné (local ou UploadedFile Streamlit) : extrait le contenu et crée un objet Document.

        Parameters:
        - file_path_or_obj (str | UploadedFile): Chemin du fichier à traiter ou un objet UploadedFile.
        - is_uploaded_file (bool): Indique si c'est un fichier Streamlit uploadé.

        Returns:
        - Document | None: L'objet Document créé, ou None en cas d'erreur ou si l'extension n'est pas supportée.
        """
        if is_uploaded_file:
            # Gérer les objets UploadedFile
            file_path = save_uploaded_file(file_path_or_obj)  # Sauvegarde le fichier et retourne le chemin temporaire
            file_extension = file_path_or_obj.name.split(".")[-1].lower()
            file_name = file_path_or_obj.name
        else:
            # Gérer les fichiers locaux
            file_path = file_path_or_obj  # Utilise directement le chemin du fichier local
            file_extension = file_path.split(".")[-1].lower()
            file_name = os.path.basename(file_path)  # Extraire le nom du fichier local

        # Obtenez l'extension du fichier

        if file_extension in supported_extensions:
            try:
                # Utiliser l'extracteur approprié pour le fichier
                extractor = supported_extensions[file_extension]
                content = extractor(file_path)  # Passe le chemin temporaire ou local
                return Document(
                    page_content=content["text"],
                    metadata={
                        "source": file_name,
                        "source_path": file_path,
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
                document = process_file(file_path, is_uploaded_file=False)
                if document:
                    documents.append(document)
    else:
        # Charger depuis une liste de fichiers téléchargés
        documents = [process_file(file_obj, is_uploaded_file=True) for file_obj in source]

    return documents


def main():
    # Test pour un dossier contenant des fichiers à extraire
    test_directory = "dev_data/archive_Ca_MR"  # Remplacez par le chemin de votre dossier
    if os.path.isdir(test_directory):
        print(f"Chargement des documents depuis le dossier : {test_directory}")
        documents = load_documents(test_directory, is_directory=True)
        print(f"{len(documents)} documents chargés depuis le dossier.")
        for doc in documents:
            print(f"Document: {doc.metadata['title']} - {doc.metadata['source']}")

if __name__ == "__main__":
    main()

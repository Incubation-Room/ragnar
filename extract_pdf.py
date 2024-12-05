import io
import logging

import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        creation_date = metadata.get("creationDate", None)

        # Formater la date si elle est présente dans les métadonnées
        if creation_date:
            creation_date = fitz.Document.convert_date(creation_date)  # Conversion automatique de date
            creation_date = creation_date.strftime("%Y-%m-%d")  # Format ISO8601

        # Ajout des valeurs calculées aux métadonnées
        content["metadata"]["title"] = extract_title(text_content, metadata)
        content["metadata"]["date"] = creation_date or "Date non définie"

        # Extraire les images et appliquer l'OCR
        image_texts = []
        for page_num, page in enumerate(doc):
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

    logger.info(f"Métadonnées extraites : {content['metadata']}")
    return content


def extract_title(text_content, metadata):
    """
    Extracts the title from metadata or attempts to infer it from the text content.

    If the title is present in the metadata, it is used. Otherwise, the function
    tries to infer the title from the first line of the text content.

    Parameters:
        text_content (str): The full text content extracted from the document.
        metadata (dict): A dictionary of metadata from the document.

    Returns:
        str: The extracted or inferred title, or "Titre inconnu" if no valid title is found.
    """
    title = metadata.get("title", None)
    if not title and text_content:
        title = infer_title(text_content)
    return title or "Titre inconnu"


def infer_title(text_content):
    """
    Infers the title from the first line of the text content.

    This function attempts to extract the first line of text and uses it as the
    title if it meets a minimum length requirement. Titles that are too short
    are rejected.

    Parameters:
        text_content (str): The full text content extracted from the document.

    Returns:
        str: The inferred title if valid, otherwise None.
    """
    first_line = text_content.split("\n")[0].strip()
    if len(first_line) > 5:  # Longueur minimale pour éviter des titres peu informatifs
        title = first_line
        return title
    else:
        logger.info(f"Rejected title '{first_line}'")
        return None

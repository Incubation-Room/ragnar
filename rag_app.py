import os
from pathlib import Path
import streamlit as st
from rag_pipeline import (
    build_context_from_docs,
    normalize_path,
    load_documents,
    split_documents,
    create_retrieval_qa_chain,
    get_initial_prompt,  # Import de la fonction pour gérer le contexte
)
from vector_store import create_vector_store
from chunking import split_documents
# Classe Document pour garantir la compatibilité avec split_documents
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def main():
    st.title("RAG System with Ollama")
    st.write("Upload your files or provide a folder path containing your documents.")

    # Initialisation des états pour gérer les documents et le chat
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Définir un contexte fixe en interne
    context = """
    This system is designed to assist an association in managing its activities.
    It provides relevant answers based on the provided documents. Please ensure your responses are concise, helpful, and aligned with this purpose.
    """

    # Section pour glisser-déposer des fichiers
    uploaded_files = st.file_uploader(
        "Drag and drop files here (or click to upload multiple files)",
        type=["pdf", "docx", "xlsx", "xls", "txt"],
        accept_multiple_files=True
    )
    
    # Get the directory where the Streamlit app is being run
    current_folder = Path(os.getcwd())  # This will be the current directory
    default_folder = current_folder / "dev_data" / "archive_Ca_MR"
    # Use the default folder for the folder path input
    folder_path = st.text_input("Or enter the path to your target folder:", placeholder=default_folder)

    # Bouton pour lancer l’analyse
    if st.button("Analyze"):
        if not uploaded_files and not folder_path:
            folder_path= str(default_folder)

        # Normaliser le chemin si défini
        if folder_path:
            folder_path = normalize_path(folder_path)

        documents = []

        # Charger les documents via rag_pipeline
        if uploaded_files or folder_path:
            try:
                documents = load_documents(
                    uploaded_files if uploaded_files else folder_path,
                    is_directory=bool(folder_path),
                )
                st.session_state.documents = documents
            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")
                return

        # Vérifier si des documents ont été chargés
        if not st.session_state.documents:
            st.warning("No supported documents found.")
        else:
            # Diviser les documents en chunks
            chunks = split_documents(st.session_state.documents)

            # Créer la base vectorielle
            try:
                vector_store = create_vector_store(chunks)
                st.session_state.vector_store = vector_store
                st.success("Knowledge base created! You can now start querying.")
            except Exception as e:
                st.error(f"An error occurred while creating the knowledge base: {e}")
                return

    # Interface de type chat (si la base vectorielle est prête)
    if st.session_state.vector_store:
        st.write("You can now ask questions about your documents!")

        # Ajouter un champ pour poser une question
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask your question:")
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                # Ajouter l'entrée utilisateur à l'historique
                st.session_state.chat_history.append({"role": "user", "message": user_input})

                with st.spinner("Fetching your answer..."):
                    try:
                        # Requête au système RAG
                        retriever, generate_answer = create_retrieval_qa_chain(
                            st.session_state.vector_store,
                            initial_context=context,  # Passer le contexte fixe ici
                        )
                        context_docs = retriever.invoke(user_input)
                        context_retrieved = build_context_from_docs(context_docs)
                        answer = generate_answer(user_input, context_retrieved)

                        # Ajouter la réponse à l'historique
                        st.session_state.chat_history.append({"role": "assistant", "message": answer})

                        # Afficher l'historique du chat
                        st.subheader("Chat History")
                        for msg in st.session_state.chat_history:
                            if msg["role"] == "user":
                                st.markdown(f"**You:** {msg['message']}")
                            else:
                                st.markdown(f"**RAG System:** {msg['message']}")

                        # Afficher les documents sources
                        display_sources(context_docs)


                    except Exception as e:
                        st.error(f"An error occurred during the query: {e}")



# Définir le répertoire de base pour les chemins relatifs (racine de votre projet)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def display_sources(context_docs):
    """
    Affiche les documents sources associés à la réponse générée par le modèle,
    avec un lien vers le fichier source (si disponible) et un moyen de visualiser 
    le contenu du chunk associé.

    Args:
        context_docs (list): Une liste de documents contenant les informations 
                              sur la source et le contenu des chunks.

    Cette fonction fait ce qui suit :
        - Affiche un sous-titre pour les documents sources.
        - Crée un lien vers le fichier source, si le chemin est valide.
        - Affiche le contenu du chunk associé, avec la possibilité de l'étendre
          pour une visualisation détaillée.
    """
    st.subheader("Source Documents")
    
    # Parcourir chaque document dans context_docs
    for doc in context_docs:
        # Récupérer le chemin de la source à partir des métadonnées
        file_source = doc.metadata.get('source', 'Unknown source')
        # Construire un chemin relatif si possible

        # Vérifiez si la source contient un chemin de fichier et si le fichier existe
        if os.path.exists(file_source):
            # Créer un lien vers le fichier source si possible
            # Notez que Streamlit utilise une URL relative pour afficher le lien
            # dans l'application web
            
            # Construire un chemin relatif par rapport au répertoire de base
            file_source_relative = os.path.relpath(file_source, start=BASE_DIR)
            st.markdown(f"[Link to source file]({file_source_relative})")
        else:
            # Affiche simplement la source sous forme de texte si le chemin est inconnu
            st.write(f"Source: {file_source}")

        # Afficher le contenu du chunk avec un bouton pour le développer
        with st.expander(f"View content the chunk at {file_source}"):
            st.write(doc.page_content)

if __name__ == "__main__":
    main()
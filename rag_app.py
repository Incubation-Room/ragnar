import os
from pathlib import Path
import streamlit as st
from rag_pipeline import (
    build_context_from_docs,
    normalize_path,
    create_retrieval_qa_chain,
    get_initial_prompt,  # Import de la fonction pour gérer le contexte
)
from rag_test import load_questions_with_headers
from vector_store import create_vector_store, load_vector_store
from chunking import split_documents
from preprocessing import load_documents
import time

# Classe Document pour garantir la compatibilité avec split_documents
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata




def main():
    # Chargement du banner et titre
    banner_path = str(Path('Images') / "Banniere_ragnar.webp")
    banner_rune_path = str(Path('Images') / "banniere_runes.png")
    st.image(banner_path, use_container_width=True)
    st.markdown("<h2 style='text-align:center;'>⚔️ Quand les tempêtes de données s’élèvent, RAGNAR reste à la barre ⚔️</h2>", unsafe_allow_html=True)
    st.image(banner_rune_path, use_container_width=True)
    st.markdown("### Déposez vos parchemins ou chargez la base des runes existantes.", unsafe_allow_html=True)

    # Initialisation des états
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Contexte interne
    context = """
    This system is designed to assist an association in managing its activities.
    It provides relevant answers based on the provided documents. Please ensure your responses are concise, helpful, and aligned with this purpose.
    """

    # Chargement des questions suggérées à partir d'un fichier fixe
    questions_file_path = "questions_test.txt"  # Adaptez si nécessaire
    questions_dict = load_questions_with_headers(questions_file_path)

    
    

    # Section d'upload et de saisie de chemin
    uploaded_files = st.file_uploader(
        "Déposez vos parchemins ici (ou cliquez pour choisir)",
        type=["pdf", "docx", "xlsx", "xls", "txt"],
        accept_multiple_files=True
    )

    current_folder = Path(os.getcwd())
    default_folder = current_folder / "dev_data" / "archive_Ca_MR"
    folder_path = st.text_input("Ou entrer le chemin de votre répertoire mystique:", placeholder=str(default_folder))

    save_path = ".vector_store"
    existing_db_exists = os.path.exists(save_path)

    col1, col2 = st.columns(2)
    with col1:
        analyze_clicked = st.button("⚒️ Forger la Base")
    with col2:
        load_db_clicked = st.button("🔮 Invoquer la Base Existante", disabled=not existing_db_exists)

    progress_placeholder = st.empty()

    # Bouton Analyze
    if analyze_clicked:
        if not uploaded_files and not folder_path:
            folder_path = str(default_folder)

        if folder_path:
            folder_path = normalize_path(folder_path)

        if uploaded_files or folder_path:
            try:
                with progress_placeholder.container():
                    st.markdown("### 🛠️ Forge en cours...")
                    progress_bar = st.progress(0)

                documents = load_documents(
                    uploaded_files if uploaded_files else folder_path,
                    is_directory=bool(folder_path),
                )
                st.session_state.documents = documents
                progress_bar.progress(30)

            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")
                return

        if not st.session_state.documents:
            st.warning("No supported documents found.")
        else:
            chunks = split_documents(st.session_state.documents)
            progress_bar.progress(60)

            try:
                vector_store = create_vector_store(chunks, save_path=save_path)
                st.session_state.vector_store = vector_store
                progress_bar.progress(100)
                st.success("⚡ Les runes ont été gravées dans la pierre ! La base des connaissances est prête.")
                progress_placeholder.empty()

            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la création de nouvelles runes : {e}")
                return
            finally:
                progress_placeholder.empty()

    # Bouton Load Existing DB
    if load_db_clicked:
        if existing_db_exists:
            with progress_placeholder.container():
                st.markdown("### 🔮 Invocation en cours...")
                progress_bar = st.progress(0)
            try:
                chunks = []
                vector_store = load_vector_store(directory_path=save_path)
                st.session_state.vector_store = vector_store
                for i in range(1, 101, 10):
                    progress_bar.progress(i)
                    time.sleep(0.05)
                st.success("🌌 La base de données ancestrale est invoquée avec succès !")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'invocation des runes existantes : {e}")
            finally:
                progress_placeholder.empty()
        else:
            st.warning("Pas de base de données ancestrale existantes")

    # Interface de chat
    
    if st.session_state.vector_store:
        st.image(banner_rune_path, use_container_width=True)
        st.markdown("### Posez votre question aux runes")

        # Rassembler toutes les questions du dictionnaire en une seule liste
        all_questions = []
        for qlist in questions_dict.values():
            all_questions.extend(qlist)

        # Afficher un menu déroulant avec les questions type, optionnel
        
        selected_question = st.selectbox("Ou choisissez une incantation rituelle :", [""] + all_questions)
        st.image(banner_rune_path, use_container_width=True)

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Posez votre question:", value=selected_question if selected_question else "")
            submitted = st.form_submit_button("Envoyer")

            if submitted and user_input:
                st.session_state.chat_history.append({"role": "user", "message": user_input})

                with st.spinner("Les runes se consultent..."):
                    try:
                        retriever, generate_answer = create_retrieval_qa_chain(
                            st.session_state.vector_store,
                            initial_context=context,
                        )
                        context_docs = retriever.invoke(user_input)
                        context_retrieved = build_context_from_docs(context_docs)
                        answer = generate_answer(user_input, context_retrieved)

                        st.session_state.chat_history.append({"role": "assistant", "message": answer})

                        st.subheader("Historique des Sages Paroles")
                        history = st.session_state.chat_history
                        # On part de la fin et on remonte par étapes de 2
                        for i in range(len(history)-1, -1, -2):
                            user_msg = history[i-1] if i-1 >= 0 else None
                            assistant_msg = history[i]

                            if user_msg and user_msg["role"] == "user":
                                st.markdown(f"**You:** {user_msg['message']}")
                            if assistant_msg and assistant_msg["role"] == "assistant":
                                st.markdown(f"**RAGnar:** {assistant_msg['message']}")



                        display_sources(context_docs)

                    except Exception as e:
                        st.error(f"Une erreur s'est produite lors de l'interrogation des runes: {e}")
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
    st.subheader("Parchemins consultés")
    
    # Parcourir chaque document dans context_docs
    for doc in context_docs:
        # Récupérer le chemin de la source à partir des métadonnées
        file_source = doc.metadata.get('source_path', 'Unknown source')
        file_name = doc.metadata.get('source', 'Unknown source')
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
            st.write(f"Source: {file_name}")

        # Afficher le contenu du chunk avec un bouton pour le développer
        with st.expander(f"View content the chunk at {file_name}"):
            st.write(doc.page_content)

if __name__ == "__main__":
    main()
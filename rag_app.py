import os
import streamlit as st
from rag_pipeline import (
    load_documents,
    split_documents,
    create_vector_store,
    create_retrieval_qa_chain,
)

# Classe Document pour garantir la compatibilité avec split_documents
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def normalize_path(path):
    """
    Normalise un chemin pour s'assurer qu'il est compatible avec les systèmes de fichiers.
    """
    if path.startswith(("'", '"')) and path.endswith(("'", '"')):
        path = path[1:-1]  # Supprime les guillemets simples ou doubles entourant le chemin
    return os.path.abspath(path)  # Renvoie le chemin absolu normalisé

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

    # Section pour glisser-déposer des fichiers
    uploaded_files = st.file_uploader(
        "Drag and drop files here (or click to upload multiple files)",
        type=["pdf", "docx", "xlsx", "xls", "txt"],
        accept_multiple_files=True
    )

    # Section pour entrer un chemin de dossier
    folder_path = st.text_input("Or enter the path to your target folder:", value="")

    # Bouton pour lancer l'analyse
    if st.button("Analyze"):
        if not uploaded_files and not folder_path:
            st.warning("Please upload files or provide a folder path before analyzing.")
            return

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
                        retriever, generate_answer = create_retrieval_qa_chain(st.session_state.vector_store)
                        context_docs = retriever.invoke(user_input)
                        context = "\n".join([doc.page_content for doc in context_docs])
                        answer = generate_answer(user_input, context)

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

def display_sources(context_docs):
    st.subheader("Source Documents")
    for doc in context_docs:
                            # Vérifiez si la source contient un chemin de fichier
        file_source = doc.metadata.get('source', 'Unknown source')
                            
                            # Créer un lien vers le fichier source si possible
        if os.path.exists(file_source):
            st.markdown(f"[Link to source file]({file_source})")
        else:
            st.write(f"Source: {file_source}")

                            # Afficher le contenu du chunk avec un bouton pour le développer
        with st.expander(f"View content of this chunk from {file_source}"):
            st.write(doc.page_content)

if __name__ == "__main__":
    main()
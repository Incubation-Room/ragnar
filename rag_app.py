import os
import streamlit as st
from rag_pipeline import (
    extract_content_from_pdf,
    extract_text_from_excel,
    extract_text_from_word,
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
    
    Parameters:
    - path (str): Le chemin fourni par l'utilisateur.

    Returns:
    - str: Le chemin nettoyé et prêt à l'emploi.
    """
    if path.startswith(("'", '"')) and path.endswith(("'", '"')):
        path = path[1:-1]  # Supprime les guillemets simples ou doubles entourant le chemin
    return os.path.abspath(path)  # Renvoie le chemin absolu normalisé

def main():
    st.title("RAG System with Ollama")
    st.write("Upload your files or provide a folder path containing your documents.")

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

        # Traiter les fichiers glissés-déposés
        if uploaded_files:
            for file in uploaded_files:
                try:
                    # Charger le contenu selon le type de fichier
                    if file.name.endswith(".pdf"):
                        content = extract_content_from_pdf(file)
                    elif file.name.endswith(".docx"):
                        content = extract_text_from_word(file)
                    elif file.name.endswith((".xlsx", ".xls")):
                        content = extract_text_from_excel(file)
                    elif file.name.endswith(".txt"):
                        content = file.read().decode("utf-8")
                    else:
                        continue

                    documents.append(Document(page_content=content, metadata={"source": file.name}))
                except Exception as e:
                    st.warning(f"Error processing file {file.name}: {e}")

        # Traiter les fichiers dans un dossier cible
        if folder_path and os.path.isdir(folder_path):
            st.write(f"Analyzing folder: {folder_path}")
            try:
                total_files = sum([len(files) for _, _, files in os.walk(folder_path)])
                progress_bar = st.progress(0)

                processed_files = 0

                for root, _, files in os.walk(folder_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        file_extension = file_name.split(".")[-1].lower()

                        if file_extension in ["pdf", "docx", "xlsx", "xls", "txt"]:
                            try:
                                # Charger le contenu selon le type de fichier
                                if file_extension == "pdf":
                                    content = extract_content_from_pdf(file_path)
                                elif file_extension == "docx":
                                    content = extract_text_from_word(file_path)
                                elif file_extension in ["xlsx", "xls"]:
                                    content = extract_text_from_excel(file_path)
                                elif file_extension == "txt":
                                    with open(file_path, "r", encoding="utf-8") as txt_file:
                                        content = txt_file.read()
                                else:
                                    continue

                                documents.append(Document(page_content=content, metadata={"source": file_name}))
                            except Exception as e:
                                st.warning(f"Error processing file {file_path}: {e}")

                        # Mettre à jour la barre de progression
                        processed_files += 1
                        progress_bar.progress(processed_files / total_files)

            except Exception as e:
                st.error(f"An error occurred while processing the folder: {e}")

        # Vérifier si des documents ont été chargés
        if not documents:
            st.warning("No supported documents found.")
        else:
            # Filtrer les documents vides
            documents = [doc for doc in documents if doc.page_content.strip()]
            if not documents:
                st.error("All documents are empty after processing.")
                return

            # Diviser les documents en chunks
            chunks = split_documents(documents)

            # Créer la base vectorielle
            vector_store = create_vector_store(chunks)

            st.success("Knowledge base created! You can now start querying.")

            # Interface de type chat
            chat_history = []

            def query_rag(user_query):
                retriever, generate_answer = create_retrieval_qa_chain(vector_store)
                context_docs = retriever.get_relevant_documents(user_query)
                context = "\n".join([doc.page_content for doc in context_docs])
                answer = generate_answer(user_query, context)
                return answer, context_docs

            # Ajouter un champ pour poser une question
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Ask your question:")
                submitted = st.form_submit_button("Send")

                if submitted and user_input:
                    # Ajouter l'entrée utilisateur à l'historique
                    chat_history.append({"role": "user", "message": user_input})

                    with st.spinner("Fetching your answer..."):
                        answer, context_docs = query_rag(user_input)

                        # Ajouter la réponse du système à l'historique
                        chat_history.append({"role": "assistant", "message": answer})

                        # Afficher l'historique du chat
                        st.subheader("Chat History")
                        for msg in chat_history:
                            if msg["role"] == "user":
                                st.markdown(f"**You:** {msg['message']}")
                            else:
                                st.markdown(f"**RAG System:** {msg['message']}")

                        # Afficher les documents sources
                        st.subheader("Source Documents")
                        for doc in context_docs:
                            st.write(f"Source: {doc.metadata['source']}")

if __name__ == "__main__":
    main()
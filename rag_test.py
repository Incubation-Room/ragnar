from datetime import datetime
from rag_pipeline import (
    load_documents,
    split_documents,
    create_retrieval_qa_chain,
    normalize_path,
)
from vector_store import create_vector_store

import os

def load_questions_with_headers(file_path):
    """
    Charge les questions à partir d'un fichier texte, en conservant les en-têtes.

    Parameters:
    - file_path (str): Chemin vers le fichier contenant les questions.

    Returns:
    - dict: Dictionnaire avec les en-têtes comme clés et une liste de questions comme valeurs.
    """
    questions_dict = {}
    current_header = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):  # En-tête de section
                    current_header = line.lstrip("#").strip()
                    questions_dict[current_header] = []
                elif line.startswith("-") and current_header:  # Question sous l'en-tête
                    question = line.lstrip("-").strip()
                    questions_dict[current_header].append(question)
    except Exception as e:
        print(f"Erreur lors du chargement des questions : {e}")
    
    return questions_dict


def main():
    """
    Test le RAG avec les questions provenant d'un fichier texte. Les résultats sont consignés dans un fichier.
    """
    print("=== Test du RAG ===")

    # Charger les questions depuis un fichier texte
    questions_file = "Docs_references/questions_test/questions_test.txt"  # Modifiez ce chemin si nécessaire
    print(f"Chargement des questions depuis {questions_file}...")
    questions_dict = load_questions_with_headers(questions_file)

    if not questions_dict:
        print("Aucune question chargée depuis le fichier.")
        return

    print(f"{sum(len(q) for q in questions_dict.values())} questions chargées avec succès.\n")

    # Demander le chemin des documents
    source_path = input("Entrez le chemin du dossier contenant les documents (laisser vide pour le chemin par défaut) : ").strip()
    if not source_path:
        source_path = '/Users/sebastienstagno/ICAM/Machine Learning/ragnar/dev_data'
        print(f"Chemin par défaut utilisé : {source_path}")

    print(f"Chargement des documents depuis {source_path}...")
    documents = load_documents(source_path, is_directory=True)
    if not documents:
        print("Aucun document valide chargé.")
        return
    print(f"{len(documents)} documents chargés avec succès.")

    chunks = split_documents(documents)
    if not chunks:
        print("Aucun chunk valide généré.")
        return
    print(f"{len(chunks)} chunks générés.")

    try:
        print("Création de la base vectorielle FAISS...")
        vector_store = create_vector_store(chunks)
        print("Base vectorielle FAISS créée avec succès.")
    except RuntimeError as e:
        print(f"Erreur lors de la création de la base vectorielle FAISS : {e}")
        return

    retriever, generate_answer = create_retrieval_qa_chain(vector_store)
    print("Chaîne de récupération et de génération de réponses prête.")


    # Générer un sous-dossier basé sur le timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)  # Crée le dossier si nécessaire

    for header, questions in questions_dict.items():
        # Définir un fichier de sortie par section
        sanitized_header = header.replace(" ", "_").replace("/", "_")  # Assurez-vous que le nom est compatible avec le système de fichiers
        output_file = os.path.join(results_dir, f"results_{sanitized_header}_{timestamp}.txt")
        print(f"\n=== Section : {header} ===")
        print(f"Les résultats pour cette section seront enregistrés dans '{output_file}'.\n")

        # Créer un fichier pour cette section
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"=== Résultats de la section : {header} ===\n\n")

        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"- Question {i}: {question}\n")

            try:
                context_docs = retriever.invoke(question)
                if not context_docs:
                    print("Aucun document pertinent trouvé.\n")
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write("Aucun document pertinent trouvé.\n\n")
                    continue

                # Inclure les documents utilisés pour répondre
                document_titles = [doc.metadata.get('title', 'Titre inconnu') for doc in context_docs]
                document_dates = [doc.metadata.get('date', 'Date inconnue') for doc in context_docs]
                document_info = "\n".join([f"- {title} ({date})" for title, date in zip(document_titles, document_dates)])

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write("Documents utilisés :\n")
                    f.write(f"{document_info}\n\n")

                # Générer la réponse
                context = "\n".join([doc.page_content for doc in context_docs])
                answer = generate_answer(question, context)
                print(f"Réponse : {answer}\n")
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"Réponse : {answer}\n\n")

            except Exception as e:
                print(f"Erreur : {e}\n")
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"Erreur : {e}\n\n")

    print(f"Tests terminés. Résultats enregistrés dans '{output_file}'.")

if __name__ == "__main__":
    main()
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
    source_path = input("Entrez le chemin du dossier contenant les documents : ").strip()
    if not source_path:
        print("Le chemin ne peut pas être vide.")
        return

    source_path = normalize_path(source_path)
    if not os.path.isdir(source_path):
        print("Le chemin fourni n'est pas un dossier valide.")
        return

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

    output_file = "test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Résultats des tests RAG ===\n\n")
    print(f"Les résultats seront enregistrés dans '{output_file}'.\n")

    for header, questions in questions_dict.items():
        print(f"\n# {header}")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n# {header}\n")

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
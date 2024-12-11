import yaml
from datetime import datetime
import os
from rag_pipeline import (
    create_retrieval_qa_chain,
    normalize_path
)
from vector_store import create_vector_store
from preprocessing import load_documents
from chunking import split_documents
from ollama_query import ollama_query
from langchain.schema import Document
from tqdm import tqdm
import time
import re

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



def load_questions_from_yaml(yaml_file):
    """
    Charge les questions et les réponses attendues depuis un fichier YAML.
    
    :param yaml_file: Chemin du fichier YAML contenant les questions et les réponses attendues.
    :return: Une liste de questions avec les réponses attendues et les documents utilisés.
    """
    with open(yaml_file, "r", encoding="utf-8") as file:
        questions_data = yaml.safe_load(file)

    questions = []
    for section, items in questions_data.items():
        for item in items:
            questions.append({
                "question": item["question"],
                "expected_answer": item["response"],
                "documents": item["documents"],  # Les documents de référence
            })
    
    return questions

    
def evaluate_answer_by_llama(question, answer):
    """
    Demande à Llama d'auto-évaluer sa réponse générée.

    :param question: La question posée
    :param answer: La réponse générée
    :return: La note d'évaluation sur une échelle de 1 à 20 et l'explication.
    """
    prompt = f"""
    Voici la réponse générée à la question suivante : {question}
    La réponse générée est : {answer}

    Comment évalues-tu cette réponse sur une échelle de 1 à 20, où :
    - 1 signifie 'très insuffisante' (très incomplète ou incorrecte)
    - 20 signifie 'très pertinente' (très complète et précise)
    
    Utilise des critères comme la pertinence, la complétude et la clarté pour évaluer cette réponse.

    Fournis ta réponse sous la forme suivante :
    - Note : [nombre entre 1 et 20]
    - Explication : [une explication détaillée de ta note]
    """

    evaluation = ollama_query(prompt)
    
    # Séparer la note et l'explication dans la réponse
    try:
        note_part, explanation = evaluation.split('Explication :')
        note = float(note_part.replace('Note :', '').strip())
        
        # S'assurer que la note est entre 1 et 20
        if 1 <= note <= 20:
            # Nettoyage de l'explication pour éviter les retours à la ligne ou autres caractères superflus
            explanation = explanation.strip().replace('\n', ' ').replace('\r', ' ').strip()
            return note, explanation  # Retourne la note et l'explication séparément
        else:
            print(f"Erreur : la note est en dehors de la plage autorisée (1 à 20).")
            return 0, "Note hors plage"
    except ValueError:
        print(f"Erreur dans la réponse d'auto-évaluation pour la question: {question}")
        return 0, "Erreur dans l'évaluation"


def generate_detailed_evaluation_prompt(question, generated_answer, expected_answer, used_docs, reference_docs):
    """
    Crée un prompt pour évaluer la réponse générée par le modèle Llama3.2, en se basant sur les critères du fichier .md.
    Ce prompt renvoie une évaluation complète (détaillée) de la réponse générée.
    """
    # Charger la méthodologie pour l'évaluation des réponses
    with open("/Users/sebastienstagno/ICAM/Machine Learning/ragnar/docs_references/evaluation/methode.md", "r", encoding="utf-8") as file:
        method_content = file.read()

    # Préparer la liste des documents utilisés et de référence
    used_document_titles = [doc.strip() for doc in used_docs]
    reference_document_titles = [doc.strip() for doc in reference_docs]
    matching_docs = [doc for doc in used_document_titles if doc in reference_document_titles]

    detailed_prompt = f"""
    Voici la méthodologie pour évaluer une réponse générée par un modèle RAG :
    
    {method_content}

    Question : {question}
    Réponse générée : {generated_answer}
    Réponse attendue : {expected_answer}
    
    Documents utilisés pour générer cette réponse :
    {', '.join(used_document_titles)}

    Documents de référence attendus :
    {', '.join(reference_document_titles)}

    Documents correspondants :
    {', '.join(matching_docs)}

    Peux-tu évaluer la réponse générée selon les critères suivants :
    - Exactitude (Accuracy)
    - Complétude (Completeness)
    - Clarté (Clarity)
    - Pertinence (Relevance)
    - Utilisation des bons documents (Doc_Matching)
    - Score Global
    
    Merci de fournir une évaluation complète pour chaque critère avec un score en pourcentage dans la forme suivante :
    - Exactitude : [valeur en %]
    - Complétude : [valeur en %]
    - Clarté : [valeur en %]
    - Pertinence : [valeur en %]
    - Utilisation des bons documents : [valeur en %]
    """
    return detailed_prompt

def extract_metrics_from_evaluation_result(evaluation_result):
    """
    Extrait les métriques d'évaluation depuis le texte retourné par Llama.
    Ce texte contient des scores sous la forme '- Accuracy : 60%' ou '- Clarity : 80%'.
    """
    metrics = {
        "accuracy": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "relevance": 0.0,
        "doc_matching": 0.0,
    }

    # Expression régulière pour extraire les scores en pourcentage, avec gestion de la casse et des variations d'espace
    pattern = r"\s*(accuracy|completeness|clarity|relevance|doc_matching)\s*\W*:\s*(\d+)%"

    # Recherche de toutes les correspondances dans l'évaluation
    matches = re.findall(pattern, evaluation_result, re.IGNORECASE)

    # Itérer sur les correspondances et remplir le dictionnaire des métriques
    for match in matches:
        metric_name, value = match
        value = float(value)  # Convertir le pourcentage extrait en float
        
        # Remap des clés en anglais pour correspondre aux métriques attendues
        if 'accuracy' in metric_name.lower():
            metrics["accuracy"] = value
        elif 'completeness' in metric_name.lower():
            metrics["completeness"] = value
        elif 'clarity' in metric_name.lower():
            metrics["clarity"] = value
        elif 'relevance' in metric_name.lower():
            metrics["relevance"] = value
        elif 'doc_matching' in metric_name.lower():
            metrics["doc_matching"] = value

    return metrics

def create_summary_report(evaluation_file, evaluation_results):
    """
    Crée un fichier récapitulatif contenant l'évaluation globale, les notes par métrique et le score global.
    
    :param evaluation_file: Le fichier où enregistrer l'évaluation détaillée.
    :param evaluation_results: Les résultats d'évaluation contenant les notes pour chaque question et chaque critère.
    """
    # Calcul de l'auto-évaluation globale, des moyennes et du score global
    total_auto_eval = sum([result["auto_evaluation"] for result in evaluation_results])  # Maintenant on peut sommer les notes
    total_accuracy = sum([result["accuracy"] for result in evaluation_results])
    total_completeness = sum([result["completeness"] for result in evaluation_results])
    total_clarity = sum([result["clarity"] for result in evaluation_results])
    total_relevance = sum([result["relevance"] for result in evaluation_results])
    total_doc_matching = sum([result["doc_matching"] for result in evaluation_results])

    num_questions = len(evaluation_results)

    avg_auto_eval = total_auto_eval / num_questions
    avg_accuracy = total_accuracy / num_questions
    avg_completeness = total_completeness / num_questions
    avg_clarity = total_clarity / num_questions
    avg_relevance = total_relevance / num_questions
    avg_doc_matching = total_doc_matching / num_questions

    global_score = (avg_accuracy + avg_completeness + avg_clarity + avg_relevance + avg_doc_matching) / 5

    # Enregistrement des résultats dans le fichier récapitulatif
    summary_file = evaluation_file.replace("evaluation", "summary")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Récapitulatif de l'évaluation du RAG\n\n")
        f.write(f"Note moyenne de l'auto-évaluation : {avg_auto_eval:.2f}/20\n")
        f.write(f"Précision moyenne : {avg_accuracy:.2f}%\n")
        f.write(f"Complétude moyenne : {avg_completeness:.2f}%\n")
        f.write(f"Clarté moyenne : {avg_clarity:.2f}%\n")
        f.write(f"Pertinence moyenne : {avg_relevance:.2f}%\n")
        f.write(f"Utilisation des bons documents moyenne : {avg_doc_matching:.2f}%\n")
        f.write(f"Score global moyen : {global_score:.2f}%\n")
    
    print(f"Résumé global enregistré dans '{summary_file}'")

def main():
    print("=== Test du RAG ===")

    # Charger les questions
    questions_file = "/Users/sebastienstagno/ICAM/Machine Learning/ragnar/docs_references/evaluation/questions.yaml"
    print(f"Chargement des questions depuis {questions_file}...")
    questions = load_questions_from_yaml(questions_file)

    if not questions:
        print("Aucune question chargée depuis le fichier.")
        return

    print(f"{len(questions)} questions chargées avec succès.\n")

    # Chemin du dossier de documents
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

    # Créer les chunks à partir des documents
    chunks = split_documents(documents)
    if not chunks:
        print("Aucun chunk valide généré.")
        return
    print(f"{len(chunks)} chunks générés.")

    try:
        # Créer la base vectorielle FAISS
        print("Création de la base vectorielle FAISS...")
        vector_store = create_vector_store(chunks)
        print("Base vectorielle FAISS créée avec succès.")
    except RuntimeError as e:
        print(f"Erreur lors de la création de la base vectorielle FAISS : {e}")
        return

    # Créer la chaîne de récupération et de génération de réponses
    retriever, generate_answer = create_retrieval_qa_chain(vector_store)
    print("Chaîne de récupération et de génération de réponses prête.")

    # Préparer le répertoire de résultats
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    evaluation_file = os.path.join(results_dir, f"evaluation_{timestamp}.txt")
    with open(evaluation_file, "w", encoding="utf-8") as eval_f:
        eval_f.write(f"Évaluation réalisée le : {timestamp}\n\n")

    evaluation_results = []

    # Lancement du traitement des questions
    for i, q in tqdm(enumerate(questions, 1), total=len(questions), desc="Traitement des questions"):
        question = q["question"]
        expected_answer = q["expected_answer"]
        reference_docs = q["documents"]

        context_docs = retriever.invoke(question)
        if not context_docs:
            print(f"Aucun document pertinent trouvé pour la question {i}.")
            continue

        # Préparer les informations sur les documents
        document_titles = [doc.metadata.get('title', 'Titre inconnu') for doc in context_docs]
        document_dates = [doc.metadata.get('date', 'Date inconnue') for doc in context_docs]
        document_info = "\n".join([f"- {title} ({date})" for title, date in zip(document_titles, document_dates)])

        context = "\n".join([doc.page_content for doc in context_docs])
        generated_answer = generate_answer(question, context)

        # Obtenir l'auto-évaluation de la réponse
        auto_evaluation, auto_explanation = evaluate_answer_by_llama(question, generated_answer)

        # Obtenir les métriques de l'évaluation de la réponse générée par Llama
        detailed_evaluation = ollama_query(generate_detailed_evaluation_prompt(question, generated_answer, expected_answer, document_titles, reference_docs))

        # Extraire les métriques du résultat d'évaluation
        metrics = extract_metrics_from_evaluation_result(detailed_evaluation)

        # Ajouter toutes les métriques dans les résultats d'évaluation
        evaluation_results.append({
            "question": question,
            "auto_evaluation": auto_evaluation,
            "auto_explanation": auto_explanation,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "evaluation_result": detailed_evaluation,
            "document_info": document_info,
            **metrics  # Ajout des métriques directement
        })

        # Sauvegarde des résultats détaillés dans le fichier d'évaluation
        with open(evaluation_file, "a", encoding="utf-8") as eval_f:
            eval_f.write(f"Question {i}: {question}\n")
            eval_f.write(f"Réponse générée : {generated_answer}\n")
            eval_f.write(f"Réponse attendue : {expected_answer}\n")
            eval_f.write(f"Documents utilisés : {document_info}\n")
            eval_f.write(f"Évaluation par Llama (auto-évaluation) : {auto_evaluation} - {auto_explanation}\n")
            eval_f.write(f"Évaluation complète par Llama : {detailed_evaluation}\n")
            eval_f.write(f"Exactitude : {metrics['accuracy']:.2f}%\n")
            eval_f.write(f"Complétude : {metrics['completeness']:.2f}%\n")
            eval_f.write(f"Clarté : {metrics['clarity']:.2f}%\n")
            eval_f.write(f"Pertinence : {metrics['relevance']:.2f}%\n")
            eval_f.write(f"Utilisation des bons documents : {metrics['doc_matching']:.2f}%\n\n")

    # Créer le fichier récapitulatif
    create_summary_report(evaluation_file, evaluation_results)

if __name__ == "__main__":
    main()
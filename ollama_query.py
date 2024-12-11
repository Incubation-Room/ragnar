
import requests
import json

def ollama_query(prompt, model="llama3.2", api_url="http://localhost:11434/api/generate"):
    """
    Interroge Ollama via une API REST locale.
    
    Parameters:
    - prompt: La question ou la commande à exécuter.
    - model: Le modèle Ollama à utiliser (par défaut : "llama3.2").
    - api_url: L'URL de l'API Ollama (par défaut : "http://localhost:11434/api/generate").
    
    Returns:
    - La réponse générée par Ollama.
    """
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": 0}
    }
    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()  # Lève une exception si le statut HTTP est une erreur
        text = response.text.strip()
        lines = text.split("\n")
        tokens = list(map(lambda line: json.loads(line)["response"], lines))
        formated = "".join(tokens)
        answer = formated.strip()
        return answer
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erreur lors de la requête à Ollama : {e}")
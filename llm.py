import requests
from vector_db import VectorDB

# Initialize vector DB
vector_db = VectorDB()

# Helper to augment prompt with retrieved context
def augment_prompt_with_context(user_prompt, top_k=3):
    results = vector_db.query(user_prompt, top_k=top_k)
    if not results:
        return user_prompt
    context = "\n".join([f"User: {r['input']}\nJarvis: {r['output']}" for r in results])
    return f"Relevant context:\n{context}\n\nUser: {user_prompt}"


def chat_with_llm(prompt):
    # Augment prompt with retrieved context from vector DB
    augmented_prompt = augment_prompt_with_context(prompt)
    print("Prompt sent to LLM:\n", augmented_prompt)  # Debug: show actual prompt
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b",
        "prompt": augmented_prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            reply = response.json()["response"].strip()
            # Store input/output in vector DB for future retrieval
            vector_db.add(prompt, reply)
            return reply
        else:
            return f"[Error] LLM returned status {response.status_code}"
    except Exception as e:
        return f"[Error] Could not connect to Ollama: {e}"

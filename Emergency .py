from sentence_transformers import SentenceTransformer
import numpy as np

emergency_sentences = [
    "heart attack",
    "stroke",
    "severe bleeding",
    "unconscious",
    "difficulty breathing"
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emergency_embeddings = embedder.encode(emergency_sentences)


def check_emergency(text):
    input_embedding = embedder.encode([text])

    similarities = np.dot(input_embedding, emergency_embeddings.T)
    max_score = np.max(similarities)

    if max_score > 0.6:
        return True

    return False

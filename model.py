# app/model.py

import os
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 1️⃣ LOAD SEMANTIC EMBEDDING MODEL
# =====================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# 2️⃣ MEDICAL KNOWLEDGE BASE (RICH DESCRIPTIONS)
# =====================================================

department_knowledge = {
    "Emergency Medicine": "Life threatening conditions including cardiac arrest, stroke, severe trauma, heavy bleeding, respiratory failure.",
    "General Medicine": "Common illnesses including fever, fatigue, infections, general weakness, non-specific symptoms.",
    "Cardiology": "Heart related disorders including chest pain, heart attack, arrhythmia, hypertension, coronary artery disease.",
    "Neurology": "Brain and nervous system disorders including stroke, seizures, migraine, neuropathy, paralysis.",
    "Dermatology": "Skin diseases including rash, eczema, acne, fungal infection, psoriasis.",
    "Orthopedics": "Bone and joint disorders including fractures, arthritis, joint pain, spine injury.",
    "Pediatrics": "Medical care for infants and children including childhood infections and growth issues.",
    "Psychiatry": "Mental health conditions including depression, anxiety, bipolar disorder, hallucinations.",
    "Gastroenterology": "Digestive system disorders including abdominal pain, vomiting, diarrhea, liver disease.",
    "Pulmonology": "Respiratory diseases including asthma, pneumonia, breathing difficulty, chronic cough.",
    "Urology": "Urinary tract disorders including kidney stones, urinary infections, prostate issues.",
    "Nephrology": "Kidney related diseases including renal failure, dialysis conditions, electrolyte imbalance.",
    "Endocrinology": "Hormonal disorders including diabetes, thyroid disease, metabolic syndrome.",
    "Oncology": "Cancer related conditions including tumor growth, chemotherapy, radiation therapy.",
    "ENT": "Ear, nose and throat disorders including sinusitis, hearing loss, throat infections.",
    "Ophthalmology": "Eye related diseases including vision loss, cataract, glaucoma, eye infection.",
    "Gynecology": "Female reproductive health including menstrual disorders, ovarian cyst, pelvic pain."
}

department_names = list(department_knowledge.keys())
knowledge_texts = list(department_knowledge.values())

# =====================================================
# 3️⃣ BUILD SEMANTIC INDEX (FAISS)
# =====================================================

knowledge_embeddings = embedder.encode(knowledge_texts)
dimension = knowledge_embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(knowledge_embeddings))

# =====================================================
# 4️⃣ OPTIONAL ML CLASSIFIER LOAD (IF TRAINED)
# =====================================================

classifier = None
if os.path.exists("models/trained_model/classifier.pkl"):
    classifier = joblib.load("models/trained_model/classifier.pkl")

# =====================================================
# 5️⃣ HYBRID PREDICTION ENGINE
# =====================================================

def hybrid_predict(symptoms: str):

    # Encode user input
    user_embedding = embedder.encode([symptoms])

    # ---------- SEMANTIC SEARCH ----------
    distances, indices = faiss_index.search(np.array(user_embedding), k=3)

    semantic_scores = 1 / (1 + distances[0])
    semantic_scores = semantic_scores / np.sum(semantic_scores)

    semantic_results = []
    for i, idx in enumerate(indices[0]):
        semantic_results.append({
            "Department": department_names[idx],
            "Semantic Confidence (%)": round(float(semantic_scores[i] * 100), 2)
        })

    # ---------- ML CLASSIFIER (IF AVAILABLE) ----------
    if classifier:
        ml_probs = classifier.predict_proba(user_embedding)[0]
        ml_indices = np.argsort(ml_probs)[::-1][:3]

        ml_results = []
        for idx in ml_indices:
            ml_results.append({
                "Department": classifier.classes_[idx],
                "ML Confidence (%)": round(float(ml_probs[idx] * 100), 2)
            })

        # ---------- HYBRID MERGE ----------
        combined = {}

        for item in semantic_results:
            combined[item["Department"]] = item["Semantic Confidence (%)"] * 0.6

        for item in ml_results:
            if item["Department"] in combined:
                combined[item["Department"]] += item["ML Confidence (%)"] * 0.4
            else:
                combined[item["Department"]] = item["ML Confidence (%)"] * 0.4

        # Sort final
        sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]

        final_results = []
        for dept, score in sorted_final:
            final_results.append({
                "Department": dept,
                "Final Confidence (%)": round(score, 2)
            })

        return {
            "System": "Hybrid Semantic + ML Engine",
            "Top 3 Recommendations": final_results
        }

    # ---------- SEMANTIC ONLY ----------
    return {
        "System": "Semantic Knowledge Engine",
        "Top 3 Recommendations": semantic_results
    }

import os
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# STEP 1: Training Data
# ----------------------------

training_data = [
    ("Chest pain radiating to left arm", "Cardiology"),
    ("Shortness of breath and chest tightness", "Pulmonology"),
    ("Frequent urination and burning sensation", "Urology"),
    ("Skin rash with itching and redness", "Dermatology"),
    ("Severe headache and dizziness", "Neurology"),
    ("Joint pain and swelling", "Orthopedics"),
    ("Fever and persistent cough", "General Medicine"),
    ("Abdominal pain and vomiting", "Gastroenterology"),
    ("Irregular heartbeat and palpitations", "Cardiology"),
    ("Seizure episode and confusion", "Neurology"),
]

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

# ----------------------------
# STEP 2: Load Embedder
# ----------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts)

# ----------------------------
# STEP 3: Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# ----------------------------
# STEP 4: Train Classifier
# ----------------------------

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# ----------------------------
# STEP 5: Evaluate Accuracy
# ----------------------------

predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n🔥 MODEL EVALUATION")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ----------------------------
# STEP 6: FAISS Index (Hybrid)
# ----------------------------

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ----------------------------
# STEP 7: Save Everything
# ----------------------------

os.makedirs("models/trained_model", exist_ok=True)

joblib.dump(classifier, "models/trained_model/classifier.pkl")
joblib.dump(embedder, "models/trained_model/embedder.pkl")
joblib.dump(labels, "models/trained_model/labels.pkl")
faiss.write_index(index, "models/trained_model/faiss.index")

print("\n✅ Training Complete. Models Saved.")


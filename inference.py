import pickle
import numpy as np

# -----------------------------
# Load model and encoder
# -----------------------------
with open("models/triage_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Risk prediction function
# -----------------------------
def predict_risk(patient):
    """
    patient: dict with keys
    Age, Gender, Systolic_BP, Diastolic_BP,
    Heart_Rate, Temperature, Symptoms
    """

    # ---- 1. Compute Mean Arterial Pressure (MAP) ----
    mean_bp = (patient["Systolic_BP"] + 2 * patient["Diastolic_BP"]) / 3

    # ---- 2. Prepare numerical features ----
    X_num = np.array([[
        patient["Age"],
        mean_bp,
        patient["Heart_Rate"],
        patient["Temperature"]
    ]])

    # ---- 3. Prepare categorical features ----
    X_cat = [[
        patient["Gender"],
        patient["Symptoms"]
    ]]

    X_cat_encoded = encoder.transform(X_cat)

    # ---- 4. Combine features ----
    X = np.hstack([X_num, X_cat_encoded])

    # ---- 5. Predict risk ----
    prediction = model.predict(X)[0]

    return prediction
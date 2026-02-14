import pandas as pd
import pickle
import json
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
app = FastAPI()

# Allow your frontend (HTML) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configure Google AI (The "Cloud Brain")
# WE ARE USING YOUR NEW MODEL HERE:
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
ai_model = genai.GenerativeModel('gemini-2.5-flash')

# 3. Load the Local XGBoost Model (The "Edge Brain")
print("⏳ Loading Models...")
bst = XGBClassifier()
bst.load_model("triage_model.json")

# Load the Encoders (to convert "Chest Pain" -> 2)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
print("✅ Models Loaded!")


# --- API ENDPOINTS ---

@app.post("/predict_risk")
async def predict_risk(
        age: int = Form(...),
        gender: str = Form(...),
        sys_bp: int = Form(...),
        dia_bp: int = Form(...),
        hr: int = Form(...),
        temp: float = Form(...),
        symptoms: str = Form(...)
):
    symptoms_lower = symptoms.lower()

    # --- LAYER 1: ABSOLUTE SAFETY RULES (The "Guardrails") ---
    # If vital signs are deadly, force HIGH RISK immediately.
    # This fixes the "200/200 BP" issue instantly.
    if sys_bp >= 180 or dia_bp >= 110 or hr >= 150:
        return {"risk": "High"}

    if "chest" in symptoms_lower or "heart" in symptoms_lower or "stroke" in symptoms_lower or "breath" in symptoms_lower:
        return {"risk": "High"}

    # --- LAYER 2: AI MODEL (For non-obvious cases) ---
    try:
        gender_enc = encoders['gender'].transform([gender])[0]
        # Try to find exact symptom, otherwise map to 'Unknown' (0)
        try:
            symptoms_enc = encoders['symptoms'].transform([symptoms])[0]
        except:
            symptoms_enc = 0
    except:
        gender_enc = 0
        symptoms_enc = 0

    input_data = pd.DataFrame([[age, gender_enc, sys_bp, dia_bp, hr, temp, symptoms_enc]],
                              columns=['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature',
                                       'Symptoms'])

    prediction = bst.predict(input_data)[0]
    risk_map = {0: "Low", 1: "Medium", 2: "High"}

    return {"risk": risk_map[prediction]}


@app.post("/analyze_document")
async def analyze_document(file: UploadFile = File(...)):
    # 1. Read the uploaded file (PDF or Image)
    content = await file.read()

    # 2. Send to Gemini 2.5 Flash
    prompt = """
    Analyze this medical report. Extract the following fields strictly as JSON:
    {"age": int, "systolic_bp": int, "diastolic_bp": int, "heart_rate": int, "temperature": float, "symptoms": "string"}
    If a value is missing, estimate it or put 0.
    """

    # Create the "Blob" for Gemini
    response = ai_model.generate_content([
        {'mime_type': file.content_type, 'data': content},
        prompt
    ])

    # 3. Clean the text to get just JSON
    text = response.text.replace("```json", "").replace("```", "")
    return json.loads(text)


@app.post("/explain")
async def explain_risk(data: dict):
    # Ask Gemini to explain WHY the patient is High Risk
    prompt = f"""
    Patient Data: {data}
    Risk Level: {data.get('risk')}

    Explain strictly in 1 sentence why this patient has this risk level. Use medical terms but keep it simple.
    """
    response = ai_model.generate_content(prompt)
    return {"explanation": response.text}
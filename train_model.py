import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# 1. Load your existing data
print("⏳ Loading data...")
try:
    df = pd.read_csv("patients_dataset.csv")
except FileNotFoundError:
    print("❌ Error: patients_dataset.csv not found!")
    exit()

# 2. Convert text to numbers (Preprocessing)
le_gender = LabelEncoder()
le_symptoms = LabelEncoder()
le_risk = LabelEncoder()
le_dept = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Symptoms'] = le_symptoms.fit_transform(df['Symptoms'])
df['Risk_Level'] = le_risk.fit_transform(df['Risk_Level'])
df['Department'] = le_dept.fit_transform(df['Department'])

# 3. Save the "Translators" (Encoders) so main.py can use them
with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "gender": le_gender,
        "symptoms": le_symptoms,
        "risk": le_risk,
        "dept": le_dept
    }, f)

# 4. Train the Brain (XGBoost)
X = df[['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Symptoms']]
y = df['Risk_Level']

print("🧠 Training Risk Model...")
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X, y)

# 5. Save the Brain
model.save_model("triage_model.json")
print("✅ SUCCESS: 'triage_model.json' created!")
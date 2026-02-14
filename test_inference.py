from inference import predict_risk

sample_patient = {
    "Age": 70,
    "Gender": "Male",
    "Systolic_BP": 85,
    "Diastolic_BP": 55,
    "Heart_Rate": 130,
    "Temperature": 39.0,
    "Symptoms": "chest_pain"
}

risk = predict_risk(sample_patient)
print("Predicted Risk Level:", risk)
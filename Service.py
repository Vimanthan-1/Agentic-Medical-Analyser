from fastapi import APIRouter
import app
from app.schemas import SymptomRequest
from app.model import hybrid_predict
from app.emergency import check_emergency
from app.database import SessionLocal, PredictionLog



router = APIRouter()

@router.post("/predict")
def predict(data: SymptomRequest):

    # 1️⃣ Emergency check first
    if check_emergency(data.symptoms):
        return {
            "Emergency": True,
            "Message": "⚠ Possible medical emergency. Please seek immediate care."
        }

    # 2️⃣ Hybrid prediction
    result = hybrid_predict(data.symptoms)

    # 3️⃣ 🔥 SAVE TO DATABASE HERE
    db = SessionLocal()

    top3 = result["Top 3 Recommendations"]

    log = PredictionLog(
        symptoms=data.symptoms,
        department_1=top3[0]["Department"],
        confidence_1=top3[0]["Final Confidence (%)"],
        department_2=top3[1]["Department"],
        confidence_2=top3[1]["Final Confidence (%)"],
        department_3=top3[2]["Department"],
        confidence_3=top3[2]["Final Confidence (%)"],
        emergency=False
    )

    db.add(log)
    db.commit()
    db.close()

    # 4️⃣ Final return
    return {
        "Emergency": False,
        "System": result["System"],
        "Top 3 Recommendations": result["Top 3 Recommendations"],
        "Similar Past Cases": result.get("Similar Past Cases", [])
    }
@router.get("/analytics")
def analytics():
    db = SessionLocal()

    total = db.query(PredictionLog).count()
    emergencies = db.query(PredictionLog).filter_by(emergency=True).count()

    logs = db.query(PredictionLog).all()

    department_counter = {}

    for log in logs:
        for dept in [log.department_1, log.department_2, log.department_3]:
            if dept not in department_counter:
                department_counter[dept] = 0
            department_counter[dept] += 1

    db.close()

    return {
        "Total Predictions": total,
        "Emergency Cases": emergencies,
        "Department Frequency": department_counter
    }

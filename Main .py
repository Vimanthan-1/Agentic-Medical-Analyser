from app.database import init_db
# Initialize the database when the app starts
from fastapi import FastAPI
from app.services import router

app = FastAPI(
    title="Medical AI Hybrid System",
    version="2.0"
)
init_db()

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Medical AI Hybrid Backend Running"}



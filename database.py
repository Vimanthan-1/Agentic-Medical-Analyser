from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./medical_ai.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symptoms = Column(String)
    department_1 = Column(String)
    confidence_1 = Column(Float)
    department_2 = Column(String)
    confidence_2 = Column(Float)
    department_3 = Column(String)
    confidence_3 = Column(Float)
    emergency = Column(Boolean)


def init_db():
    Base.metadata.create_all(bind=engine)

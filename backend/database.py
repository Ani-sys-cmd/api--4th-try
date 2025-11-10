# backend/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import settings

# SQLAlchemy setup
DATABASE_URL = settings.DATABASE_URL

# SQLite needs check_same_thread=False for FastAPI multithreading
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependency used inside routes
def get_db():
    """Provide a SQLAlchemy session for request scope."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


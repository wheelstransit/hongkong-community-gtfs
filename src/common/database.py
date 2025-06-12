import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def get_db_engine() -> Engine:
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable is not set. "
        )

    print("Creating database engine...")
    engine = create_engine(db_url)
    print("Database engine created successfully.")
    return engine
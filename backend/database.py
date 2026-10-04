"""Database bootstrap and small SQLite migration helper for TRUST-X."""
from __future__ import annotations
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("TRUSTX_DATABASE_URL", "sqlite:///./trustx.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _sqlite_add_missing_columns() -> None:
    if not DATABASE_URL.startswith("sqlite"):
        return
    insp = inspect(engine)
    additions = {
        "models": {"file_path": "TEXT", "architecture": "TEXT"},
        "training_runs": {"batch_id": "TEXT"},
    }
    with engine.begin() as conn:
        for table, columns in additions.items():
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'))

def init_db():
    from . import models_db  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _sqlite_add_missing_columns()

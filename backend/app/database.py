"""SQLAlchemy engine and session setup for PostgreSQL."""

import os
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)

DATABASE_URL = os.getenv("DATABASE_URL")


class Base(DeclarativeBase):
	"""Base class for future SQLAlchemy models."""


engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None

if DATABASE_URL:
	database_url = make_url(DATABASE_URL)
	if database_url.drivername == "postgresql":
		database_url = database_url.set(drivername="postgresql+psycopg2")

	engine = create_engine(database_url, pool_pre_ping=True)
	SessionLocal = sessionmaker(
		bind=engine,
		autoflush=False,
		autocommit=False,
	)


def get_db() -> Generator[Session, None, None]:
	"""Yield a database session and close it after the request."""
	if SessionLocal is None:
		raise RuntimeError("DATABASE_URL is not configured; no database session is available.")

	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def check_database_connection() -> None:
	"""Raise a SQLAlchemy error if PostgreSQL cannot execute a simple query."""
	if engine is None:
		raise RuntimeError("DATABASE_URL is not configured; cannot check the database connection.")

	with engine.connect() as connection:
		connection.execute(text("SELECT 1"))

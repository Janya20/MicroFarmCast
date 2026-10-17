"""Safely verify the isolated MicroFarmCast test database connection."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.database import DATABASE_URL

TEST_DATABASE = "microfarmcast_test_db"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

# Reuse local credentials and connection options, changing only the database name.
test_url = make_url(DATABASE_URL).set(database=TEST_DATABASE)

if test_url.get_backend_name() != "postgresql":
    raise RuntimeError("Refusing to connect: DATABASE_URL must use PostgreSQL.")
if test_url.database != TEST_DATABASE:
    raise RuntimeError("Refusing to connect to an unexpected database.")
if test_url.drivername == "postgresql":
    test_url = test_url.set(drivername="postgresql+psycopg2")

test_engine = create_engine(test_url, pool_pre_ping=True)

try:
    with test_engine.connect() as connection:
        actual_database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        if actual_database != TEST_DATABASE:
            raise RuntimeError(
                f"Safety check failed: connected to {actual_database!r}"
            )

        current_schema = connection.execute(
            text("SELECT current_schema()")
        ).scalar_one()
        search_path = connection.execute(text("SHOW search_path")).scalar_one()
        public_tables = connection.execute(
            text(
                "SELECT tablename "
                "FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public' "
                "ORDER BY tablename"
            )
        ).scalars().all()

        print(f"SUCCESS: Connected to {TEST_DATABASE}.")
        print(f"Verified database: {actual_database}")
        print(f"Current schema: {current_schema}")
        print(f"Search path: {search_path}")
        print(f"Public tables: {public_tables}")
finally:
    test_engine.dispose()

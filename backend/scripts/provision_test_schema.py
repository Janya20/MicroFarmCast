"""Provision the isolated PostgreSQL test schema after explicit confirmation."""

from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

TEST_DATABASE = "microfarmcast_test_db"
PUBLIC_SCHEMA = "public"
REQUIRED_TABLES = {
    "farms",
    "weather_observations",
    "predictions",
    "alerts",
}
EXPECTED_INDEXES = {
    "farms_pkey": ("unique index farms_pkey", "using btree (id)"),
    "weather_observations_pkey": (
        "unique index weather_observations_pkey",
        "using btree (id)",
    ),
    "idx_observations_farm_time": (
        "using btree (farm_id, observed_at desc)",
    ),
    "predictions_pkey": ("unique index predictions_pkey", "using btree (id)"),
    "idx_predictions_farm_time": (
        "using btree (farm_id, predicted_at desc)",
    ),
    "alerts_pkey": ("unique index alerts_pkey", "using btree (id)"),
    "idx_alerts_farm_time": (
        "using btree (farm_id, created_at desc)",
    ),
    "idx_alerts_status": ("using btree (status)",),
}
EXPECTED_CONSTRAINTS = {
    "farms_pkey": ("p", ("primary key (id)",)),
    "farms_latitude_check": (
        "c",
        ("latitude >=", "-90", "latitude <=", "90"),
    ),
    "farms_longitude_check": (
        "c",
        ("longitude >=", "-180", "longitude <=", "180"),
    ),
    "weather_observations_pkey": ("p", ("primary key (id)",)),
    "weather_observations_farm_id_fkey": (
        "f",
        ("foreign key (farm_id)", "references public.farms(id)", "on delete cascade"),
    ),
    "weather_observations_humidity_percent_check": (
        "c",
        ("humidity_percent >=", "(0)", "humidity_percent <=", "(100)"),
    ),
    "predictions_pkey": ("p", ("primary key (id)",)),
    "predictions_farm_id_fkey": (
        "f",
        ("foreign key (farm_id)", "references public.farms(id)", "on delete cascade"),
    ),
    "alerts_pkey": ("p", ("primary key (id)",)),
    "alerts_farm_id_fkey": (
        "f",
        ("foreign key (farm_id)", "references public.farms(id)", "on delete cascade"),
    ),
}

# Each value is (PostgreSQL type, nullable, max length, identity, generation).
EXPECTED_COLUMNS = {
    "farms": {
        "id": ("int8", "NO", None, "YES", "ALWAYS"),
        "name": ("varchar", "NO", 150, "NO", None),
        "latitude": ("float8", "NO", None, "NO", None),
        "longitude": ("float8", "NO", None, "NO", None),
        "crop_type": ("varchar", "YES", 100, "NO", None),
        "location_description": ("text", "YES", None, "NO", None),
        "created_at": ("timestamptz", "NO", None, "NO", None),
    },
    "weather_observations": {
        "id": ("int8", "NO", None, "YES", "ALWAYS"),
        "farm_id": ("int8", "NO", None, "NO", None),
        "temperature_c": ("float8", "YES", None, "NO", None),
        "humidity_percent": ("float8", "YES", None, "NO", None),
        "soil_moisture": ("float8", "YES", None, "NO", None),
        "precipitation_mm": ("float8", "YES", None, "NO", None),
        "wind_speed_mps": ("float8", "YES", None, "NO", None),
        "source": ("varchar", "YES", 50, "NO", None),
        "observed_at": ("timestamptz", "NO", None, "NO", None),
        "created_at": ("timestamptz", "NO", None, "NO", None),
    },
    "predictions": {
        "id": ("int8", "NO", None, "YES", "ALWAYS"),
        "farm_id": ("int8", "NO", None, "NO", None),
        "prediction_type": ("varchar", "NO", 50, "NO", None),
        "predicted_value": ("jsonb", "NO", None, "NO", None),
        "risk_level": ("varchar", "YES", 30, "NO", None),
        "model_name": ("varchar", "YES", 150, "NO", None),
        "predicted_at": ("timestamptz", "NO", None, "NO", None),
        "created_at": ("timestamptz", "NO", None, "NO", None),
    },
    "alerts": {
        "id": ("int8", "NO", None, "YES", "ALWAYS"),
        "farm_id": ("int8", "NO", None, "NO", None),
        "title": ("varchar", "NO", 200, "NO", None),
        "message": ("text", "NO", None, "NO", None),
        "severity": ("varchar", "YES", 30, "NO", None),
        "status": ("varchar", "NO", 30, "NO", None),
        "created_at": ("timestamptz", "NO", None, "NO", None),
        "acknowledged_at": ("timestamptz", "YES", None, "NO", None),
    },
}
EXPECTED_DEFAULTS = {
    ("farms", "created_at"): "now()",
    ("weather_observations", "observed_at"): "now()",
    ("weather_observations", "created_at"): "now()",
    ("predictions", "predicted_at"): "now()",
    ("predictions", "created_at"): "now()",
    ("alerts", "status"): "'open'",
    ("alerts", "created_at"): "now()",
}

CREATE_TABLE_STATEMENTS = (
    """
    CREATE TABLE public.farms (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        name VARCHAR(150) NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        crop_type VARCHAR(100),
        location_description TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT farms_pkey PRIMARY KEY (id),
        CONSTRAINT farms_latitude_check
            CHECK (latitude >= -90 AND latitude <= 90),
        CONSTRAINT farms_longitude_check
            CHECK (longitude >= -180 AND longitude <= 180)
    )
    """,
    """
    CREATE TABLE public.weather_observations (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        farm_id BIGINT NOT NULL,
        temperature_c DOUBLE PRECISION,
        humidity_percent DOUBLE PRECISION,
        soil_moisture DOUBLE PRECISION,
        precipitation_mm DOUBLE PRECISION,
        wind_speed_mps DOUBLE PRECISION,
        source VARCHAR(50),
        observed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT weather_observations_pkey PRIMARY KEY (id),
        CONSTRAINT weather_observations_farm_id_fkey
            FOREIGN KEY (farm_id) REFERENCES public.farms(id) ON DELETE CASCADE,
        CONSTRAINT weather_observations_humidity_percent_check
            CHECK (humidity_percent >= 0 AND humidity_percent <= 100)
    )
    """,
    """
    CREATE TABLE public.predictions (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        farm_id BIGINT NOT NULL,
        prediction_type VARCHAR(50) NOT NULL,
        predicted_value JSONB NOT NULL,
        risk_level VARCHAR(30),
        model_name VARCHAR(150),
        predicted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT predictions_pkey PRIMARY KEY (id),
        CONSTRAINT predictions_farm_id_fkey
            FOREIGN KEY (farm_id) REFERENCES public.farms(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE public.alerts (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        farm_id BIGINT NOT NULL,
        title VARCHAR(200) NOT NULL,
        message TEXT NOT NULL,
        severity VARCHAR(30),
        status VARCHAR(30) NOT NULL DEFAULT 'open',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        acknowledged_at TIMESTAMP WITH TIME ZONE,
        CONSTRAINT alerts_pkey PRIMARY KEY (id),
        CONSTRAINT alerts_farm_id_fkey
            FOREIGN KEY (farm_id) REFERENCES public.farms(id) ON DELETE CASCADE
    )
    """,
)

CREATE_INDEX_STATEMENTS = (
    "CREATE INDEX idx_observations_farm_time ON public.weather_observations USING btree (farm_id, observed_at DESC)",
    "CREATE INDEX idx_predictions_farm_time ON public.predictions USING btree (farm_id, predicted_at DESC)",
    "CREATE INDEX idx_alerts_farm_time ON public.alerts USING btree (farm_id, created_at DESC)",
    "CREATE INDEX idx_alerts_status ON public.alerts USING btree (status)",
)

CREATED_OBJECT_NAMES = REQUIRED_TABLES | set(EXPECTED_INDEXES) | {
    f"{table}_id_seq" for table in REQUIRED_TABLES
}


def _database_url():
    env_file = Path(__file__).resolve().parents[1] / ".env"
    env_values = dotenv_values(env_file)
    configured_url = env_values.get("DATABASE_URL")
    if not configured_url:
        raise RuntimeError(f"DATABASE_URL is missing from {env_file}.")

    source_url = make_url(configured_url)
    if source_url.get_backend_name() != "postgresql":
        raise RuntimeError("Refusing to proceed: DATABASE_URL must use PostgreSQL.")

    test_url = source_url.set(database=TEST_DATABASE).set(
        drivername="postgresql+psycopg2"
    )
    if test_url.database != TEST_DATABASE:
        raise RuntimeError("Refusing to connect to an unexpected database.")
    return test_url


def _preflight(connection):
    actual_database = connection.execute(
        text("SELECT current_database()")
    ).scalar_one()
    if actual_database != TEST_DATABASE:
        raise RuntimeError(
            f"Refusing to execute DDL in unexpected database {actual_database!r}."
        )

    current_schema = connection.execute(
        text("SELECT current_schema()")
    ).scalar_one()
    if current_schema != PUBLIC_SCHEMA:
        raise RuntimeError(
            f"Refusing to execute DDL outside public; current schema is {current_schema!r}."
        )

    public_tables = set(
        connection.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public'"
            )
        ).scalars()
    )
    if public_tables:
        raise RuntimeError(
            "Refusing to provision: public already contains tables: "
            + ", ".join(sorted(public_tables))
        )

    existing_relations = set(
        connection.execute(
            text(
                "SELECT c.relname FROM pg_catalog.pg_class AS c "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public'"
            )
        ).scalars()
    )
    relation_conflicts = CREATED_OBJECT_NAMES & existing_relations
    if relation_conflicts:
        raise RuntimeError(
            "Refusing to overwrite existing public relations: "
            + ", ".join(sorted(relation_conflicts))
        )

    existing_types = set(
        connection.execute(
            text(
                "SELECT t.typname FROM pg_catalog.pg_type AS t "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = t.typnamespace "
                "WHERE n.nspname = 'public'"
            )
        ).scalars()
    )
    type_conflicts = REQUIRED_TABLES & existing_types
    if type_conflicts:
        raise RuntimeError(
            "Refusing to overwrite existing public types: "
            + ", ".join(sorted(type_conflicts))
        )

    return actual_database, current_schema, public_tables


def _verify_schema(connection):
    tables = set(
        connection.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public'"
            )
        ).scalars()
    )
    if tables != REQUIRED_TABLES:
        raise RuntimeError(f"Table verification failed: found {sorted(tables)}.")

    column_rows = connection.execute(
        text(
            "SELECT table_name, column_name, udt_name, is_nullable, "
            "character_maximum_length, is_identity, identity_generation, "
            "column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name IN ('farms', 'weather_observations', 'predictions', 'alerts')"
        )
    ).all()
    actual_columns = {}
    actual_defaults = {}
    for row in column_rows:
        key = (row.table_name, row.column_name)
        actual_columns[key] = (
            row.udt_name,
            row.is_nullable,
            row.character_maximum_length,
            row.is_identity,
            row.identity_generation,
        )
        if row.column_default is not None:
            actual_defaults[key] = row.column_default.lower()

    expected_columns = {
        (table_name, column_name): definition
        for table_name, columns in EXPECTED_COLUMNS.items()
        for column_name, definition in columns.items()
    }
    if actual_columns != expected_columns:
        raise RuntimeError("Column, type, nullability, or identity verification failed.")

    for key, required_default in EXPECTED_DEFAULTS.items():
        actual_default = actual_defaults.get(key, "")
        if required_default not in actual_default:
            raise RuntimeError(f"Default verification failed for {key!r}.")
    unexpected_defaults = set(actual_defaults) - set(EXPECTED_DEFAULTS)
    if unexpected_defaults:
        raise RuntimeError(
            "Unexpected column defaults found: "
            + ", ".join(f"{table}.{column}" for table, column in sorted(unexpected_defaults))
        )

    constraint_rows = connection.execute(
        text(
            "SELECT con.conname, con.contype, "
            "pg_catalog.pg_get_constraintdef(con.oid) AS definition "
            "FROM pg_catalog.pg_constraint AS con "
            "JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' "
            "AND c.relname IN ('farms', 'weather_observations', 'predictions', 'alerts') "
            "AND con.contype <> 'n'"
        )
    ).all()
    actual_constraints = {
        row.conname: (row.contype, row.definition.lower())
        for row in constraint_rows
    }
    if set(actual_constraints) != set(EXPECTED_CONSTRAINTS):
        raise RuntimeError("Constraint names do not match the verified schema.")
    for name, (constraint_type, required_fragments) in EXPECTED_CONSTRAINTS.items():
        actual_type, definition = actual_constraints[name]
        if actual_type != constraint_type or not all(
            fragment in definition for fragment in required_fragments
        ):
            raise RuntimeError(f"Constraint verification failed for {name}.")

    index_rows = connection.execute(
        text(
            "SELECT indexname, indexdef FROM pg_catalog.pg_indexes "
            "WHERE schemaname = 'public' "
            "AND tablename IN ('farms', 'weather_observations', 'predictions', 'alerts')"
        )
    ).all()
    actual_indexes = {row.indexname: row.indexdef.lower() for row in index_rows}
    if set(actual_indexes) != set(EXPECTED_INDEXES):
        raise RuntimeError(
            f"Index names do not match the verified schema: {sorted(actual_indexes)}."
        )
    for name, required_fragments in EXPECTED_INDEXES.items():
        if not all(fragment in actual_indexes[name] for fragment in required_fragments):
            raise RuntimeError(f"Index definition verification failed for {name}.")


def main():
    test_url = _database_url()
    engine = create_engine(test_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            actual_database, current_schema, public_tables = _preflight(connection)
        print(f"Verified database: {actual_database}")
        print(f"Verified schema: {current_schema}")
        print(f"Existing public tables: {sorted(public_tables)}")
        print()
        print(f"To confirm, enter exactly: {TEST_DATABASE}")
        print("Enter only this token, not a shell command.")
        confirmation = input("Confirmation token: ")
        if confirmation != TEST_DATABASE:
            raise RuntimeError(
                f"Confirmation did not match; expected exactly {TEST_DATABASE!r}. "
                "No DDL was executed."
            )

        with engine.begin() as connection:
            _preflight(connection)
            for statement in CREATE_TABLE_STATEMENTS:
                connection.exec_driver_sql(statement)
            for statement in CREATE_INDEX_STATEMENTS:
                connection.exec_driver_sql(statement)
            _verify_schema(connection)

        print(
            "SUCCESS: Provisioned and verified public tables: "
            + ", ".join(sorted(REQUIRED_TABLES))
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

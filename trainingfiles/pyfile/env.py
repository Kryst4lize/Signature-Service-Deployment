"""
migrations/env.py
─────────────────────────────────────────────────────────────────────────────
Alembic environment script.

Reads the database URL from the same environment variables used by db_utils
so you never have to hard-code credentials here.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

# ── Alembic config object ──────────────────────────────────────────────────
config = context.config
fileConfig(config.config_file_name)

# ── Override URL from environment ──────────────────────────────────────────
host     = os.environ["DB_HOST"]
port     = os.getenv("DB_PORT", "5432")
name     = os.environ["DB_NAME"]
user     = os.environ["DB_USER"]
password = os.environ["DB_PASSWORD"]
config.set_main_option(
    "sqlalchemy.url",
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
)

# ── Import your models so autogenerate can diff them ──────────────────────
from db_utils import Base  # noqa: E402

target_metadata = Base.metadata


# ─────────────────────────────────────────────────────────────────────────────
# Offline mode  (generates SQL without connecting)
# ─────────────────────────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ─────────────────────────────────────────────────────────────────────────────
# Online mode  (connects and migrates)
# ─────────────────────────────────────────────────────────────────────────────

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

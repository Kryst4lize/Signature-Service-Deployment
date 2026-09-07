"""Alembic environment.

The database URL comes from the application's own `Settings`, so a migration
cannot end up pointed at a different database than the service. Nothing here
reads a credential from alembic.ini.

The engine is async because the app's URL is `postgresql+asyncpg`; alembic
drives it through `run_sync`.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from app.config import settings
from app.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url)

# `Base.metadata` is what `alembic revision --autogenerate` diffs against, so
# adding a column to app/db.py and running autogenerate produces the migration.
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Keep autogenerate away from the ANN indexes.

    They are expression indexes over `binary_quantize(...)::bit(4096)`, which
    SQLAlchemy does not model. Left visible, autogenerate would propose
    dropping them on every run.
    """
    return not (type_ == "index" and name and name.startswith("idx_items_ann_"))


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

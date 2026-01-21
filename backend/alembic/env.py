from __future__ import with_statement

import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

# add the app path so imports work
here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, '..'))

from app.config import settings
from app.database import Base

# Import models so Alembic autogenerate can detect table metadata
import app.models  # noqa: F401

# If the configured URL is async (e.g. sqlite+aiosqlite), convert to sync URL
sync_db_url = settings.database_url
if sync_db_url.startswith("sqlite+aiosqlite:///"):
    sync_db_url = sync_db_url.replace("sqlite+aiosqlite:///", "sqlite:///")
elif "+asyncpg" in sync_db_url:
    sync_db_url = sync_db_url.replace("+asyncpg", "")

config.set_main_option('sqlalchemy.url', sync_db_url)

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
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

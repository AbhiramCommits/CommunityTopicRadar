"""Database engine, session, and portable upsert helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from ctr.config import load_settings
from ctr.models import Base

logger = logging.getLogger(__name__)


def get_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine for the supplied database URL."""
    if url.startswith("sqlite") and ":memory:" not in url:
        connect_args = {"check_same_thread": False}
    else:
        connect_args = {}
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    """Create every warehouse table and index if it does not exist."""
    Base.metadata.create_all(engine)
    logger.info("Initialized database schema at %s", engine.url)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Provide a transactional SQLAlchemy session."""
    active_engine = engine or get_engine(load_settings().database_url)
    factory = sessionmaker(bind=active_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database transaction rolled back")
        raise
    finally:
        session.close()


def _dedupe(rows: Iterable[Mapping[str, Any]], pk_cols: Sequence[str]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        record = dict(row)
        key = tuple(record.get(column) for column in pk_cols)
        unique[key] = record
    return list(unique.values())


def upsert_many(
    session: Session,
    model: type[Base],
    rows: Iterable[Mapping[str, Any]],
    pk_cols: Sequence[str],
) -> int:
    """Insert or update rows on SQLite and PostgreSQL using each dialect's upsert.

    Rows are de-duplicated by their conflict key before they reach the database,
    which keeps a single multi-row statement valid when the input has duplicates.
    """
    records = _dedupe(rows, pk_cols)
    if not records:
        return 0

    table = model.__table__
    dialect = session.get_bind().dialect.name
    columns = list(records[0].keys())
    non_pk = [column for column in columns if column not in pk_cols]
    batch_size = max(1, 900 // max(1, len(columns)))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if dialect == "sqlite":
            statement = sqlite_insert(table).values(batch)
            if non_pk:
                statement = statement.on_conflict_do_update(
                    index_elements=list(pk_cols),
                    set_={column: getattr(statement.excluded, column) for column in non_pk},
                )
            else:
                statement = statement.on_conflict_do_nothing(index_elements=list(pk_cols))
            session.execute(statement)
        elif dialect == "postgresql":
            statement = pg_insert(table).values(batch)
            if non_pk:
                statement = statement.on_conflict_do_update(
                    index_elements=list(pk_cols),
                    set_={column: getattr(statement.excluded, column) for column in non_pk},
                )
            else:
                statement = statement.on_conflict_do_nothing(index_elements=list(pk_cols))
            session.execute(statement)
        else:
            for record in batch:
                identity = {column: record[column] for column in pk_cols}
                existing = session.scalar(select(model).filter_by(**identity))
                if existing is None:
                    session.add(model(**record))
                else:
                    for key, value in record.items():
                        setattr(existing, key, value)
    session.flush()
    return len(records)

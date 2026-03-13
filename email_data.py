import os
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote_plus

from sqlalchemy import Date, DateTime, MetaData, and_, create_engine, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.sqltypes import String, Text, Unicode, UnicodeText

TEXT_TYPES = (String, Text, Unicode, UnicodeText)

EMAIL_SNIPPET_KEYS = [
    "subject",
    "sender_email",
    "recipient_emails",
    "received_date",
    "search_domain",
    "ai_summary",
    "body_text",
]

ATTACHMENT_SNIPPET_KEYS = [
    "file_name",
    "local_file_path",
    "extracted_content",
]


def build_email_database_url() -> str:
    url = os.getenv("EMAILS_DATABASE_URL") or os.getenv("EMAILS_DB_URL")
    if url:
        return url

    host = os.getenv("EMAILS_DB_HOST") or os.getenv("DB_HOST", "avo-adb-002.postgres.database.azure.com")
    port = os.getenv("EMAILS_DB_PORT") or os.getenv("DB_PORT", "5432")
    name = os.getenv("EMAILS_DB_NAME", "avo_emails")
    user = os.getenv("EMAILS_DB_USER") or os.getenv("DB_USER", "administrationSTS")
    password_raw = os.getenv("EMAILS_DB_PASSWORD") or os.getenv("DB_PASSWORD", "")
    password = quote_plus(password_raw)
    sslmode = os.getenv("EMAILS_DB_SSLMODE") or os.getenv("DB_SSLMODE", "require")

    base = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return f"{base}?sslmode={sslmode}" if sslmode else base


@lru_cache(maxsize=1)
def get_email_engine() -> Engine:
    return create_engine(build_email_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_email_tables():
    metadata = MetaData()
    metadata.reflect(bind=get_email_engine(), only=["emails", "attachments"])
    tables = {}
    for name in ["emails", "attachments"]:
        if name in metadata.tables:
            tables[name] = metadata.tables[name]
    return tables


def get_email_table(name: str):
    return get_email_tables().get(name)


def row_to_dict(row):
    if row is None:
        return None
    return dict(row._mapping)


def is_text_column(column) -> bool:
    try:
        if hasattr(column.type, "python_type"):
            return column.type.python_type is str
    except NotImplementedError:
        return False
    return isinstance(column.type, TEXT_TYPES)


def is_date_column(column) -> bool:
    return isinstance(column.type, (Date, DateTime))


def first_column(table, candidates):
    for name in candidates:
        if name in table.c:
            return table.c[name]
    return None


def _date_value_to_str(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_date_filter(
    table,
    date_from=None,
    date_to=None,
    date_column_candidates=None,
):
    if not date_from and not date_to:
        return None

    candidates = date_column_candidates or [
        "received_date",
        "received_at",
        "sent_at",
        "created_at",
        "date",
        "timestamp",
    ]
    date_col = first_column(table, candidates)
    if date_col is None:
        return None

    filters = []
    if is_date_column(date_col):
        if date_from:
            filters.append(date_col >= date_from)
        if date_to:
            filters.append(date_col <= date_to)
    else:
        date_expr = func.substring(date_col, 1, 10)
        date_from_str = _date_value_to_str(date_from)
        date_to_str = _date_value_to_str(date_to)
        if date_from_str:
            filters.append(date_expr >= date_from_str)
        if date_to_str:
            filters.append(date_expr <= date_to_str)

    if not filters:
        return None
    return and_(*filters) if len(filters) > 1 else filters[0]


def run_email_query(statement):
    engine = get_email_engine()
    try:
        with engine.begin() as conn:
            result = conn.execute(statement)
            return [row_to_dict(row) for row in result.fetchall()]
    except SQLAlchemyError as exc:
        raise RuntimeError("Database error") from exc


def search_table(table, query: str, limit: int):
    text_columns = [column for column in table.c if is_text_column(column)]
    if not text_columns:
        return []
    pattern = f"%{query}%"
    filters = [column.ilike(pattern) for column in text_columns]
    statement = select(table).where(or_(*filters)).limit(limit)
    return run_email_query(statement)


def list_table_with_filters(table, limit: int, offset: int, date_from=None, date_to=None):
    statement = select(table)
    date_filter = build_date_filter(table, date_from=date_from, date_to=date_to)
    if date_filter is not None:
        statement = statement.where(date_filter)
    statement = statement.limit(limit).offset(offset)
    return run_email_query(statement)


def search_table_with_filters(
    table,
    query: str | None,
    limit: int,
    date_from=None,
    date_to=None,
):
    filters = []
    if query:
        text_columns = [column for column in table.c if is_text_column(column)]
        if text_columns:
            pattern = f"%{query}%"
            filters.append(or_(*[column.ilike(pattern) for column in text_columns]))

    date_filter = build_date_filter(table, date_from=date_from, date_to=date_to)
    if date_filter is not None:
        filters.append(date_filter)

    statement = select(table)
    if filters:
        statement = statement.where(and_(*filters))
    statement = statement.limit(limit)
    return run_email_query(statement)


def clamp_text(value, limit: int = 180) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def build_row_snippet(row, keys):
    if not row:
        return ""
    parts = []
    for key in keys:
        if key in row and row[key] not in (None, ""):
            parts.append(f"{key}: {clamp_text(row[key])}")
        if len(parts) >= 3:
            break
    return " | ".join(parts)


def build_rows_block(rows, keys, limit: int = 5) -> str:
    if not rows:
        return "Aucun."
    lines = []
    for row in rows[:limit]:
        snippet = build_row_snippet(row, keys)
        if snippet:
            lines.append(f"- {snippet}")
    return "\n".join(lines) if lines else "Aucun."


def generate_email_report():
    tables = get_email_tables()
    engine = get_email_engine()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": list(tables.keys()),
        "available_columns": {name: [col.name for col in table.c] for name, table in tables.items()},
    }

    try:
        with engine.begin() as conn:
            if "emails" in tables:
                emails = tables["emails"]
                report["total_emails"] = (
                    conn.execute(select(func.count()).select_from(emails)).scalar() or 0
                )

                sender_col = first_column(
                    emails,
                    [
                        "sender_email",
                        "sender",
                        "from_email",
                        "from_address",
                        "from",
                    ],
                )
                if sender_col is not None:
                    rows = conn.execute(
                        select(sender_col.label("sender"), func.count().label("count"))
                        .where(sender_col.isnot(None))
                        .group_by(sender_col)
                        .order_by(func.count().desc())
                        .limit(20)
                    ).fetchall()
                    report["emails_by_sender"] = [
                        {"sender": row.sender, "count": row.count} for row in rows
                    ]

                date_col = first_column(
                    emails, ["received_date", "received_at", "sent_at", "created_at", "date", "timestamp"]
                )
                if date_col is not None:
                    if is_date_column(date_col):
                        rows = conn.execute(
                            select(func.date(date_col).label("date"), func.count().label("count"))
                            .where(date_col.isnot(None))
                            .group_by(func.date(date_col))
                            .order_by(func.date(date_col).desc())
                            .limit(30)
                        ).fetchall()
                        report["emails_by_date"] = [
                            {
                                "date": row.date.isoformat() if row.date else None,
                                "count": row.count,
                            }
                            for row in rows
                        ]
                    else:
                        date_key = func.substring(date_col, 1, 10)
                        rows = conn.execute(
                            select(date_key.label("date"), func.count().label("count"))
                            .where(date_col.isnot(None))
                            .group_by(date_key)
                            .order_by(func.count().desc())
                            .limit(30)
                        ).fetchall()
                        report["emails_by_date"] = [
                            {"date": row.date, "count": row.count} for row in rows
                        ]

            if "attachments" in tables:
                attachments = tables["attachments"]
                report["total_attachments"] = (
                    conn.execute(select(func.count()).select_from(attachments)).scalar() or 0
                )

                extension_col = first_column(
                    attachments, ["file_extension", "extension", "ext"]
                )
                if extension_col is not None:
                    rows = conn.execute(
                        select(extension_col.label("extension"), func.count().label("count"))
                        .where(extension_col.isnot(None))
                        .group_by(extension_col)
                        .order_by(func.count().desc())
                        .limit(20)
                    ).fetchall()
                    report["attachments_by_extension"] = [
                        {"extension": row.extension, "count": row.count} for row in rows
                    ]

    except SQLAlchemyError as exc:
        raise RuntimeError("Database error") from exc

    return report

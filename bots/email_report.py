from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, func, or_, select

from email_data import (
    ATTACHMENT_SNIPPET_KEYS,
    EMAIL_SNIPPET_KEYS,
    build_date_filter,
    build_rows_block,
    first_column,
    get_email_engine,
    get_email_tables,
    is_date_column,
    is_text_column,
    run_email_query,
)
from groq_client import client as groq_client, MODEL as GROQ_MODEL
from openai_client import client as oai_client, MODEL as OAI_MODEL

LLM_SYSTEM_PROMPT = (
    "You are a request parser for an email analytics assistant. "
    "Return ONLY a JSON object (no markdown, no extra text). "
    "Today is {today}. "
    "JSON schema:\n"
    "{{"
    "\"intent\": \"search|report|both|clarify\", "
    "\"language\": \"ISO 639-1 code\" or null, "
    "\"query\": string or null, "
    "\"date_from\": \"YYYY-MM-DD\" or null, "
    "\"date_to\": \"YYYY-MM-DD\" or null, "
    "\"tables\": [\"emails\", \"attachments\"] or null, "
    "\"filters\": {{\"sender\": string|null, \"recipient\": string|null, \"subject\": string|null, "
    "\"client\": string|null, \"domain\": string|null, \"extension\": string|null, "
    "\"file_name\": string|null}} or null, "
    "\"needs_clarification\": boolean, "
    "\"clarification_question\": string or null"
    "}}\n"
    "Rules:\n"
    "- If the user asks for statistics, summary, or report (e.g., report, statistics), intent includes report.\n"
    "- If the user asks to find/show/list/retrieve information (e.g., search, find, show), intent includes search.\n"
    "- If both are requested, intent is \"both\".\n"
    "- If request is too vague (no topic, no report request, no timeframe), "
    "set intent to \"clarify\", needs_clarification=true, and add a short question in the user's language.\n"
    "- Detect the user's language and set language to an ISO 639-1 code (e.g., en, fr, ar).\n"
    "- Convert relative dates (e.g. last week, last month) to absolute dates based on today.\n"
    "- If a single date is mentioned, set both date_from and date_to to that date.\n"
    "- If tables are not mentioned, return null for tables.\n"
    "- If a condition maps to a specific field (sender, recipient, subject, client, domain, "
    "extension, file_name), fill it.\n"
    "- Use query ONLY for content keywords (e.g., invoice number, contract, reference). "
    "If the user only asks to list details or show emails, set query to null.\n"
    "- If the user mentions a customer/company name without an email, prefer client (search_domain) "
    "and do NOT set recipient.\n"
    "- Put remaining constraints into query.\n"
)

FIELD_FILTER_CANDIDATES = {
    "sender": ["sender_email", "sender", "from_email", "from_address", "from"],
    "recipient": ["recipient_emails", "to_email", "to_address", "to", "recipient"],
    "subject": ["subject", "email_subject", "title"],
    "client": ["search_domain", "domain", "sender_domain", "recipient_domain"],
    "domain": ["search_domain", "domain", "sender_domain", "recipient_domain"],
    "extension": ["file_extension", "extension", "ext"],
    "file_name": ["file_name", "filename", "name"],
}

PRESERVE_TAG = "preserve"
IMPORTANT_EMAILS_MAX_CANDIDATES = 500
IMPORTANT_EMAILS_MAX_OUTPUT = 30
EMAIL_ADDRESS_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
ATTACHMENT_EMAIL_LINK_COLUMN = "message_id"
STARTER_MESSAGES = {
    "ask about emails": "search",
    "ask about email": "search",
    "generate reports": "report",
    "generate report": "report",
}


def _normalize_language(value: Any) -> str:
    if not value:
        return "en"
    text = str(value).strip().lower()
    return text if re.fullmatch(r"[a-z]{2}", text) else "en"


def _wrap_value(value: Any) -> str:
    if value is None:
        text = "-"
    else:
        text = str(value)
    return f"<{PRESERVE_TAG}>{text}</{PRESERVE_TAG}>"


def _strip_preserve_tags(text: str) -> str:
    if not text:
        return text
    return re.sub(rf"</?{PRESERVE_TAG}>", "", text)


def _normalize_links(text: str) -> str:
    if not text:
        return text
    cleaned = str(text)
    cleaned = re.sub(r"([^\s])<\s*((?:https?|mailto|tel):[^>]+?)\s*>", r"\1 \2", cleaned)
    cleaned = re.sub(r"(https?)\s*:\s*/\s*/", r"\1://", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(mailto|tel)\s*:\s*", r"\1:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*((?:https?|mailto|tel):[^>]+?)\s*>", r"\1", cleaned)
    return cleaned


def _detect_language_with_llm(message: str) -> str:
    if not message:
        return "en"
    prompt = (
        "Detect the language of the user's message and reply with ONLY the ISO 639-1 code "
        "(e.g., en, fr, ar)."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    raw = _call_llm(messages)
    return _normalize_language(raw)


def _translate_response(text: str, language: str) -> str:
    if not text:
        return text
    if language == "en":
        return _strip_preserve_tags(text)
    prompt = (
        "Translate the assistant response to the language with ISO 639-1 code {language}. "
        "Do NOT translate text inside "
        f"<{PRESERVE_TAG}>...</{PRESERVE_TAG}> tags. Keep those tags unchanged. "
        "Preserve emails, dates, numbers, URLs, and formatting (including line breaks). "
        "Return plain text only."
    ).format(language=language)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    translated = _call_llm(messages)
    return _strip_preserve_tags(translated or text)


def _split_sender_groups(items: list[dict], company_keywords: list[str]) -> tuple[list[dict], list[dict]]:
    if not items:
        return [], []
    company = []
    customers = []
    keywords = [k.lower() for k in company_keywords if k]
    for row in items:
        sender = (row.get("sender") or "").lower()
        if sender and any(k in sender for k in keywords):
            company.append(row)
        else:
            customers.append(row)
    return company, customers


def _split_contact_groups(contacts: list[str], company_keywords: list[str]) -> tuple[list[str], list[str]]:
    if not contacts:
        return [], []
    company = []
    customers = []
    keywords = [k.lower() for k in company_keywords if k]
    for contact in contacts:
        value = str(contact or "").lower()
        if value and any(k in value for k in keywords):
            company.append(contact)
        else:
            customers.append(contact)
    return company, customers


def _format_contact_items(items) -> str:
    if not items:
        return "None"
    parts = []
    for item in items:
        if isinstance(item, dict):
            label = item.get("sender") or "unknown"
        else:
            label = item or "unknown"
        parts.append(str(label))
    return ", ".join(parts)


def _extract_emails(value: Any) -> list[str]:
    if not value:
        return []
    text = str(value)
    return re.findall(EMAIL_ADDRESS_REGEX, text)


def _normalize_contact_list(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for raw in values:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _starter_prompt(kind: str) -> str:
    if kind == "report":
        return (
            "To generate a complete report, please provide filters such as customer, "
            "date range, sender/recipient, and subject or keywords. Example: Generate a report "
            "for client Valeo, dates 2026-02-01..2026-02-10, subject RFQ."
        )
    return (
        "To give a complete answer, please provide filters such as customer, "
        "date range, sender/recipient, and subject or keywords. Example: Show emails from "
        "Valeo between 2026-02-01 and 2026-02-10 about RFQ."
    )


def _detect_starter_message(message: str) -> str | None:
    if not message:
        return None
    cleaned = " ".join(message.strip().lower().split())
    return STARTER_MESSAGES.get(cleaned)


def _has_filter_values(filters: dict | None) -> bool:
    if not isinstance(filters, dict):
        return False
    return any(bool(value) for value in filters.values())


def _find_attachment_email_link(attachments_table, emails_table):
    if attachments_table is None or emails_table is None:
        return None, None
    if ATTACHMENT_EMAIL_LINK_COLUMN in attachments_table.c and ATTACHMENT_EMAIL_LINK_COLUMN in emails_table.c:
        return (
            attachments_table.c[ATTACHMENT_EMAIL_LINK_COLUMN],
            emails_table.c[ATTACHMENT_EMAIL_LINK_COLUMN],
        )
    return None, None


def _resolve_attachment_link(attachments_table, email_row: dict) -> tuple[Any, Any]:
    if attachments_table is None or not email_row:
        return None, None
    if ATTACHMENT_EMAIL_LINK_COLUMN in attachments_table.c:
        value = email_row.get(ATTACHMENT_EMAIL_LINK_COLUMN)
        if value not in (None, ""):
            return attachments_table.c[ATTACHMENT_EMAIL_LINK_COLUMN], value
    return None, None


def _build_attachment_count_statement(
    attachments_table,
    emails_table,
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
):
    attachment_clauses = _build_filters(
        attachments_table,
        query,
        filters,
        date_from=date_from,
        date_to=date_to,
    )
    attach_col, email_col = _find_attachment_email_link(attachments_table, emails_table)
    if attach_col is None or email_col is None:
        stmt = select(func.count()).select_from(attachments_table)
        if attachment_clauses:
            stmt = stmt.where(and_(*attachment_clauses))
        return stmt

    email_clauses = _build_filters(
        emails_table,
        query,
        filters,
        date_from=date_from,
        date_to=date_to,
    )
    stmt = select(func.count()).select_from(
        attachments_table.join(emails_table, attach_col == email_col)
    )
    if attachment_clauses:
        stmt = stmt.where(and_(*attachment_clauses))
    if email_clauses:
        stmt = stmt.where(and_(*email_clauses))
    return stmt


def _run_scalar(statement) -> int:
    engine = get_email_engine()
    try:
        with engine.begin() as conn:
            value = conn.execute(statement).scalar()
            return int(value or 0)
    except Exception:
        return 0


def _get_attachments_for_email(attachments_table, email_row: dict) -> tuple[list[dict], bool]:
    col, value = _resolve_attachment_link(attachments_table, email_row)
    if col is None or value is None:
        return [], False
    statement = select(attachments_table).where(col == value)
    order_col = first_column(attachments_table, ["id", "attachment_id", "created_at", "uploaded_at"])
    if order_col is not None:
        statement = statement.order_by(order_col.desc())
    return run_email_query(statement), True


def _strip_pdf_extension(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(?i)\.pdf\b", "", text)


def _call_llm(messages: list[dict], temperature: float = 0) -> str | None:
    clients = [
        ("groq", groq_client, GROQ_MODEL),
        ("openai", oai_client, OAI_MODEL),
    ]
    for label, client, model in clients:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return (res.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[EMAIL_REPORT LLM] {label} failed: {exc}")
    return None


def _trim_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _select_important_emails_with_llm(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    candidates = []
    for idx, row in enumerate(rows, start=1):
        subject = _trim_text(row.get("subject"))
        summary = _trim_text(row.get("ai_summary"), limit=400)
        candidates.append(
            {
                "ref": f"E{idx}",
                "subject": subject or "-",
                "ai_summary": summary or "-",
            }
        )

    system_prompt = (
        "You select the most important emails from a list for a report. "
        "Use ONLY subject and ai_summary. "
        "Choose a reasonable number (not fixed) and return ONLY JSON with this schema:\n"
        "{\"important_refs\": [\"E1\", \"E2\", ...]}\n"
        "Return an empty list if none are important."
    )
    payload = json.dumps({"candidates": candidates}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    raw = _call_llm(messages)
    data = _extract_json(raw or "")
    if not isinstance(data, dict):
        return []
    refs = data.get("important_refs")
    if not isinstance(refs, list):
        return []

    ref_to_row = {f"E{idx}": row for idx, row in enumerate(rows, start=1)}
    selected = []
    seen = set()
    for ref in refs:
        key = str(ref).strip()
        if not key or key in seen:
            continue
        row = ref_to_row.get(key)
        if row is not None:
            selected.append(row)
            seen.add(key)
    return selected


def _get_important_email_rows(
    table,
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
    max_candidates: int = IMPORTANT_EMAILS_MAX_CANDIDATES,
) -> list[dict]:
    candidates = _search_rows(
        table,
        query,
        filters,
        date_from=date_from,
        date_to=date_to,
        limit=max_candidates,
    )
    return _select_important_emails_with_llm(candidates)


def _collect_contact_emails(
    table,
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
) -> list[str]:
    sender_col = _find_filter_column(table, "sender")
    recipient_col = _find_filter_column(table, "recipient")
    if sender_col is None and recipient_col is None:
        return []

    cols = []
    if sender_col is not None:
        cols.append(sender_col.label("sender_value"))
    if recipient_col is not None:
        cols.append(recipient_col.label("recipient_value"))

    clauses = _build_filters(table, query, filters, date_from=date_from, date_to=date_to)
    statement = select(*cols).select_from(table)
    if clauses:
        statement = statement.where(and_(*clauses))

    rows = run_email_query(statement)
    emails = []
    for row in rows:
        if sender_col is not None:
            emails.extend(_extract_emails(row.get("sender_value")))
        if recipient_col is not None:
            emails.extend(_extract_emails(row.get("recipient_value")))

    return _normalize_contact_list(emails)


def _generate_llm_insights(
    important_emails: list[dict],
    filters_summary: str,
    customer_label: str,
    date_range_label: str,
) -> dict:
    if not important_emails:
        return {}

    payload = {
        "customer": customer_label,
        "date_range": date_range_label,
        "filters": filters_summary,
        "emails": [
            {
                "subject": _strip_pdf_extension(_trim_text(row.get("subject"), limit=200)),
                "ai_summary": _strip_pdf_extension(_trim_text(row.get("ai_summary"), limit=500)),
                "sender": row.get("sender_email") or row.get("sender") or "",
                "recipients": row.get("recipient_emails") or row.get("to_email") or "",
                "date": row.get("received_date")
                or row.get("received_at")
                or row.get("sent_at")
                or "",
            }
            for row in important_emails
        ],
    }

    system_prompt = (
        "You are creating a structured report insight from email summaries. "
        "Use ONLY the provided data. Do NOT invent names, dates, owners, or facts. "
        "If a field is unknown, use \"UNKNOWN\" or leave it empty. "
        "Remove any .pdf extension from filenames. "
        "Return ONLY JSON with this schema:\n"
        "{"
        "\"risks\": [\"...\"],"
        "\"opportunities\": [\"...\"],"
    "\"conclusion\": \"...\""
        "}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw = _call_llm(messages)
    data = _extract_json(raw or "")
    return data if isinstance(data, dict) else {}


def _format_report(
    report: dict,
    filters_summary: str,
    important_emails: list[dict],
    contacts: list[str] | None = None,
) -> str:
    total_emails = report.get("total_emails", 0)
    total_attachments = report.get("total_attachments", 0)
    contacts_list = contacts or []
    if not contacts_list:
        senders = report.get("emails_by_sender", [])
        contacts_list = [row.get("sender") for row in senders if row.get("sender")]
    avocarbon_contacts, customer_contacts = _split_contact_groups(
        contacts_list,
        ["avocarbon"],
    )

    date_range = "All time"
    date_from = report.get("filters", {}).get("date_from")
    date_to = report.get("filters", {}).get("date_to")
    if date_from and date_to:
        date_range = f"{date_from} to {date_to}"
    elif date_from:
        date_range = str(date_from)

    customer_label = "UNKNOWN"
    fields = report.get("filters", {}).get("fields", {})
    if isinstance(fields, dict):
        customer_label = fields.get("client") or fields.get("domain") or "UNKNOWN"
        if isinstance(customer_label, str) and customer_label.startswith("@"):
            customer_label = customer_label[1:]

    insights = _generate_llm_insights(
        important_emails,
        filters_summary=filters_summary,
        customer_label=customer_label,
        date_range_label=date_range,
    )

    lines: list[str] = []
    title = (
        f"{customer_label} - Email Activity Review"
        if customer_label != "UNKNOWN"
        else "Email Activity Review"
    )

    lines.append("REPORT")
    lines.append("INTERNAL EMAIL REPORT")
    lines.append(title)
    lines.append(f"Date range: {date_range}")
    lines.append("")

    lines.append("CONTEXT")
    lines.append(f"Customer: {customer_label}")
    lines.append("Purpose: Review of email activity, key topics, and follow-ups")
    lines.append(f"Totals: Emails {total_emails} | Attachments {total_attachments}")
    lines.append("")

    lines.append("CONTACTS")
    lines.append(f"AVO: {_format_contact_items(avocarbon_contacts)}")
    lines.append(f"Customer: {_format_contact_items(customer_contacts)}")
    lines.append("")

    lines.append("IMPORTANT EMAILS")
    if important_emails:
        sliced = important_emails[:IMPORTANT_EMAILS_MAX_OUTPUT]
        for idx, row in enumerate(sliced, start=1):
            subject = _strip_pdf_extension(row.get("subject") or "UNKNOWN")
            summary = _strip_pdf_extension(row.get("ai_summary") or "UNKNOWN")
            sender = row.get("sender_email") or row.get("sender") or "UNKNOWN"
            recipients = row.get("recipient_emails") or row.get("to_email") or "UNKNOWN"
            date_value = row.get("received_date") or row.get("received_at") or row.get("sent_at") or "UNKNOWN"
            lines.append(f"{idx}. {subject}")
            lines.append(f"From: {sender} | To: {recipients} | Date: {date_value}")
            lines.append(f"Summary: {summary}")
            lines.append("-")
        if lines and lines[-1] == "-":
            lines.pop()
    else:
        lines.append("No important emails identified.")
    lines.append("")

    lines.append("RISKS")
    risks = insights.get("risks") if isinstance(insights, dict) else None
    if isinstance(risks, list) and risks:
        lines.extend([f"- {_strip_pdf_extension(str(risk))}" for risk in risks])
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("OPPORTUNITIES")
    opportunities = insights.get("opportunities") if isinstance(insights, dict) else None
    if isinstance(opportunities, list) and opportunities:
        lines.extend([f"- {_strip_pdf_extension(str(opp))}" for opp in opportunities])
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("CONCLUSION")
    conclusion = insights.get("conclusion") if isinstance(insights, dict) else None
    lines.append(_strip_pdf_extension(str(conclusion)) if conclusion else "UNKNOWN")

    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
        raw = raw.replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _parse_date(value: Any):
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_date_tokens(text: str) -> str:
    if not text:
        return text

    def repl(match):
        day, month, year = match.group(1), match.group(2), match.group(3)
        try:
            parsed = datetime(int(year), int(month), int(day)).date()
        except ValueError:
            return match.group(0)
        return parsed.isoformat()

    return re.sub(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", repl, text)


def _normalize_tables(value: Any):
    if not value:
        return None
    if isinstance(value, str):
        tokens = re.split(r"[,\s]+", value.lower())
        value = [t for t in tokens if t]
    if not isinstance(value, list):
        return None
    items = {str(v).strip().lower() for v in value if v}
    selected = [t for t in ["emails", "attachments"] if t in items]
    return selected if selected else None


def _normalize_filters(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key in FIELD_FILTER_CANDIDATES:
        raw = value.get(key)
        if raw is None:
            continue
        text = " ".join(str(raw).strip().split())
        if text:
            cleaned[key] = text
    return cleaned


def _infer_domain_filter(filters: dict) -> dict:
    if not filters:
        return filters
    domain_value = None
    if filters.get("domain"):
        domain_value = filters.get("domain") or ""
    elif filters.get("client"):
        domain_value = filters.get("client") or ""
    if domain_value is not None:
        for key in ("recipient", "sender"):
            value = filters.get(key)
            if value and "@" not in value and value not in domain_value:
                filters.pop(key, None)
        return filters
    for key in ("recipient", "sender"):
        value = filters.get(key)
        if value and "@" not in value:
            filters["domain"] = f"@{value}"
            filters.pop(key, None)
            break
    return filters


def _normalize_plan(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    intent = raw.get("intent") or raw.get("action")
    intent = (intent or "").strip().lower()
    if intent not in {"search", "report", "both", "clarify"}:
        intent = "search"

    query = raw.get("query")
    query_text = " ".join(str(query).strip().split()) if query else None
    if query_text == "":
        query_text = None

    date_from = _parse_date(raw.get("date_from"))
    date_to = _parse_date(raw.get("date_to"))
    if date_from and not date_to:
        date_to = date_from
    if date_to and not date_from:
        date_from = date_to

    tables = _normalize_tables(raw.get("tables"))
    filters = _normalize_filters(raw.get("filters"))
    filters = _infer_domain_filter(filters)

    limit = None
    language = _normalize_language(raw.get("language") or raw.get("lang"))

    needs_clarification = bool(raw.get("needs_clarification"))
    clarification_question = raw.get("clarification_question")
    clarification_question = (
        " ".join(str(clarification_question).strip().split())
        if clarification_question
        else None
    )

    if intent == "clarify":
        needs_clarification = True

    if intent in {"search", "both"} and not (
        query_text or filters or date_from or date_to
    ):
        needs_clarification = True

    return {
        "intent": intent,
        "language": language,
        "query": query_text,
        "date_from": date_from,
        "date_to": date_to,
        "tables": tables,
        "filters": filters,
        "limit": limit,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }


def _parse_request_with_llm(message: str) -> dict | None:
    today = datetime.now().date().isoformat()
    system_prompt = LLM_SYSTEM_PROMPT.format(today=today)
    normalized_message = _normalize_date_tokens(message)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": normalized_message},
    ]
    raw = _call_llm(messages)
    data = _extract_json(raw or "")
    return _normalize_plan(data)


def _find_filter_column(table, field_key: str):
    candidates = FIELD_FILTER_CANDIDATES.get(field_key, [])
    return first_column(table, candidates)


def _build_filters(
    table,
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
):
    clauses = []
    if query:
        text_columns = [column for column in table.c if is_text_column(column)]
        if text_columns:
            pattern = f"%{query}%"
            clauses.append(or_(*[column.ilike(pattern) for column in text_columns]))

    for field_key, value in (filters or {}).items():
        col = _find_filter_column(table, field_key)
        if col is not None and value:
            clauses.append(col.ilike(f"%{value}%"))

    date_filter = build_date_filter(table, date_from=date_from, date_to=date_to)
    if date_filter is not None:
        clauses.append(date_filter)

    return clauses


def _default_order_columns(table):
    order_cols = []
    date_col = first_column(
        table,
        ["received_date", "received_at", "sent_at", "created_at", "date", "timestamp"],
    )
    if date_col is not None:
        if is_date_column(date_col):
            order_cols.append(date_col.desc())
        else:
            order_cols.append(func.substring(date_col, 1, 10).desc())

    id_col = first_column(table, ["id", "message_id"])
    if id_col is not None:
        order_cols.append(id_col.desc())

    return order_cols


def _search_rows(
    table,
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
    limit: int | None = None,
):
    clauses = _build_filters(table, query, filters, date_from=date_from, date_to=date_to)
    statement = select(table)
    if clauses:
        statement = statement.where(and_(*clauses))
    order_cols = _default_order_columns(table)
    if order_cols:
        statement = statement.order_by(*order_cols)
    if limit is not None:
        statement = statement.limit(limit)
    return run_email_query(statement)


def _generate_filtered_report(
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
    include_emails: bool = True,
    include_attachments: bool = True,
) -> dict:
    tables = get_email_tables()
    engine = get_email_engine()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": list(tables.keys()),
        "filters": {
            "query": query,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "fields": filters or {},
            "include_emails": include_emails,
            "include_attachments": include_attachments,
        },
    }

    try:
        with engine.begin() as conn:
            if include_emails and "emails" in tables:
                emails = tables["emails"]
                email_filters = _build_filters(
                    emails, query, filters, date_from=date_from, date_to=date_to
                )
                where_clause = and_(*email_filters) if email_filters else None

                stmt = select(func.count()).select_from(emails)
                if where_clause is not None:
                    stmt = stmt.where(where_clause)
                report["total_emails"] = conn.execute(stmt).scalar() or 0

                sender_col = first_column(
                    emails,
                    ["sender_email", "sender", "from_email", "from_address", "from"],
                )
                if sender_col is not None:
                    stmt = select(
                        sender_col.label("sender"),
                        func.count().label("count"),
                    ).select_from(emails)
                    if where_clause is not None:
                        stmt = stmt.where(where_clause)
                    stmt = (
                        stmt.where(sender_col.isnot(None))
                        .group_by(sender_col)
                        .order_by(func.count().desc())
                    )
                    rows = conn.execute(stmt).fetchall()
                    report["emails_by_sender"] = [
                        {"sender": row.sender, "count": row.count} for row in rows
                    ]

                date_col = first_column(
                    emails,
                    ["received_date", "received_at", "sent_at", "created_at", "date", "timestamp"],
                )
                if date_col is not None:
                    if is_date_column(date_col):
                        date_key = func.date(date_col)
                    else:
                        date_key = func.substring(date_col, 1, 10)

                    stmt = select(
                        date_key.label("date"),
                        func.count().label("count"),
                    ).select_from(emails)
                    if where_clause is not None:
                        stmt = stmt.where(where_clause)
                    stmt = (
                        stmt.where(date_col.isnot(None))
                        .group_by(date_key)
                        .order_by(date_key.desc())
                        .limit(30)
                    )
                    rows = conn.execute(stmt).fetchall()
                    report["emails_by_date"] = [
                        {
                            "date": row.date.isoformat() if hasattr(row.date, "isoformat") else row.date,
                            "count": row.count,
                        }
                        for row in rows
                    ]

            if include_attachments and "attachments" in tables:
                attachments = tables["attachments"]
                emails_table = tables.get("emails")
                stmt = _build_attachment_count_statement(
                    attachments,
                    emails_table,
                    query,
                    filters,
                    date_from=date_from,
                    date_to=date_to,
                )
                report["total_attachments"] = conn.execute(stmt).scalar() or 0

    except Exception:
        raise RuntimeError("Database error")

    return report


def _format_filters_summary(
    query: str | None,
    filters: dict,
    date_from=None,
    date_to=None,
    include_emails: bool = True,
    include_attachments: bool = True,
) -> str:
    parts = []
    if query:
        parts.append(f'text="{query}"')
    for key, value in (filters or {}).items():
        if value:
            parts.append(f"{key}='{value}'")
    if date_from and date_to:
        parts.append(f"dates: {date_from} to {date_to}")
    elif date_from:
        parts.append(f"date: {date_from}")
    if include_emails and include_attachments:
        parts.append("tables: emails + attachments")
    return " | ".join(parts) if parts else "no filters"


def _wants_body(message: str) -> bool:
    if not message:
        return False
    text = message.lower()
    patterns = [
        r"\bbody\b",
        r"\bfull body\b",
        r"\bemail body\b",
        r"\bcorps\b",
        r"\bcorps du mail\b",
        r"\bcontenu\b",
        r"\bcontenu du mail\b",
        r"\btexte complet\b",
        r"\bmessage complet\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _wants_attachments(message: str) -> bool:
    if not message:
        return False
    text = message.lower()
    patterns = [
        r"\battachment\b",
        r"\battachments\b",
        r"\battachement\b",
        r"\battachements\b",
        r"\battached file\b",
        r"\battached files\b",
        r"\bpi[eè]ce\s+jointe\b",
        r"\bpi[eè]ces\s+jointes\b",
        r"\bfichier\s+joint\b",
        r"\bfichiers\s+joints\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _message_has_explicit_date(message: str) -> bool:
    if not message:
        return False
    text = message.lower()

    numeric_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ]
    if any(re.search(p, text) for p in numeric_patterns):
        return True

    relative_patterns = [
        r"\btoday\b",
        r"\byesterday\b",
        r"\btomorrow\b",
        r"\blast\s+week\b",
        r"\blast\s+month\b",
        r"\blast\s+year\b",
        r"\bthis\s+week\b",
        r"\bthis\s+month\b",
        r"\bthis\s+year\b",
        r"\bpast\s+\d+\s+days?\b",
        r"\bpast\s+\d+\s+weeks?\b",
        r"\bpast\s+\d+\s+months?\b",
        r"\bpast\s+\d+\s+years?\b",
        r"\baujourd\b",
        r"\bhier\b",
        r"\bdemain\b",
        r"\bsemaine\s+derniere\b",
        r"\bmois\s+dernier\b",
        r"\bannee\s+derniere\b",
        r"\bcette\s+semaine\b",
        r"\bce\s+mois\b",
        r"\bcette\s+annee\b",
        r"\bderniers?\s+\d+\s+jours?\b",
        r"\bderniers?\s+\d+\s+semaines?\b",
        r"\bderniers?\s+\d+\s+mois\b",
        r"\bderniers?\s+\d+\s+ans\b",
    ]
    if any(re.search(p, text) for p in relative_patterns):
        return True

    month_patterns = [
        r"\bjan(uary)?\b",
        r"\bfeb(ruary)?\b",
        r"\bmar(ch)?\b",
        r"\bapr(il)?\b",
        r"\bmay\b",
        r"\bjun(e)?\b",
        r"\bjul(y)?\b",
        r"\baug(ust)?\b",
        r"\bsep(tember)?\b",
        r"\boct(ober)?\b",
        r"\bnov(ember)?\b",
        r"\bdec(ember)?\b",
        r"\bjanv\b",
        r"\bfev\b",
        r"\bfevr\b",
        r"\bfevrier\b",
        r"\bmars\b",
        r"\bavr\b",
        r"\bavril\b",
        r"\bmai\b",
        r"\bjuin\b",
        r"\bjuil\b",
        r"\bjuillet\b",
        r"\baout\b",
        r"\bsept\b",
        r"\bseptembre\b",
        r"\boct\b",
        r"\boctobre\b",
        r"\bnov\b",
        r"\bnovembre\b",
        r"\bdec\b",
        r"\bdecembre\b",
    ]
    if any(re.search(p, text) for p in month_patterns):
        return True

    return False


def _wants_full_report(message: str) -> bool:
    if not message:
        return False
    text = message.lower()
    patterns = [
        r"\bfull\s+report\b",
        r"\bcomplete\s+report\b",
        r"\bentire\s+report\b",
        r"\boverall\s+report\b",
        r"\bglobal\s+report\b",
        r"\bfull\s+summary\b",
        r"\bcomplete\s+summary\b",
        r"\ball\s+emails?\b",
        r"\ball\s+messages\b",
        r"\ball\s+mail\b",
        r"\breport\s+complet\b",
        r"\brapport\s+complet\b",
        r"\brapport\s+integral\b",
        r"\brapport\s+global\b",
        r"\brapport\s+total\b",
        r"\brapport\s+entier\b",
        r"\brapport\s+general\b",
        r"\btous\s+les\s+emails\b",
        r"\btoutes\s+les\s+emails\b",
        r"\btous\s+les\s+messages\b",
        r"\btoutes\s+les\s+messages\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _should_clear_dates_for_full_report(message: str) -> bool:
    if not _wants_full_report(message):
        return False
    return not _message_has_explicit_date(message)


def _format_email_details(
    rows,
    include_body: bool = False,
    attachment_groups: list[tuple[list[dict], bool]] | None = None,
    limit: int | None = None,
) -> str:
    if not rows:
        return "None."

    lines = []
    sliced_rows = rows if limit is None else rows[:limit]

    for idx, row in enumerate(sliced_rows, start=1):
        subject = _wrap_value(row.get("subject") or "-")
        sender = _wrap_value(row.get("sender_email") or row.get("sender") or "-")
        recipients = _wrap_value(row.get("recipient_emails") or row.get("to_email") or "-")
        date_value = _wrap_value(
            row.get("received_date") or row.get("received_at") or row.get("sent_at") or "-"
        )
        summary = row.get("ai_summary")

        parts = [
            f"Email {idx}:",
            f"Subject: {subject}",
            f"Sender: {sender}",
            f"Recipients: {recipients}",
            f"Date: {date_value}",
            f"AI Summary: {summary or '-'}",
        ]

        if include_body:
            body_raw = row.get("body_text") or "-"
            body = _wrap_value(_normalize_links(body_raw))
            parts.append(f"Body: {body}")
        elif not summary:
            body_raw = row.get("body_text")
            if body_raw:
                body = _wrap_value(_normalize_links(body_raw))
                parts.append(f"Body: {body}")

        if attachment_groups is not None and idx - 1 < len(attachment_groups):
            attachments, has_link = attachment_groups[idx - 1]
            if not has_link:
                parts.append("Attachments: Not available.")
            elif attachments:
                parts.append("Attachments:")
                parts.append(build_rows_block(attachments, ATTACHMENT_SNIPPET_KEYS, limit=20))
            else:
                parts.append("Attachments: None.")

        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def _format_single_email(row: dict, index: int, include_body: bool) -> str:
    subject = _wrap_value(row.get("subject") or "-")
    sender = _wrap_value(row.get("sender_email") or row.get("sender") or "-")
    recipients = _wrap_value(row.get("recipient_emails") or row.get("to_email") or "-")
    date_value = _wrap_value(
        row.get("received_date") or row.get("received_at") or row.get("sent_at") or "-"
    )
    summary = row.get("ai_summary")
    parts = [
        f"Email {index}:",
        f"Subject: {subject}",
        f"Sender: {sender}",
        f"Recipients: {recipients}",
        f"Date: {date_value}",
        f"AI Summary: {summary or '-'}",
    ]
    if include_body:
        body_raw = row.get("body_text") or "-"
        body = _wrap_value(_normalize_links(body_raw))
        parts.append(f"Body: {body}")
    elif not summary:
        body_raw = row.get("body_text")
        if body_raw:
            body = _wrap_value(_normalize_links(body_raw))
            parts.append(f"Body: {body}")
    return "\n".join(parts)


def _extract_email_reference(message: str) -> dict | None:
    if not message:
        return None
    text = message.lower()
    match = re.search(r"\b(?:email|e-mail|mail|message)\s*#?\s*(\d+)\b", text)
    if not match:
        return None
    try:
        index = int(match.group(1))
    except ValueError:
        return None
    if index < 1:
        return None
    wants_body = _wants_body(message)
    wants_attachments = _wants_attachments(message)
    return {
        "index": index,
        "wants_body": wants_body,
        "wants_attachments": wants_attachments,
    }


def _get_previous_user_message(history: list[dict], current_message: str) -> str | None:
    if not history:
        return None
    current = (current_message or "").strip()
    skipped_current = False
    for item in reversed(history):
        if item.get("role") != "user":
            continue
        content = (item.get("content") or "").strip()
        if not skipped_current and content == current:
            skipped_current = True
            continue
        if content:
            return content
    return None


def run(message: str, session: dict) -> str:
    query = (message or "").strip()
    if not query:
        return "Please enter a query to search emails."

    starter_kind = _detect_starter_message(query)
    if starter_kind:
        language = _detect_language_with_llm(query)
        return _translate_response(_starter_prompt(starter_kind), language)

    history = session.get("history", []) if isinstance(session, dict) else []
    email_ref = _extract_email_reference(query)
    if email_ref:
        language = _detect_language_with_llm(query)
        previous_message = _get_previous_user_message(history, query)
        if not previous_message:
            return _translate_response(
                "I don't have the previous email list. Please ask again with your filters.",
                language,
            )
        previous_plan = _parse_request_with_llm(previous_message)
        if not previous_plan:
            return _translate_response(
                "I couldn't retrieve the previous search. Please ask again with your filters.",
                language,
            )
        if previous_plan.get("needs_clarification"):
            return _translate_response(
                "Please repeat your previous request with clearer filters.",
                language,
            )
        selected_tables = previous_plan["tables"]
        include_emails = (
            "emails" in previous_plan["tables"]
            if selected_tables
            else True
        )
        if not include_emails:
            return _translate_response(
                "The previous request did not target emails. Please ask again with email filters.",
                language,
            )
        tables = get_email_tables()
        emails_table = tables.get("emails")
        if emails_table is None:
            return _translate_response(
                "The emails table is not available in the database.",
                language,
            )
        emails = _search_rows(
            emails_table,
            previous_plan["query"],
            previous_plan["filters"],
            date_from=previous_plan["date_from"],
            date_to=previous_plan["date_to"],
            limit=email_ref["index"],
        )
        if len(emails) < email_ref["index"]:
            return _translate_response(
                f"Email {email_ref['index']} is not available for the previous search.",
                language,
            )
        email_row = emails[email_ref["index"] - 1]
        response = _format_single_email(
            email_row,
            email_ref["index"],
            include_body=email_ref["wants_body"],
        )
        if email_ref["wants_body"] or email_ref.get("wants_attachments"):
            attachments_table = tables.get("attachments")
            attachments_block = ""
            if attachments_table is not None:
                attachments, has_link = _get_attachments_for_email(attachments_table, email_row)
                if not has_link:
                    attachments_block = "Attachments: Not available."
                elif attachments:
                    attachments_block = "Attachments:\n" + build_rows_block(
                        attachments,
                        ATTACHMENT_SNIPPET_KEYS,
                        limit=20,
                    )
                else:
                    attachments_block = "Attachments: None."
            else:
                attachments_block = "Attachments: Not available."
            if attachments_block:
                response = f"{response}\n\n{attachments_block}"
        return _translate_response(response, language)

    tables = get_email_tables()
    if "emails" not in tables and "attachments" not in tables:
        language = _detect_language_with_llm(query)
        return _translate_response(
            "The emails/attachments tables are not available in the database.",
            language,
        )

    plan = _parse_request_with_llm(query)
    language = _normalize_language(plan.get("language")) if plan else _detect_language_with_llm(query)
    if not plan:
        return _translate_response(
            "I couldn't understand your request. Please rephrase and specify what you want to search or the type of report.",
            language,
        )

    if plan["needs_clarification"]:
        response = plan["clarification_question"] or (
            "Please clarify what you're looking for (subject, sender, date, extension, etc.) "
            "or the report type."
        )
        return _translate_response(response, language)

    if _should_clear_dates_for_full_report(query):
        plan["date_from"] = None
        plan["date_to"] = None

    intent = plan["intent"]
    want_search = intent in {"search", "both"}
    want_report = intent in {"report", "both"}
    if want_report:
        want_search = False

    has_any_filters = bool(plan.get("query")) or _has_filter_values(plan.get("filters")) or plan.get("date_from") or plan.get("date_to")
    if want_report and not has_any_filters and not _wants_full_report(query):
        return _translate_response(_starter_prompt("report"), language)
    if want_search and not has_any_filters:
        return _translate_response(_starter_prompt("search"), language)

    selected_tables = plan["tables"]
    include_emails = "emails" in tables if not selected_tables else "emails" in selected_tables
    include_attachments = "attachments" in tables if not selected_tables else "attachments" in selected_tables
    include_emails = include_emails and "emails" in tables
    include_attachments = include_attachments and "attachments" in tables

    if not include_emails and not include_attachments:
        return _translate_response("The requested tables are not available.", language)

    date_from = plan["date_from"]
    date_to = plan["date_to"]
    query_text = plan["query"]
    filters = plan["filters"]
    limit = None

    lines = []
    filters_summary = _format_filters_summary(
        query_text,
        filters,
        date_from=date_from,
        date_to=date_to,
        include_emails=include_emails,
        include_attachments=include_attachments,
    )

    include_body = _wants_body(query)
    include_attachment_details = _wants_attachments(query)

    if want_search:
        emails = []
        attachments = []
        attachments_count = 0
        attachment_groups = None
        if include_emails:
            emails = _search_rows(
                tables["emails"],
                query_text,
                filters,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
        if include_attachments:
            attachments_table = tables["attachments"]
            if include_emails and "emails" in tables:
                stmt = _build_attachment_count_statement(
                    attachments_table,
                    tables.get("emails"),
                    query_text,
                    filters,
                    date_from=date_from,
                    date_to=date_to,
                )
                attachments_count = _run_scalar(stmt)
            else:
                attachments = _search_rows(
                    attachments_table,
                    query_text,
                    filters,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                )
                attachments_count = len(attachments)

            if include_emails and emails and include_attachment_details:
                attachment_groups = []
                for email_row in emails:
                    email_attachments, has_link = _get_attachments_for_email(attachments_table, email_row)
                    attachment_groups.append((email_attachments, has_link))

        lines.append(f"Results: {len(emails)} emails, {attachments_count} attachments.")
        lines.append(f"Filters: {filters_summary}")
        if emails:
            lines.append("Email details:")
            lines.append(
                _format_email_details(
                    emails,
                    include_body=include_body,
                    attachment_groups=attachment_groups,
                    limit=None,
                )
            )
        if attachments and not include_emails:
            lines.append("Attachment examples:")
            lines.append(build_rows_block(attachments, ATTACHMENT_SNIPPET_KEYS))

    if want_report:
        try:
            report = _generate_filtered_report(
                query=query_text,
                filters=filters,
                date_from=date_from,
                date_to=date_to,
                include_emails=include_emails,
                include_attachments=include_attachments,
            )
            important_emails = []
            contacts = []
            if include_emails:
                important_emails = _get_important_email_rows(
                    tables["emails"],
                    query_text,
                    filters,
                    date_from=date_from,
                    date_to=date_to,
                )
                contacts = _collect_contact_emails(
                    tables["emails"],
                    query_text,
                    filters,
                    date_from=date_from,
                    date_to=date_to,
                )
            lines.append(
                _format_report(
                    report,
                    filters_summary=filters_summary,
                    important_emails=important_emails,
                    contacts=contacts,
                )
            )
        except Exception:
            lines.append("Unable to generate the report right now.")

    return _translate_response("\n".join(lines), language)


def run_stream(message: str, session: dict):
    yield run(message, session)

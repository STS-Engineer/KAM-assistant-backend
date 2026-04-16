from datetime import datetime, timezone, timedelta
import os
import json
import re
import secrets
import hashlib
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from urllib.parse import quote

from fastapi import FastAPI, Request, Depends, HTTPException, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, select

from auth import (
    COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    SECRET_KEY,
    decode_token,
    create_access_token,
    generate_refresh_token,
    hash_refresh_token,
    hash_password,
    verify_password,
)
from bots import BOTS
from openai_client import client as oai_client, MODEL as OAI_MODEL
from db import (
    ChatbotUser,
    PasswordResetToken,
    RefreshToken,
    get_db,
    get_chatbot_session,
    Conversation,
    Message,
)
from email_data import (
    generate_email_report,
    get_email_table,
    get_email_tables,
    list_table_with_filters,
    run_email_query,
    search_table as search_email_table,
    search_table_with_filters,
)
from azure_attachments import (
    AttachmentAmbiguousError,
    AttachmentNotFoundError,
    AttachmentStorageError,
    azure_storage_enabled,
    open_attachment_download,
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ==============================
# CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utcnow():
    return datetime.now(timezone.utc)


# ==============================
# Password reset config
# ==============================

RESET_TOKEN_TTL_HOURS = int(os.getenv("RESET_TOKEN_TTL_HOURS", "1"))

REFRESH_TOKEN_TTL_DAYS = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "7"))
REFRESH_COOKIE_SECURE = (os.getenv("REFRESH_COOKIE_SECURE") or "true").strip().lower() in {
    "1", "true", "yes"
}
REFRESH_COOKIE_SAMESITE = (os.getenv("REFRESH_COOKIE_SAMESITE") or "none").strip().lower()
REFRESH_COOKIE_MAX_AGE = REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60

FRONTEND_RESET_URL = (
    os.getenv("RESET_PASSWORD_URL") or "http://localhost:3000/reset-password"
).strip()

SMTP_HOST = os.getenv("SMTP_HOST", "avocarbon-com.mail.protection.outlook.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
EMAIL_FROM = os.getenv("EMAIL_FROM", "administration.STS@avocarbon.com").strip()
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Administration STS").strip()
SMTP_USE_TLS = (os.getenv("SMTP_USE_TLS") or "false").strip().lower() in {"1", "true", "yes"}


# ==============================
# Token helpers
# ==============================

def hash_reset_token(token: str) -> str:
    secret = str(SECRET_KEY or "")
    raw = f"{token}:{secret}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_reset_link(token: str, email: str | None = None) -> str:
    base = (FRONTEND_RESET_URL or "").strip()
    if not base:
        raise ValueError("RESET_PASSWORD_URL is empty")

    joiner = "&" if "?" in base else "?"
    if email:
        return f"{base}{joiner}token={quote(token)}&email={quote(email)}"
    return f"{base}{joiner}token={quote(token)}"


def set_refresh_cookie(resp: Response, token: str, expires_at: datetime):
    resp.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=REFRESH_COOKIE_SECURE,
        samesite=REFRESH_COOKIE_SAMESITE,
        max_age=REFRESH_COOKIE_MAX_AGE,
        expires=expires_at,
        path="/",
    )


def clear_refresh_cookie(resp: Response):
    resp.delete_cookie(REFRESH_COOKIE_NAME, path="/")


def create_refresh_token_record(db: Session, email: str, user_id) -> tuple[str, datetime]:
    refresh_token = generate_refresh_token()
    expires_at = utcnow() + timedelta(days=REFRESH_TOKEN_TTL_DAYS)
    token_hash = hash_refresh_token(refresh_token)
    db.add(
        RefreshToken(
            user_id=user_id,
            email=email,
            token_hash=token_hash,
            expires_at=expires_at,
        )
    )
    db.commit()
    return refresh_token, expires_at


# ==============================
# HTML Email Template
# ==============================

def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_reset_html_body(reset_link: str, expires_hours: int, user_email: str | None = None) -> str:
    reset_link_escaped = _escape_html(reset_link)
    email_escaped = _escape_html(user_email or "-")
    received_on = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Password Reset</title>
</head>
<body style="margin:0;padding:0;background:#f3f5f7;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f3f5f7;">
    <tr>
      <td align="center" style="padding:32px 16px;">

        <table role="presentation" width="680" cellspacing="0" cellpadding="0" border="0"
               style="max-width:680px;width:100%;background:#ffffff;border-radius:16px;border:1px solid #e9edf3;">
          <tr>
            <td style="padding:28px 24px 18px 24px;">

              <div style="font-family:Arial,Helvetica,sans-serif;font-size:24px;font-weight:700;color:#111827;text-align:center;">
                AVO Emails Password Reset
              </div>

              <div style="height:18px;"></div>

              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
                     style="border:1px solid #e6eaf0;border-radius:12px;overflow:hidden;">
                <tr>
                  <td width="8" style="background:#2563eb;">&nbsp;</td>

                  <td style="padding:18px 18px 10px 18px;background:#ffffff;">

                    <div style="height:16px;"></div>

                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                      <tr>
                        <td style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#111827;font-weight:700;">
                          ❗&nbsp;&nbsp;Action required :
                        </td>
                      </tr>
                      <tr>
                        <td style="padding-top:10px;">
                          <div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#374151;line-height:1.6;
                                      background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
                                      padding:12px 12px;">
                            We received a request to reset your password for <strong>{email_escaped}</strong>.
                            <br><br>
                            Use the button below to set a new password.
                            <br><br>
                            This link expires in <strong>{expires_hours} hour</strong>.
                            <br><br>
                            Request received on: <strong>{received_on}</strong>
                          </div>
                        </td>
                      </tr>
                    </table>

                    <div style="height:14px;"></div>

                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="left">
                      <tr>
                        <td style="border-radius:8px;background:#2563eb;">
                          <a href="{reset_link_escaped}"
                             style="display:inline-block;padding:12px 18px;font-family:Arial,Helvetica,sans-serif;
                                    font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;border-radius:8px;">
                            Reset Password
                          </a>
                        </td>
                      </tr>
                    </table>

                    <div style="height:18px;clear:both;"></div>

                    <div style="font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#6b7280;line-height:1.6;">
                      If you did not request this, you can safely ignore this email.
                    </div>

                  </td>
                </tr>
              </table>

              <div style="font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#9ca3af;text-align:center;margin-top:18px;">
                © 2026 AVO Emails
              </div>

            </td>
          </tr>
        </table>

      </td>
    </tr>
  </table>
</body>
</html>"""


# ==============================
# Send email
# ==============================

def send_reset_email(to_email: str, reset_link: str) -> None:
    subject = "Reset your AVO Emails password"

    text_body = (
        "Hello,\n\n"
        "We received a request to reset your password.\n"
        f"Click this link to reset it:\n{reset_link}\n\n"
        f"This link expires in {RESET_TOKEN_TTL_HOURS} hours.\n\n"
        "If you did not request this, please ignore this email.\n\n"
        "AVO Emails Support"
    )

    html_body = build_reset_html_body(reset_link, RESET_TOKEN_TTL_HOURS, user_email=to_email)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((EMAIL_FROM_NAME, EMAIL_FROM))
    msg["To"] = to_email

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        if SMTP_USE_TLS:
            context = ssl.create_default_context()
            server.starttls(context=context)
        server.sendmail(EMAIL_FROM, [to_email], msg.as_string())


# ==============================
# Chat helpers
# ==============================

def is_meaningful_message(text: str) -> bool:
    t = " ".join((text or "").strip().split())
    if not t:
        return False
    if len(t) < 12:
        return False
    words = t.split(" ")
    if len(words) < 2:
        return False
    alpha_count = sum(1 for c in t if c.isalpha())
    if alpha_count < 6:
        return False
    if len(words) == 1 and t.lower() in {"hi", "hello", "salut", "hey", "yo", "ok", "okay", "test", "bonjour"}:
        return False
    return True


def normalize_title(text: str | None) -> str:
    t = " ".join((text or "").strip().split())
    return t or "New chat"


def generate_title_llm(text: str, max_words: int = 4) -> str | None:
    system = (
        "You are a title generator. Create a short, concise title (maximum "
        f"{max_words} words) that summarizes the user's question or topic.\n\n"
        "IMPORTANT: MUST include the most important nouns from the message.\n\n"
        "DO NOT abstract or paraphrase.\n"
        "DO NOT invent synonyms.\n"
        "No quotes. No punctuation at the end. "
        "Keep the language of the user's message. "
        "NEVER just copy the beginning of their message."
    )
    user = f"User message:\n{text}\n\nTitle:"
    try:
        res = oai_client.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        raw = (res.choices[0].message.content or "").strip()
        raw = raw.strip('"\'').strip()
        raw = raw.rstrip(" .,:;")
        if not raw:
            return None
        words = raw.split()
        if len(words) > max_words:
            raw = " ".join(words[:max_words])
        return raw
    except Exception as exc:
        print(f"[TITLE] LLM title generation failed: {exc}")
        return None


def make_title(first_user_message: str) -> str:
    t = " ".join((first_user_message or "").strip().split())
    if not is_meaningful_message(t):
        return "New chat"

    title = generate_title_llm(t, max_words=4)
    if title:
        return title

    words = t.split()
    return " ".join(words[:4]) if words else "New chat"


def normalize_username(raw: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]", "", (raw or "").strip().lower())
    if not base:
        base = "user"
    return base[:50]


def unique_username(db: Session, base: str) -> str:
    base = normalize_username(base)
    exists = db.query(ChatbotUser.id).filter(ChatbotUser.username == base).first()
    if not exists:
        return base

    suffix = 2
    while True:
        candidate = f"{base}{suffix}"
        if len(candidate) > 100:
            candidate = candidate[: 100 - len(str(suffix))] + str(suffix)
        exists = db.query(ChatbotUser.id).filter(ChatbotUser.username == candidate).first()
        if not exists:
            return candidate
        suffix += 1


def sse_event(data: dict, event: str | None = None) -> str:
    payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    if event:
        payload = f"event: {event}\n" + payload
    return payload


def unique_title(db: Session, email: str, base: str, exclude_id: int | None = None) -> str:
    base = (base or "New chat").strip() or "New chat"

    q = db.query(Conversation.title).filter(
        Conversation.email == email,
        Conversation.is_deleted == False,
    )
    if exclude_id is not None:
        q = q.filter(Conversation.id != exclude_id)

    rows = q.all()
    existing = {r[0] for r in rows if r and r[0]}

    if base not in existing:
        return base

    used_nums = set()
    prefix = base + " ("
    for t in existing:
        if t.startswith(prefix) and t.endswith(")"):
            num = t[len(prefix):-1]
            if num.isdigit():
                used_nums.add(int(num))

    n = 2
    while n in used_nums:
        n += 1
    return f"{base} ({n})"


def list_conversations(db: Session, email: str):
    q = db.query(Conversation).filter(
        Conversation.email == email,
        Conversation.is_deleted == False,
    )
    return (
        q.order_by(func.coalesce(Conversation.updated_at, Conversation.created_at).desc(), Conversation.id.desc())
        .limit(50)
        .all()
    )


def build_history_items(convs):
    items = []
    for c in convs:
        ts = c.updated_at or c.created_at or utcnow()
        items.append(
            {
                "chat_id": c.id,
                "title": normalize_title(c.title),
                "updated_at": ts.isoformat(),
            }
        )
    return items


def get_chatbot_user_id(email: str):
    if not email:
        return None
    chat_db = get_chatbot_session()
    try:
        row = chat_db.query(ChatbotUser.id).filter(ChatbotUser.email == email).first()
        return row[0] if row else None
    except Exception as e:
        print(f"Chatbot user lookup error: {e}")
        return None
    finally:
        chat_db.close()


def create_conversation(db: Session, email: str, user_id=None) -> Conversation:
    now = utcnow()
    conv = Conversation(
        user_id=user_id,
        email=email,
        title="New chat",
        created_at=now,
        updated_at=now,
        is_deleted=False,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


# ==============================
# Payloads
# ==============================

class SignupPayload(BaseModel):
    full_name: str
    email: str
    password: str
    confirm_password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class ForgotPasswordPayload(BaseModel):
    email: str


class ResetPasswordPayload(BaseModel):
    token: str
    password: str
    confirm_password: str
    email: str | None = None


class RenameChatPayload(BaseModel):
    title: str


class EditMessagePayload(BaseModel):
    content: str
    regenerate: bool = False
    bot_mode: str | None = None


# ==============================
# Auth helpers
# ==============================

def get_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def require_user(request: Request) -> str:
    token = get_bearer_token(request) or request.cookies.get(COOKIE_NAME)
    email = decode_token(token) if token else None
    if not email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return email


# ==============================
# Basic routes
# ==============================

@app.get("/")
def root():
    return {"status": "ok"}


# ==============================
# Email data helpers
# ==============================

def email_table_or_404(name: str):
    table = get_email_table(name)
    if table is None:
        raise HTTPException(status_code=404, detail=f"Table not found: {name}")
    return table


def parse_date_param(value: str | None, label: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label}. Use YYYY-MM-DD.")


def serialize_attachment_row(row: dict):
    item = dict(row or {})
    attachment_id = item.get("attachment_id")
    if attachment_id is not None:
        item["download_url"] = f"/api/attachments/{attachment_id}/download"
    item["azure_storage_enabled"] = azure_storage_enabled()
    return item


def build_attachment_stream_response(local_file_path: str | None, file_name: str | None = None):
    if not azure_storage_enabled():
        raise HTTPException(status_code=503, detail="Azure attachment storage is not configured")

    try:
        download = open_attachment_download(local_file_path=local_file_path, file_name=file_name)
    except AttachmentNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment blob not found in Azure storage")
    except AttachmentAmbiguousError:
        raise HTTPException(status_code=409, detail="Multiple Azure blobs match this attachment")
    except AttachmentStorageError:
        raise HTTPException(status_code=502, detail="Azure attachment storage is unavailable")

    headers = dict(download["headers"])
    content_length = download.get("content_length")
    if content_length is not None:
        headers["Content-Length"] = str(content_length)

    return StreamingResponse(
        download["chunks"],
        media_type=download["content_type"],
        headers=headers,
    )


# ==============================
# Email API
# ==============================

@app.get("/api/email-meta")
def email_meta(request: Request):
    require_user(request)
    tables = get_email_tables()
    return {name: [column.name for column in table.c] for name, table in tables.items()}


@app.get("/api/emails")
def list_emails(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    require_user(request)
    date_from_value = parse_date_param(date_from, "date_from")
    date_to_value = parse_date_param(date_to, "date_to")
    table = email_table_or_404("emails")
    items = list_table_with_filters(
        table,
        limit=limit,
        offset=offset,
        date_from=date_from_value,
        date_to=date_to_value,
    )
    return {"items": items, "limit": limit, "offset": offset}


@app.get("/api/attachments")
def list_attachments(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    require_user(request)
    table = email_table_or_404("attachments")
    statement = select(table).limit(limit).offset(offset)
    items = [serialize_attachment_row(row) for row in run_email_query(statement)]
    return {"items": items, "limit": limit, "offset": offset}


@app.get("/api/attachments/download")
def download_attachment_from_path(
    request: Request,
    local_file_path: str = Query(..., min_length=1),
    file_name: str | None = Query(None),
):
    require_user(request)
    return build_attachment_stream_response(local_file_path=local_file_path, file_name=file_name)


@app.get("/api/attachments/{attachment_id}/download")
def download_attachment_by_id(attachment_id: int, request: Request):
    require_user(request)
    table = email_table_or_404("attachments")

    id_column = None
    for candidate in ["attachment_id", "id"]:
        if candidate in table.c:
            id_column = table.c[candidate]
            break

    if id_column is None:
        raise HTTPException(status_code=400, detail="Attachment id column not found")

    rows = run_email_query(select(table).where(id_column == attachment_id).limit(1))
    if not rows:
        raise HTTPException(status_code=404, detail="Attachment not found")

    attachment = rows[0]
    return build_attachment_stream_response(
        local_file_path=attachment.get("local_file_path"),
        file_name=attachment.get("file_name"),
    )


@app.get("/api/emails/{message_id}")
def get_email(message_id: str, request: Request):
    require_user(request)
    emails = email_table_or_404("emails")
    if "message_id" not in emails.c:
        raise HTTPException(status_code=400, detail="Column message_id not found")

    statement = select(emails).where(emails.c.message_id == message_id).limit(1)
    rows = run_email_query(statement)
    if not rows:
        raise HTTPException(status_code=404, detail="Email not found")

    attachments = []
    tables = get_email_tables()
    if "attachments" in tables and "message_id" in tables["attachments"].c:
        attachment_table = tables["attachments"]
        attachment_statement = select(attachment_table).where(
            attachment_table.c.message_id == message_id
        )
        attachments = [serialize_attachment_row(row) for row in run_email_query(attachment_statement)]

    return {"email": rows[0], "attachments": attachments}


@app.get("/api/email-search")
def email_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    require_user(request)
    date_from_value = parse_date_param(date_from, "date_from")
    date_to_value = parse_date_param(date_to, "date_to")
    tables = get_email_tables()
    results = {"emails": [], "attachments": []}

    if "emails" in tables:
        results["emails"] = search_table_with_filters(
            tables["emails"],
            q,
            limit,
            date_from=date_from_value,
            date_to=date_to_value,
        )
    if "attachments" in tables:
        results["attachments"] = [
            serialize_attachment_row(row)
            for row in search_email_table(tables["attachments"], q, limit)
        ]

    return results


@app.get("/api/email-report")
def email_report(request: Request):
    require_user(request)
    try:
        return generate_email_report()
    except Exception:
        raise HTTPException(status_code=500, detail="Database error")


# ==============================
# Auth API
# ==============================

@app.post("/api/auth/signup")
async def signup(payload: SignupPayload, db: Session = Depends(get_db)):
    try:
        full_name = (payload.full_name or "").strip()
        email = (payload.email or "").strip().lower()
        password = payload.password or ""
        confirm = payload.confirm_password or ""

        if len(full_name) < 2:
            raise HTTPException(status_code=400, detail="full_name too short")
        if "@" not in email:
            raise HTTPException(status_code=400, detail="invalid email")
        if not email.endswith("@avocarbon.com"):
            raise HTTPException(status_code=400, detail="email must be @avocarbon.com")
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="password too short")
        if password != confirm:
            raise HTTPException(status_code=400, detail="password_mismatch")

        chat_db = get_chatbot_session()
        try:
            exists = chat_db.query(ChatbotUser.id).filter(ChatbotUser.email == email).first()
            if exists:
                raise HTTPException(status_code=409, detail="email already exists")

            base_username = email.split("@")[0] if "@" in email else email
            if not base_username:
                base_username = full_name

            username = unique_username(chat_db, base_username)
            password_hash = hash_password(password)

            chat_user = ChatbotUser(
                email=email,
                username=username,
                password_hash=password_hash,
                full_name=full_name,
            )
            chat_db.add(chat_user)
            chat_db.commit()
            chat_db.refresh(chat_user)

            chat_user_id = chat_user.id
            chat_user_full_name = chat_user.full_name
        except HTTPException:
            chat_db.rollback()
            raise
        except Exception as e:
            chat_db.rollback()
            print(f"Signup chatbot_users error: {e}")
            raise HTTPException(status_code=500, detail="server_error")
        finally:
            chat_db.close()

        access_token = create_access_token(email)
        refresh_token, refresh_expires = create_refresh_token_record(db, email, chat_user_id)

        response = JSONResponse(
            {
                "ok": True,
                "token": access_token,
                "user": {
                    "id": str(chat_user_id),
                    "email": email,
                    "full_name": chat_user_full_name,
                },
            }
        )
        set_refresh_cookie(response, refresh_token, refresh_expires)
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="server_error")


@app.post("/api/auth/login")
async def login(payload: LoginPayload, db: Session = Depends(get_db)):
    try:
        email = (payload.email or "").strip().lower()
        password = payload.password or ""

        chat_db = get_chatbot_session()
        try:
            chat_user = chat_db.query(ChatbotUser).filter(ChatbotUser.email == email).first()
            if not chat_user:
                raise HTTPException(status_code=401, detail="invalid credentials")

            if not verify_password(password, chat_user.password_hash):
                raise HTTPException(status_code=401, detail="invalid credentials")

            chat_user_id = chat_user.id
            chat_user_full_name = chat_user.full_name

            chat_user.last_login = datetime.utcnow()
            chat_db.commit()
        except HTTPException:
            chat_db.rollback()
            raise
        except Exception as e:
            chat_db.rollback()
            print(f"Login chatbot_users error: {e}")
            raise HTTPException(status_code=500, detail="server_error")
        finally:
            chat_db.close()

        access_token = create_access_token(email)
        refresh_token, refresh_expires = create_refresh_token_record(db, email, chat_user_id)

        resp = JSONResponse(
            {
                "ok": True,
                "token": access_token,
                "user": {
                    "id": str(chat_user_id),
                    "email": email,
                    "full_name": chat_user_full_name,
                },
            }
        )
        set_refresh_cookie(resp, refresh_token, refresh_expires)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="server_error")


@app.post("/api/auth/refresh")
def refresh_token(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="refresh_token_missing")

    token_hash = hash_refresh_token(token)
    now = utcnow()

    record = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    if not record or record.revoked_at is not None or record.expires_at <= now:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")

    record.revoked_at = now
    new_refresh = generate_refresh_token()
    new_hash = hash_refresh_token(new_refresh)
    new_expires = now + timedelta(days=REFRESH_TOKEN_TTL_DAYS)

    db.add(
        RefreshToken(
            user_id=record.user_id,
            email=record.email,
            token_hash=new_hash,
            expires_at=new_expires,
        )
    )
    db.commit()

    access_token = create_access_token(record.email)
    resp = JSONResponse({"ok": True, "token": access_token})
    set_refresh_cookie(resp, new_refresh, new_expires)
    return resp


@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordPayload, db: Session = Depends(get_db)):
    email = (payload.email or "").strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    chat_db = get_chatbot_session()
    try:
        user = chat_db.query(ChatbotUser).filter(ChatbotUser.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User does not exist")

        now = utcnow()

        db.query(PasswordResetToken).filter(
            PasswordResetToken.email == email,
            PasswordResetToken.used_at == None,
        ).update({PasswordResetToken.used_at: now}, synchronize_session=False)

        token = secrets.token_urlsafe(32)
        token_hash = hash_reset_token(token)
        expires_at = now + timedelta(hours=RESET_TOKEN_TTL_HOURS)

        record = PasswordResetToken(
            user_id=user.id,
            email=email,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        db.add(record)
        db.commit()

        reset_link = build_reset_link(token=token, email=email)
        send_reset_email(email, reset_link)

        return {"ok": True}
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        print(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="server_error")
    finally:
        chat_db.close()


@app.post("/api/auth/reset-password")
def reset_password(payload: ResetPasswordPayload, db: Session = Depends(get_db)):
    token = (payload.token or "").strip()
    password = payload.password or ""
    confirm = payload.confirm_password or ""
    email = (payload.email or "").strip().lower() if payload.email else None

    if not token:
        raise HTTPException(status_code=400, detail="token required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="password too short")
    if password != confirm:
        raise HTTPException(status_code=400, detail="password_mismatch")

    token_hash = hash_reset_token(token)
    chat_db = get_chatbot_session()

    try:
        now = utcnow()
        record = db.query(PasswordResetToken).filter(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at == None,
            PasswordResetToken.expires_at > now,
        ).first()

        if not record:
            raise HTTPException(status_code=400, detail="invalid_or_expired_token")

        user = chat_db.query(ChatbotUser).filter(ChatbotUser.id == record.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="user_not_found")
        if email and user.email.lower() != email:
            raise HTTPException(status_code=400, detail="email_mismatch")

        user.password_hash = hash_password(password)
        record.used_at = now

        chat_db.commit()
        db.commit()

        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        chat_db.rollback()
        db.rollback()
        print(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="server_error")
    finally:
        chat_db.close()


@app.post("/auth/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if token:
        token_hash = hash_refresh_token(token)
        now = utcnow()
        try:
            db.query(RefreshToken).filter(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at == None,
            ).update({RefreshToken.revoked_at: now}, synchronize_session=False)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Logout refresh token revoke error: {e}")

    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME, path="/")
    clear_refresh_cookie(resp)
    return resp


# ==============================
# Chat API
# ==============================

ACTIVE_BOT_ID = "email_report"


def get_active_bot():
    bot = BOTS.get(ACTIVE_BOT_ID)
    if not bot:
        raise HTTPException(status_code=500, detail="bot_not_configured")
    return bot


@app.post("/api/chat")
async def chat_api(payload: dict, request: Request, db: Session = Depends(get_db)):
    email = require_user(request)

    message = (payload.get("message") or "").strip()
    chat_id = payload.get("chat_id")
    bot_mode = payload.get("bot_mode")

    if not message:
        raise HTTPException(status_code=400, detail="message required")
    bot = get_active_bot()

    conv = None
    if chat_id is not None:
        try:
            payload_id = int(chat_id)
            conv = (
                db.query(Conversation)
                .filter(
                    Conversation.id == payload_id,
                    Conversation.email == email,
                    Conversation.is_deleted == False,
                )
                .first()
            )
        except (ValueError, TypeError):
            pass

    if not conv:
        user_id = get_chatbot_user_id(email)
        conv = create_conversation(db=db, email=email, user_id=user_id)
    elif conv.user_id is None:
        user_id = get_chatbot_user_id(email)
        if user_id:
            conv.user_id = user_id

    now = utcnow()

    user_message = Message(
        conversation_id=conv.id,
        role="user",
        content=message,
        created_at=now,
    )
    db.add(user_message)
    db.flush()

    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(60)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in msgs]

    session = {
        "history": history,
        "bot_mode": bot_mode,
        "user_email": email,
    }

    reply = bot["runner"](message, session)

    assistant_message = Message(
        conversation_id=conv.id,
        role="assistant",
        content=reply,
        created_at=utcnow(),
    )
    db.add(assistant_message)

    conv.updated_at = utcnow()

    if conv.title == "New chat":
        base = make_title(message)
        if base != "New chat":
            conv.title = unique_title(db=db, email=email, base=base, exclude_id=conv.id)

    db.commit()
    db.refresh(conv)

    return JSONResponse(
        {
            "reply": reply,
            "chat_id": conv.id,
            "title": normalize_title(conv.title),
            "updated_at": (conv.updated_at or utcnow()).isoformat(),
        }
    )


@app.post("/api/chat/stream")
async def chat_api_stream(payload: dict, request: Request, db: Session = Depends(get_db)):
    email = require_user(request)

    message = (payload.get("message") or "").strip()
    chat_id = payload.get("chat_id")
    bot_mode = payload.get("bot_mode")

    if not message:
        raise HTTPException(status_code=400, detail="message required")
    bot = get_active_bot()

    conv = None
    if chat_id is not None:
        try:
            payload_id = int(chat_id)
            conv = (
                db.query(Conversation)
                .filter(
                    Conversation.id == payload_id,
                    Conversation.email == email,
                    Conversation.is_deleted == False,
                )
                .first()
            )
        except (ValueError, TypeError):
            pass

    if not conv:
        user_id = get_chatbot_user_id(email)
        conv = create_conversation(db=db, email=email, user_id=user_id)
    elif conv.user_id is None:
        user_id = get_chatbot_user_id(email)
        if user_id:
            conv.user_id = user_id

    now = utcnow()

    user_message = Message(
        conversation_id=conv.id,
        role="user",
        content=message,
        created_at=now,
    )
    db.add(user_message)
    db.flush()

    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(60)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in msgs]

    session = {
        "history": history,
        "bot_mode": bot_mode,
        "user_email": email,
    }

    runner = bot["runner"]
    runner_stream = bot.get("runner_stream")

    def event_stream():
        assistant_parts = []
        try:
            yield sse_event({"chat_id": conv.id}, event="meta")

            if runner_stream:
                for chunk in runner_stream(message, session):
                    if not chunk:
                        continue
                    assistant_parts.append(chunk)
                    yield sse_event({"delta": chunk}, event="delta")
            else:
                reply = runner(message, session)
                for i in range(0, len(reply), 20):
                    chunk = reply[i:i + 20]
                    assistant_parts.append(chunk)
                    yield sse_event({"delta": chunk}, event="delta")

            assistant_text = "".join(assistant_parts)
            assistant_message = Message(
                conversation_id=conv.id,
                role="assistant",
                content=assistant_text,
                created_at=utcnow(),
            )
            db.add(assistant_message)

            conv.updated_at = utcnow()

            if conv.title == "New chat":
                base = make_title(message)
                if base != "New chat":
                    conv.title = unique_title(
                        db=db,
                        email=email,
                        base=base,
                        exclude_id=conv.id,
                    )

            db.commit()

            yield sse_event(
                {
                    "chat_id": conv.id,
                    "title": normalize_title(conv.title),
                    "updated_at": (conv.updated_at or utcnow()).isoformat(),
                },
                event="done",
            )
        except Exception as e:
            db.rollback()
            print(f"chat stream error: {e}")
            yield sse_event({"message": "server_error"}, event="error")

    resp = StreamingResponse(event_stream(), media_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


# ==============================
# History API
# ==============================

@app.get("/api/history")
def history_list(request: Request, db: Session = Depends(get_db)):
    email = require_user(request)
    convs = list_conversations(db=db, email=email)
    return {"items": build_history_items(convs)}


@app.post("/api/history/new")
def history_new(request: Request, db: Session = Depends(get_db)):
    email = require_user(request)
    user_id = get_chatbot_user_id(email)
    conv = create_conversation(db=db, email=email, user_id=user_id)
    return {"chat_id": conv.id, "title": conv.title}


@app.get("/api/history/{chat_id:int}")
def history_get(chat_id: int, request: Request, db: Session = Depends(get_db)):
    email = require_user(request)

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == chat_id,
            Conversation.email == email,
            Conversation.is_deleted == False,
        )
        .first()
    )
    if not conv:
        return {"chat_id": chat_id, "messages": []}

    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    return {
        "chat_id": conv.id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "is_edited": m.is_edited,
                "edited_at": m.edited_at.isoformat() if m.edited_at else None,
            }
            for m in msgs
        ],
        "title": normalize_title(conv.title),
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


@app.post("/api/history/{chat_id:int}/rename")
def history_rename(chat_id: int, payload: RenameChatPayload, request: Request, db: Session = Depends(get_db)):
    email = require_user(request)

    new_title = " ".join((payload.title or "").strip().split())
    if not new_title:
        raise HTTPException(status_code=400, detail="title required")

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == chat_id,
            Conversation.email == email,
            Conversation.is_deleted == False,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="chat not found")

    conv.title = unique_title(db=db, email=email, base=new_title, exclude_id=conv.id)
    conv.updated_at = utcnow()
    db.commit()

    return {
        "ok": True,
        "chat_id": conv.id,
        "title": normalize_title(conv.title),
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "full_title": conv.title,
    }


@app.post("/api/history/{chat_id:int}/messages/{message_id:int}/edit")
def message_edit(
    chat_id: int,
    message_id: int,
    payload: EditMessagePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    email = require_user(request)

    new_content = (payload.content or "").strip()
    if not new_content:
        raise HTTPException(status_code=400, detail="content required")

    msg = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(
            Conversation.id == chat_id,
            Conversation.email == email,
            Conversation.is_deleted == False,
            Message.id == message_id,
        )
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="message not found")

    if msg.role != "user":
        raise HTTPException(status_code=400, detail="only user messages can be edited")

    msg.content = new_content
    msg.is_edited = True
    msg.edited_at = utcnow()

    conv = msg.conversation
    if conv:
        conv.updated_at = utcnow()

    assistant_message = None

    if payload.regenerate:
        db.query(Message).filter(
            Message.conversation_id == msg.conversation_id,
            Message.id > msg.id,
        ).delete(synchronize_session=False)

        if conv:
            msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(Message.created_at.asc(), Message.id.asc())
                .all()
            )
            history = [{"role": m.role, "content": m.content} for m in msgs]
            session = {
                "history": history,
                "bot_mode": payload.bot_mode,
                "user_email": email,
            }

            bot = get_active_bot()
            reply = bot["runner"](new_content, session)
            now = utcnow()

            assistant_message = Message(
                conversation_id=conv.id,
                role="assistant",
                content=reply,
                created_at=now,
            )
            db.add(assistant_message)
            conv.updated_at = now

    db.commit()
    if assistant_message:
        db.refresh(assistant_message)

    return {
        "ok": True,
        "chat_id": chat_id,
        "message_id": msg.id,
        "edited_at": msg.edited_at.isoformat() if msg.edited_at else None,
        "assistant_message_id": assistant_message.id if assistant_message else None,
    }


@app.post("/api/history/{chat_id:int}/messages/{message_id:int}/edit/stream")
async def message_edit_stream(
    chat_id: int,
    message_id: int,
    payload: EditMessagePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    email = require_user(request)

    new_content = (payload.content or "").strip()
    if not new_content:
        raise HTTPException(status_code=400, detail="content required")

    msg = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(
            Conversation.id == chat_id,
            Conversation.email == email,
            Conversation.is_deleted == False,
            Message.id == message_id,
        )
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="message not found")

    if msg.role != "user":
        raise HTTPException(status_code=400, detail="only user messages can be edited")

    msg.content = new_content
    msg.is_edited = True
    msg.edited_at = utcnow()

    db.query(Message).filter(
        Message.conversation_id == msg.conversation_id,
        Message.id > msg.id,
    ).delete(synchronize_session=False)

    conv = msg.conversation
    bot = get_active_bot()

    def event_stream():
        assistant_parts = []
        try:
            yield sse_event({"chat_id": chat_id}, event="meta")

            msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(Message.created_at.asc(), Message.id.asc())
                .all()
            )
            history = [{"role": m.role, "content": m.content} for m in msgs]

            session = {
                "history": history,
                "bot_mode": payload.bot_mode,
                "user_email": email,
            }

            runner = bot["runner"]
            runner_stream = bot.get("runner_stream")

            if runner_stream:
                for chunk in runner_stream(new_content, session):
                    if not chunk:
                        continue
                    assistant_parts.append(chunk)
                    yield sse_event({"delta": chunk}, event="delta")
            else:
                reply = runner(new_content, session)
                for i in range(0, len(reply), 20):
                    chunk = reply[i:i + 20]
                    assistant_parts.append(chunk)
                    yield sse_event({"delta": chunk}, event="delta")

            assistant_text = "".join(assistant_parts)
            now = utcnow()

            assistant_message = Message(
                conversation_id=conv.id,
                role="assistant",
                content=assistant_text,
                created_at=now,
            )
            db.add(assistant_message)

            conv.updated_at = now
            db.commit()

            yield sse_event(
                {"chat_id": conv.id, "updated_at": now.isoformat()},
                event="done",
            )
        except Exception as e:
            db.rollback()
            print(f"message_edit_stream error: {e}")
            yield sse_event({"message": "server_error"}, event="error")

    resp = StreamingResponse(event_stream(), media_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.post("/api/history/{chat_id:int}/delete")
def history_delete(chat_id: int, request: Request, db: Session = Depends(get_db)):
    email = require_user(request)

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == chat_id,
            Conversation.email == email,
            Conversation.is_deleted == False,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="chat not found")

    conv.is_deleted = True
    conv.updated_at = utcnow()
    db.commit()

    return {"ok": True, "chat_id": chat_id}


@app.delete("/api/history/{chat_id:int}")
def history_delete_rest(chat_id: int, request: Request, db: Session = Depends(get_db)):
    return history_delete(chat_id=chat_id, request=request, db=db)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

import os
import uuid
from datetime import datetime
from urllib.parse import quote_plus

from sqlalchemy import DateTime, ForeignKey, Integer, Text, create_engine, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.sql import func

# Configuration
DB_USER = os.getenv("DB_USER", "administrationSTS")
DB_PASSWORD = os.getenv("DB_PASSWORD", "St$@0987")
DB_HOST = os.getenv("DB_HOST", "avo-adb-002.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "emails_chatbot")
CHATBOT_DB_NAME = os.getenv("CHATBOT_DB_NAME", "Chatbots_users")

# Encodez le mot de passe
password_encoded = quote_plus(DB_PASSWORD)
print(f"  Mot de passe encodé: {password_encoded}")

# Construire l'URL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
CHATBOT_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{CHATBOT_DB_NAME}?sslmode=require"

# Créer le moteur
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=30,
        max_overflow=10,
        echo=True
    )
    print("✅ Moteur SQLAlchemy créé")
except Exception as e:
    print(f"❌ Erreur création moteur: {e}")
    engine = None

# Creer le moteur chatbots_users
try:
    engine_chatbot = create_engine(
        CHATBOT_DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=30,
        max_overflow=10,
        echo=True
    )
    print("Moteur SQLAlchemy cree (Chatbots_users)")
except Exception as e:
    print(f"Erreur creation moteur Chatbots_users: {e}")
    engine_chatbot = None

if engine:
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
else:
    SessionLocal = None

if engine_chatbot:
    SessionChatbot = sessionmaker(
        bind=engine_chatbot,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
else:
    SessionChatbot = None

class Base(DeclarativeBase):
    pass

class ChatbotBase(DeclarativeBase):
    pass

class ChatbotUser(ChatbotBase):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    is_verified: Mapped[bool] = mapped_column(default=False)

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)
    email: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)
    email: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # UUID as text for simplicity
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True, nullable=True)
    email: Mapped[str] = mapped_column(Text, index=True)
    title: Mapped[str] = mapped_column(Text, default="New chat")
    created_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False)
    is_deleted: Mapped[bool] = mapped_column(default=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at.asc()",
    )

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) 
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(Text)     # user|assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_edited: Mapped[bool] = mapped_column(default=False)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

def get_db():
    if not SessionLocal:
        raise Exception("Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_chatbot_session():
    if not SessionChatbot:
        raise Exception("Chatbot database not initialized")
    return SessionChatbot()


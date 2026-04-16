import mimetypes
import os
from functools import lru_cache
from urllib.parse import quote

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

AZURE_CONNECTION_STRING_ENV = "AZURE_CONNECTION_STRING"
AZURE_CONTAINER_NAME_ENV = "AZURE_CONTAINER_NAME"
LOCAL_PATH_PREFIX = "extracted_emails/"


class AttachmentStorageError(RuntimeError):
    pass


class AttachmentNotFoundError(FileNotFoundError):
    pass


class AttachmentAmbiguousError(RuntimeError):
    pass


def _clean_value(value: str | None) -> str:
    return (value or "").strip()


def _normalize_blob_path(value: str | None) -> str:
    cleaned = _clean_value(value).replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def azure_storage_enabled() -> bool:
    return bool(_clean_value(os.getenv(AZURE_CONNECTION_STRING_ENV))) and bool(
        _clean_value(os.getenv(AZURE_CONTAINER_NAME_ENV))
    )


def get_container_name() -> str:
    name = _clean_value(os.getenv(AZURE_CONTAINER_NAME_ENV))
    if not name:
        raise AttachmentStorageError("Azure container name is not configured")
    return name


@lru_cache(maxsize=1)
def get_blob_service_client() -> BlobServiceClient:
    connection_string = _clean_value(os.getenv(AZURE_CONNECTION_STRING_ENV))
    if not connection_string:
        raise AttachmentStorageError("Azure connection string is not configured")
    return BlobServiceClient.from_connection_string(connection_string)


def get_blob_client(blob_name: str):
    return get_blob_service_client().get_blob_client(
        container=get_container_name(),
        blob=blob_name,
    )


def _blob_exists(blob_name: str) -> bool:
    if not blob_name:
        return False
    try:
        return get_blob_client(blob_name).exists()
    except AzureError as exc:
        raise AttachmentStorageError("Unable to reach Azure Blob Storage") from exc


def build_blob_name_candidates(local_file_path: str | None, file_name: str | None) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str | None):
        normalized = _normalize_blob_path(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    normalized_path = _normalize_blob_path(local_file_path)
    add(normalized_path)

    if normalized_path.startswith(LOCAL_PATH_PREFIX):
        add(normalized_path[len(LOCAL_PATH_PREFIX) :])

    if file_name:
        add(os.path.basename(_normalize_blob_path(file_name)))

    if normalized_path:
        add(os.path.basename(normalized_path))

    return candidates


@lru_cache(maxsize=1)
def get_blob_names_by_basename() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    container = get_blob_service_client().get_container_client(get_container_name())
    try:
        for blob in container.list_blobs(results_per_page=500):
            name = blob.name
            base_name = name.rsplit("/", 1)[-1].lower()
            mapping.setdefault(base_name, []).append(name)
    except AzureError as exc:
        raise AttachmentStorageError("Unable to index Azure Blob Storage") from exc
    return mapping


def _prefer_matches(matches: list[str], candidates: list[str]) -> list[str]:
    if len(matches) <= 1:
        return matches

    hints = []
    for candidate in candidates:
        parts = [part for part in candidate.split("/") if part]
        if len(parts) > 1:
            hints.extend(parts[:-1])

    filtered = matches
    for hint in reversed(hints):
        narrowed = [item for item in filtered if hint.lower() in item.lower()]
        if narrowed:
            filtered = narrowed
        if len(filtered) == 1:
            return filtered
    return filtered


def resolve_attachment_blob_name(
    local_file_path: str | None = None,
    file_name: str | None = None,
) -> str:
    candidates = build_blob_name_candidates(local_file_path, file_name)
    for candidate in candidates:
        if _blob_exists(candidate):
            return candidate

    normalized_file_name = _clean_value(file_name).lower()
    if normalized_file_name:
        matches = list(get_blob_names_by_basename().get(normalized_file_name, []))
        matches = _prefer_matches(matches, candidates)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttachmentAmbiguousError("Multiple Azure blobs match this attachment")

    raise AttachmentNotFoundError("Attachment blob not found in Azure storage")

def build_download_headers(file_name: str | None) -> dict[str, str]:
    resolved_name = _clean_value(file_name) or "attachment"
    ascii_name = resolved_name.encode("ascii", "ignore").decode("ascii") or "attachment"
    ascii_name = ascii_name.replace('"', "")
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(resolved_name)}"
        ),
        "Cache-Control": "no-store",
    }


def open_attachment_download(
    local_file_path: str | None = None,
    file_name: str | None = None,
) -> dict:
    blob_name = resolve_attachment_blob_name(local_file_path=local_file_path, file_name=file_name)
    blob_client = get_blob_client(blob_name)
    try:
        properties = blob_client.get_blob_properties()
        downloader = blob_client.download_blob()
    except ResourceNotFoundError as exc:
        raise AttachmentNotFoundError("Attachment blob not found in Azure storage") from exc
    except AzureError as exc:
        raise AttachmentStorageError("Unable to download attachment from Azure storage") from exc

    resolved_name = _clean_value(file_name) or blob_name.rsplit("/", 1)[-1]
    content_type = properties.content_settings.content_type or mimetypes.guess_type(resolved_name)[0]

    return {
        "blob_name": blob_name,
        "file_name": resolved_name,
        "content_type": content_type or "application/octet-stream",
        "content_length": properties.size,
        "headers": build_download_headers(resolved_name),
        "chunks": downloader.chunks(),
    }

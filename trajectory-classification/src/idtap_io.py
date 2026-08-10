"""Reusable IDTAP API helpers for authentication and transcription loading."""

from __future__ import annotations

from typing import Any

import requests
from idtap import Piece, SwaraClient, login_google


class IDTAPError(Exception):
    """Base exception for IDTAP I/O errors."""


class AuthenticationError(IDTAPError):
    """Raised when authentication fails or credentials are missing."""


class ConnectionError(IDTAPError):
    """Raised when the IDTAP API cannot be reached."""


class TranscriptionNotFoundError(IDTAPError):
    """Raised when a transcription ID does not exist or is not accessible."""


class MalformedDataError(IDTAPError):
    """Raised when transcription JSON cannot be parsed into a Piece."""


def create_client(*, authenticate: bool = True) -> SwaraClient:
    """Create an authenticated SwaraClient.

    Uses stored tokens when available. Calls ``login_google()`` only when
    ``authenticate`` is True and no valid token is present.

    Raises:
        AuthenticationError: If authentication is required but fails.
        ConnectionError: If the API cannot be reached during login.
    """
    try:
        if authenticate:
            client = SwaraClient(auto_login=True)
        else:
            client = SwaraClient(auto_login=False)
    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Could not connect to the IDTAP API. Check your network connection."
        ) from exc
    except requests.Timeout as exc:
        raise ConnectionError("Timed out while connecting to the IDTAP API.") from exc
    except Exception as exc:
        raise AuthenticationError(
            "Failed to authenticate with IDTAP. Run login_google() or check stored tokens."
        ) from exc

    if authenticate and not client.token:
        try:
            login_google(storage=client.secure_storage, base_url=client.base_url)
            client.load_token()
        except requests.ConnectionError as exc:
            raise ConnectionError(
                "Could not connect to the IDTAP API during authentication."
            ) from exc
        except Exception as exc:
            raise AuthenticationError("IDTAP authentication failed.") from exc

    if authenticate and not client.token:
        raise AuthenticationError(
            "No valid IDTAP authentication token is available after login."
        )

    return client


def list_transcriptions(client: SwaraClient) -> list[dict[str, Any]]:
    """Return transcriptions accessible to the authenticated user.

    Wraps ``client.get_viewable_transcriptions()`` (the verified API method).

    Raises:
        AuthenticationError: On 401/unauthorized responses.
        ConnectionError: On network failures.
    """
    try:
        result = client.get_viewable_transcriptions()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Could not connect to the IDTAP API while listing transcriptions."
        ) from exc
    except requests.Timeout as exc:
        raise ConnectionError(
            "Timed out while listing transcriptions from the IDTAP API."
        ) from exc
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise AuthenticationError(
                "Not authorized to list transcriptions. Re-authenticate with IDTAP."
            ) from exc
        raise ConnectionError(
            f"HTTP error while listing transcriptions: {exc}"
        ) from exc
    except Exception as exc:
        raise ConnectionError(
            f"Unexpected error while listing transcriptions: {exc}"
        ) from exc

    if not isinstance(result, list):
        raise MalformedDataError(
            f"Expected a list of transcriptions, got {type(result).__name__}."
        )
    return result


def load_transcription_json(client: SwaraClient, piece_id: str) -> dict[str, Any]:
    """Load raw transcription JSON for a piece ID.

    Wraps ``client.get_piece(piece_id)`` (the verified API method).

    Raises:
        TranscriptionNotFoundError: If the transcription is missing or forbidden.
        AuthenticationError: On unauthorized access.
        ConnectionError: On network failures.
        MalformedDataError: If the response is not a dictionary.
    """
    try:
        piece_data = client.get_piece(piece_id)
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to the IDTAP API while loading piece {piece_id!r}."
        ) from exc
    except requests.Timeout as exc:
        raise ConnectionError(
            f"Timed out while loading piece {piece_id!r} from the IDTAP API."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise TranscriptionNotFoundError(
                f"Transcription not found: {piece_id!r}"
            ) from exc
        if status == 403:
            raise TranscriptionNotFoundError(
                f"Transcription not accessible: {piece_id!r}"
            ) from exc
        if status == 401:
            raise AuthenticationError(
                f"Not authorized to load transcription {piece_id!r}."
            ) from exc
        raise ConnectionError(
            f"HTTP error while loading transcription {piece_id!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise ConnectionError(
            f"Unexpected error while loading transcription {piece_id!r}: {exc}"
        ) from exc

    if not isinstance(piece_data, dict):
        raise MalformedDataError(
            f"Expected dict transcription data for {piece_id!r}, "
            f"got {type(piece_data).__name__}."
        )
    return piece_data


def load_piece(client: SwaraClient, piece_id: str) -> Piece:
    """Load a transcription and convert it to an IDTAP Piece object.

    Raises:
        TranscriptionNotFoundError: If the transcription is missing or forbidden.
        AuthenticationError: On unauthorized access.
        ConnectionError: On network failures.
        MalformedDataError: If the JSON cannot be converted to a Piece.
    """
    piece_data = load_transcription_json(client, piece_id)
    try:
        return Piece.from_json(piece_data)
    except Exception as exc:
        raise MalformedDataError(
            f"Could not parse transcription {piece_id!r} into a Piece: {exc}"
        ) from exc


def piece_id_from_entry(entry: dict[str, Any]) -> str | None:
    """Extract a transcription piece ID from a list entry."""
    piece_id = entry.get("_id")
    if piece_id is None:
        return None
    return str(piece_id)


def piece_title_from_entry(entry: dict[str, Any]) -> str:
    """Extract a display title from a transcription list entry."""
    title = entry.get("title") or entry.get("name")
    if title:
        return str(title)
    piece_id = piece_id_from_entry(entry)
    return piece_id or "unknown"


def audio_filename_from_entry(entry: dict[str, Any]) -> str:
    """Best-effort audio filename from a transcription list entry."""
    for key in (
        "originalFileName",
        "original_filename",
        "filename",
        "audioFilename",
        "audio_filename",
        "fileName",
        "file_name",
    ):
        value = entry.get(key)
        if value:
            return str(value)
    return ""

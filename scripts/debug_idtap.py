#!/usr/bin/env python3
"""Inspect the runtime structure of the installed IDTAP Python API."""

from __future__ import annotations

import traceback
from typing import Any

from idtap import Piece, SwaraClient, login_google

SENSITIVE_KEY_PARTS = (
    "token",
    "auth",
    "cookie",
    "credential",
    "password",
    "secret",
    "bearer",
    "authorization",
)

STRUCTURAL_FIELD_NAMES = (
    "phrases",
    "phrase_grid",
    "trajectories",
    "trajectory_grid",
    "instrumentation",
    "sections",
    "meters",
    "raga",
)

TRAJECTORY_FIELD_NAMES = ("id", "unique_id", "dur_tot", "name_")


def heading(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def sanitize_value(value: Any, *, max_len: int = 300) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if is_sensitive_key(str(key))
            else sanitize_value(item, max_len=max_len)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 5:
            preview = [sanitize_value(item, max_len=max_len) for item in value[:5]]
            return f"{type(value).__name__}(len={len(value)}, preview={preview!r})"
        return [sanitize_value(item, max_len=max_len) for item in value]
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return value


def short_repr(value: Any, *, max_len: int = 500) -> str:
    text = repr(sanitize_value(value, max_len=max_len))
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def public_dir(obj: Any) -> list[str]:
    return sorted(name for name in dir(obj) if not name.startswith("__"))


def public_attributes(obj: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in public_dir(obj):
        try:
            value = getattr(obj, name)
        except Exception as exc:
            attrs[name] = f"<unreadable: {type(exc).__name__}: {exc}>"
            continue
        if callable(value):
            continue
        attrs[name] = value
    return attrs


def print_value_summary(label: str, value: Any) -> None:
    print(f"{label} type: {type(value)!r}")
    try:
        length = len(value)
    except TypeError:
        pass
    else:
        print(f"{label} length: {length}")

    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{label} dictionary keys: {keys}")
        print(f"{label} shortened: {short_repr(value)}")
        return

    print(f"{label} shortened: {short_repr(value)}")
    if not isinstance(value, (str, bytes, int, float, bool, type(None))):
        print(f"{label} public dir(): {public_dir(value)}")


def discover_transcription_identifier(item: Any) -> tuple[str | None, Any]:
    """Return (field_name, value) for the first scalar id-like field found."""
    if isinstance(item, dict):
        candidates: list[tuple[str, Any]] = []
        for key, value in item.items():
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            if "id" in str(key).lower():
                candidates.append((str(key), value))
        if candidates:
            candidates.sort(key=lambda pair: pair[0])
            return candidates[0]

    for name in public_dir(item):
        if "id" not in name.lower():
            continue
        try:
            value = getattr(item, name)
        except Exception:
            continue
        if value is None or callable(value) or isinstance(value, (dict, list, tuple, set)):
            continue
        return name, value

    return None, None


def print_exception(prefix: str, exc: BaseException) -> None:
    print(f"{prefix}: {type(exc).__name__}: {exc}")


def main() -> None:
    client: SwaraClient | None = None
    transcriptions: Any = None
    first_item: Any = None
    transcription_id: Any = None
    transcription_id_field: str | None = None
    transcription_detail: Any = None
    piece: Piece | None = None
    trajectories: Any = None

    heading("1. AUTHENTICATION")
    try:
        client = SwaraClient(auto_login=False)
        if not client.token:
            print("No stored token found; calling login_google().")
            login_google(storage=client.secure_storage, base_url=client.base_url)
            client.load_token()
        if not client.token:
            raise RuntimeError("Authentication completed but no token is available.")
        auth_info = client.get_auth_info()
        safe_auth_info = {
            key: "<redacted>" if is_sensitive_key(key) else value
            for key, value in auth_info.items()
        }
        print(f"Authenticated: {safe_auth_info.get('authenticated')}")
        print(f"User id: {safe_auth_info.get('user_id')!r}")
        print(f"User email: {safe_auth_info.get('user_email')!r}")
        print(f"Storage info: {short_repr(safe_auth_info.get('storage_info'))}")
    except Exception as exc:
        print_exception("Authentication failed", exc)
        traceback.print_exc()
        return

    heading("2. CREATE CLIENT")
    try:
        if client is None:
            client = SwaraClient(auto_login=True)
        print(f"Client type: {type(client)!r}")
        print(f"Client public dir(): {public_dir(client)}")
        print(f"Has get_transcriptions: {hasattr(client, 'get_transcriptions')}")
        print(f"Has get_transcription: {hasattr(client, 'get_transcription')}")
        print(f"Has get_viewable_transcriptions: {hasattr(client, 'get_viewable_transcriptions')}")
        print(f"Has get_piece: {hasattr(client, 'get_piece')}")
    except Exception as exc:
        print_exception("Client creation failed", exc)
        traceback.print_exc()
        return

    heading("3. LIST TRANSCRIPTIONS")
    list_method_used: str | None = None
    if hasattr(client, "get_transcriptions"):
        try:
            transcriptions = client.get_transcriptions()
            list_method_used = "get_transcriptions"
            print_value_summary("get_transcriptions() return value", transcriptions)
            if transcriptions:
                first_item = transcriptions[0]
                heading("3a. FIRST LIST ITEM")
                print_value_summary("first item", first_item)
        except Exception as exc:
            print_exception("client.get_transcriptions() failed", exc)
            traceback.print_exc()
    else:
        print("client.get_transcriptions() is not present on this SwaraClient.")

    if transcriptions is None and hasattr(client, "get_viewable_transcriptions"):
        heading("3b. client.get_viewable_transcriptions()")
        try:
            transcriptions = client.get_viewable_transcriptions()
            list_method_used = "get_viewable_transcriptions"
            print_value_summary(
                "get_viewable_transcriptions() return value", transcriptions
            )
            if transcriptions:
                first_item = transcriptions[0]
                heading("3c. FIRST LIST ITEM")
                print_value_summary("first item", first_item)
        except Exception as exc:
            print_exception("client.get_viewable_transcriptions() failed", exc)
            traceback.print_exc()

    heading("4. SELECT FIRST TRANSCRIPTION AND DISCOVER IDENTIFIER")
    try:
        if transcriptions is None:
            raise RuntimeError("No transcription list is available from step 3.")
        if list_method_used:
            print(f"List loaded via: client.{list_method_used}()")
        if not transcriptions:
            raise RuntimeError("Transcription list is empty.")
        first_item = transcriptions[0]
        transcription_id_field, transcription_id = discover_transcription_identifier(
            first_item
        )
        if transcription_id_field is None:
            if isinstance(first_item, dict):
                print(f"Available keys on first item: {list(first_item.keys())}")
            else:
                print(f"Public attributes on first item: {list(public_attributes(first_item))}")
            raise RuntimeError(
                "Could not discover a scalar id-like field on the first transcription."
            )
        print(f"Discovered identifier field: {transcription_id_field!r}")
        print(f"Discovered identifier value: {transcription_id!r}")
    except Exception as exc:
        print_exception("Identifier discovery failed", exc)
        traceback.print_exc()

    heading("5. LOAD ONE TRANSCRIPTION")
    detail_method_used: str | None = None
    if hasattr(client, "get_transcription"):
        try:
            if transcription_id is None:
                raise RuntimeError("No transcription identifier is available from step 4.")
            transcription_detail = client.get_transcription(transcription_id)
            detail_method_used = "get_transcription"
            print_value_summary("get_transcription() return value", transcription_detail)
        except Exception as exc:
            print_exception("client.get_transcription(...) failed", exc)
            traceback.print_exc()
    else:
        print("client.get_transcription(...) is not present on this SwaraClient.")

    if transcription_detail is None and transcription_id is not None and hasattr(
        client, "get_piece"
    ):
        heading("5b. client.get_piece(...)")
        try:
            transcription_detail = client.get_piece(transcription_id)
            detail_method_used = "get_piece"
            print_value_summary("get_piece() return value", transcription_detail)
            if detail_method_used:
                print(f"Detail loaded via: client.{detail_method_used}({transcription_id!r})")
        except Exception as exc:
            print_exception("client.get_piece(...) failed", exc)
            traceback.print_exc()

    heading("6. Piece.from_json(...)")
    try:
        if transcription_detail is None:
            raise RuntimeError("No transcription detail is available from step 5.")
        piece = Piece.from_json(transcription_detail)
        print(f"Piece type: {type(piece)!r}")
        piece_attrs = public_attributes(piece)
        print(f"Piece public attributes: {sorted(piece_attrs)}")
        for field_name in STRUCTURAL_FIELD_NAMES:
            if hasattr(piece, field_name):
                value = getattr(piece, field_name)
                print(
                    f"piece.{field_name} type: {type(value)!r}; "
                    f"value: {short_repr(value)}"
                )
    except Exception as exc:
        print_exception("Piece.from_json(...) failed", exc)
        traceback.print_exc()

    heading("7. piece.all_trajectories()")
    try:
        if piece is None:
            raise RuntimeError("No Piece object is available from step 6.")
        trajectories = piece.all_trajectories()
        print(f"all_trajectories() return type: {type(trajectories)!r}")
        try:
            print(f"all_trajectories() length: {len(trajectories)}")
        except TypeError:
            pass
        if trajectories:
            first_traj = trajectories[0]
            heading("7a. FIRST TRAJECTORY")
            print(f"first trajectory type: {type(first_traj)!r}")
            print(f"first trajectory public dir(): {public_dir(first_traj)}")
            print(f"first trajectory shortened: {short_repr(first_traj)}")
            for field_name in TRAJECTORY_FIELD_NAMES:
                if hasattr(first_traj, field_name):
                    value = getattr(first_traj, field_name)
                    print(f"first trajectory.{field_name}: {short_repr(value)}")
    except Exception as exc:
        print_exception("piece.all_trajectories() failed", exc)
        traceback.print_exc()

    heading("8. piece.traj_start_times()")
    try:
        if piece is None:
            raise RuntimeError("No Piece object is available from step 6.")
        start_times = piece.traj_start_times()
        print(f"traj_start_times() return type: {type(start_times)!r}")
        print(f"traj_start_times() shortened: {short_repr(start_times)}")
    except Exception as exc:
        print_exception("piece.traj_start_times() failed", exc)
        traceback.print_exc()


if __name__ == "__main__":
    main()

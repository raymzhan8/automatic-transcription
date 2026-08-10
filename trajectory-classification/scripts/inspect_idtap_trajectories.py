"""Inspect IDTAP transcriptions and export a trajectory inventory CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idtap_io import (  # noqa: E402
    AuthenticationError,
    ConnectionError,
    IDTAPError,
    MalformedDataError,
    TranscriptionNotFoundError,
    audio_filename_from_entry,
    create_client,
    list_transcriptions,
    load_piece,
    piece_id_from_entry,
    piece_title_from_entry,
)
from trajectory_inventory import (  # noqa: E402
    FailedTranscription,
    InventoryResult,
    extract_piece_inventory,
    print_inventory_summary,
    print_warnings,
    write_counts_csv,
    write_failures_csv,
    write_inventory_csv,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
INVENTORY_PATH = OUTPUT_DIR / "trajectory_inventory.csv"
COUNTS_PATH = OUTPUT_DIR / "trajectory_counts.csv"
FAILURES_PATH = OUTPUT_DIR / "failed_transcriptions.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export an inventory of IDTAP trajectory annotations."
    )
    parser.add_argument(
        "--piece-id",
        help="Process only this transcription piece ID",
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        help="Maximum number of transcriptions to process",
    )
    return parser.parse_args()


def select_transcriptions(
    entries: list[dict],
    *,
    piece_id: str | None,
    max_pieces: int | None,
) -> list[dict]:
    """Filter transcription list entries based on CLI options."""
    selected = entries
    if piece_id is not None:
        selected = [
            entry
            for entry in selected
            if piece_id_from_entry(entry) == piece_id
        ]
        if not selected:
            raise TranscriptionNotFoundError(
                f"Transcription {piece_id!r} was not found in the accessible list."
            )
    if max_pieces is not None:
        if max_pieces < 0:
            raise ValueError("--max-pieces must be nonnegative")
        selected = selected[:max_pieces]
    return selected


def process_transcriptions(
    entries: list[dict],
    *,
    client,
) -> InventoryResult:
    """Load each transcription and collect inventory rows and failures."""
    result = InventoryResult()

    for entry in entries:
        piece_id = piece_id_from_entry(entry)
        if piece_id is None:
            result.failures.append(
                FailedTranscription(
                    piece_id="",
                    piece_title=piece_title_from_entry(entry),
                    error_type="MalformedDataError",
                    error_message="Transcription list entry is missing _id",
                )
            )
            continue

        piece_title = piece_title_from_entry(entry)
        audio_filename = audio_filename_from_entry(entry)

        try:
            piece = load_piece(client, piece_id)
            resolved_title = piece.title or piece_title
            resolved_piece_id = str(piece._id or piece_id)
            rows, warnings = extract_piece_inventory(
                piece,
                piece_id=resolved_piece_id,
                piece_title=resolved_title,
                audio_filename=audio_filename,
            )
            result.rows.extend(rows)
            result.warnings.extend(warnings)
            print(
                f"OK: {resolved_title} ({resolved_piece_id}) "
                f"- {len(rows)} trajectories"
            )
        except IDTAPError as exc:
            result.failures.append(
                FailedTranscription(
                    piece_id=piece_id,
                    piece_title=piece_title,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            print(f"FAIL: {piece_title} ({piece_id}) - {type(exc).__name__}: {exc}")
        except Exception as exc:
            result.failures.append(
                FailedTranscription(
                    piece_id=piece_id,
                    piece_title=piece_title,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            print(f"FAIL: {piece_title} ({piece_id}) - {type(exc).__name__}: {exc}")

    return result


def main() -> None:
    """Run the IDTAP trajectory inventory export."""
    args = parse_args()

    try:
        client = create_client()
        entries = list_transcriptions(client)
        selected = select_transcriptions(
            entries,
            piece_id=args.piece_id,
            max_pieces=args.max_pieces,
        )
    except (AuthenticationError, ConnectionError, TranscriptionNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Processing {len(selected)} transcription(s)...")
    result = process_transcriptions(selected, client=client)

    write_inventory_csv(result.rows, INVENTORY_PATH)
    write_counts_csv(result.rows, COUNTS_PATH)
    write_failures_csv(result.failures, FAILURES_PATH)

    print_inventory_summary(result.rows)
    print_warnings(result.warnings)

    print("\n=== Output Files ===")
    print(f"  Inventory: {INVENTORY_PATH}")
    print(f"  Counts:    {COUNTS_PATH}")
    print(f"  Failures:  {FAILURES_PATH}")
    print(
        f"\nProcessed {len(selected)} transcription(s); "
        f"{len(result.rows)} trajectories exported; "
        f"{len(result.failures)} failure(s); "
        f"{len(result.warnings)} warning(s)"
    )

    if result.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

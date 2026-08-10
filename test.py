from pathlib import Path

from idtap import SwaraClient, Piece

OUTPUT_DIR = Path(__file__).parent / "output"
AUDIO_FORMAT = "wav"

client = SwaraClient()

transcriptions = client.get_viewable_transcriptions()
print(f"Found {len(transcriptions)} transcriptions")

piece = None
piece_data = None
transcription = None

for candidate in transcriptions:
    if not candidate.get("audioID"):
        continue

    try:
        client.download_audio(candidate["audioID"], format=AUDIO_FORMAT)
        piece_data = client.get_piece(candidate["_id"])
        piece = Piece.from_json(piece_data)
    except Exception:
        continue

    transcription = candidate
    break

if piece is None:
    raise RuntimeError("No transcription with downloadable audio was found.")

print(f"Loading: {piece.title} ({transcription['_id']})")
print(f"Raga: {piece.raga.name if piece.raga else 'Unknown'}")
print(f"Instrument: {piece.instrumentation}")
print(f"Trajectories: {sum(len(p.trajectories) for p in piece.phrases)}")
print(f"Audio ID: {piece.audio_id}")

OUTPUT_DIR.mkdir(exist_ok=True)

audio_path = client.download_and_save_transcription_audio(
    piece,
    format=AUDIO_FORMAT,
    filepath=str(OUTPUT_DIR),
)
print(f"Audio saved to: {audio_path}")

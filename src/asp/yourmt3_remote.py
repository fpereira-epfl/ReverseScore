"""Remote transcription backend using the YourMT3 HuggingFace Space.

This module uploads a single stem audio file to the public YourMT3 Gradio demo
and downloads the resulting multi-track MIDI. It requires a network connection
and may be subject to HuggingFace Spaces queueing / rate limits.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .utils import ensure_dirs

logger = logging.getLogger(__name__)

_MIDI_DATA_URI_RE = re.compile(r"data:audio/midi;base64,([^\s\"'<>]+)")


def _load_gradio_client() -> Any:
    """Lazy import gradio_client to keep CLI startup fast when unused."""
    from gradio_client import Client, handle_file

    return Client, handle_file


def _midi_has_notes(midi_bytes: bytes) -> bool:
    """Return True if the MIDI file contains at least one Note On event."""
    # Scan raw bytes for running-status-friendly Note On status bytes (0x90-0x9F)
    # followed by a pitch and a non-zero velocity. This is intentionally simple
    # because we only need to distinguish an empty placeholder from a real file.
    i = 0
    length = len(midi_bytes)
    while i < length - 2:
        byte = midi_bytes[i]
        if 0x90 <= byte <= 0x9F:
            # Note On: pitch, velocity. Velocity 0 is a Note Off in disguise.
            velocity = midi_bytes[i + 2]
            if velocity > 0:
                return True
            i += 3
        elif 0x80 <= byte <= 0xEF:
            # Other channel voice/mode messages: skip fixed-length data bytes.
            if (
                0x80 <= byte <= 0x9F or 0xA0 <= byte <= 0xBF
            ):  # Note On/Off (already handled On above)
                i += 3
            elif 0xC0 <= byte <= 0xDF:  # Program change / channel pressure
                i += 2
            elif 0xE0 <= byte <= 0xEF:  # Pitch bend
                i += 3
        elif byte == 0xFF:
            # Meta-event: type + variable-length length + data.
            if i + 2 >= length:
                break
            i += 2
            data_len = 0
            while i < length:
                part = midi_bytes[i]
                i += 1
                data_len = (data_len << 7) | (part & 0x7F)
                if not (part & 0x80):
                    break
            i += data_len
        elif byte == 0xF0 or byte == 0xF7:
            # SysEx: skip until next 0xF7 (or bail safely).
            i += 1
            while i < length and midi_bytes[i] != 0xF7:
                i += 1
            i += 1
        else:
            i += 1
    return False


def _extract_midi_bytes(html: str) -> bytes:
    """Decode the first base64-encoded MIDI data URI from Space HTML output.

    The YourMT3 Space returns an HTML player page with the generated MIDI
    embedded as one or more ``data:audio/midi;base64,...`` URIs. We extract
    and decode the first one that actually contains notes.
    """
    matches = _MIDI_DATA_URI_RE.findall(html)
    if not matches:
        raise ValueError("YourMT3 response did not contain a MIDI data URI")

    for b64 in matches:
        data = base64.b64decode(b64)
        if _midi_has_notes(data):
            return data

    # All URIs were empty; return the first one so callers get a valid empty MIDI.
    return base64.b64decode(matches[0])


def transcribe_stem_yourmt3(
    audio_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcribe a single stem audio file to MIDI via the YourMT3 Space.

    Args:
        audio_path: Path to the stem audio file.
        output_dir: Directory for the generated MIDI file.
        config: Pipeline configuration. ``yourmt3_space_id`` and
            ``yourmt3_timeout_seconds`` are read from this.
        label: Optional label used for the output filename.
        overwrite: Re-transcribe even if MIDI already exists.

    Returns:
        Path to the generated ``.mid`` file.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If the Space call fails or no MIDI is returned.
    """
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Stem audio not found: {audio_path}")

    ensure_dirs(output_dir)
    from .utils import safe_stem_name

    name = safe_stem_name(label or audio_path.stem)
    midi_path = output_dir / f"{name}.mid"

    if midi_path.exists() and not overwrite:
        logger.info("Reusing existing MIDI: %s", midi_path)
        return midi_path

    logger.info(
        "Transcribing %s -> %s via YourMT3 Space %s",
        audio_path,
        midi_path,
        config.yourmt3_space_id,
    )

    # gradio_client and httpx are chatty at INFO/DEBUG; raise them to WARNING
    # for the duration of the Space call.
    noisy_loggers = [
        logging.getLogger(name) for name in ("gradio_client", "httpx", "urllib3", "huggingface_hub")
    ]
    old_levels = [(lg, lg.level) for lg in noisy_loggers]
    for lg, _ in old_levels:
        lg.setLevel(logging.WARNING)

    Client, handle_file = _load_gradio_client()
    try:
        client = Client(config.yourmt3_space_id)
        result = client.predict(
            audio_filepath=handle_file(str(audio_path)),
            api_name="/process_audio",
        )
    except Exception as exc:
        raise RuntimeError(f"YourMT3 Space call failed for {audio_path}") from exc
    finally:
        for lg, level in old_levels:
            lg.setLevel(level)

    if not isinstance(result, str):
        raise RuntimeError(f"Unexpected YourMT3 response type: {type(result).__name__}")

    try:
        midi_bytes = _extract_midi_bytes(result)
    except Exception as exc:
        raise RuntimeError(
            f"Could not extract MIDI from YourMT3 response for {audio_path}"
        ) from exc

    midi_path.write_bytes(midi_bytes)
    logger.info(
        "YourMT3 transcription complete: %s (%d bytes)",
        midi_path,
        len(midi_bytes),
    )
    return midi_path

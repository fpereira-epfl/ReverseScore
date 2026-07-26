"""Song metadata recognition using ShazamIO.

This module recognizes the artist and title of an audio file by sending it to
Shazam via the ``shazamio`` library. Inputs that are not in ShazamIO's reliably
supported set (e.g. ``.aifc``) are transparently converted to WAV using the
existing ffmpeg-based conversion utilities before recognition.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .conversion import convert_to_wav
from .utils import safe_stem_name

logger = logging.getLogger(__name__)

# File extensions ShazamIO generally handles without conversion.
SHAZAMIO_NATIVE_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".flac", ".wav", ".ogg", ".aac", ".wma"}
)


def _slugify(text: str) -> str:
    """Convert a string to a lowercase, hyphen-separated slug.

    Non-alphanumeric characters are replaced with hyphens and consecutive
    hyphens are collapsed.
    """
    cleaned = "".join(c if c.isalnum() or c.isspace() else "-" for c in text.lower())
    parts = [part for part in cleaned.split() if part]
    slug = "-".join(parts)
    # Collapse consecutive hyphens.
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "unknown"


def format_track_filename(artist: str | None, title: str | None) -> str:
    """Return a filename stem like ``artist-name_song-name``.

    Args:
        artist: Recognized artist name.
        title: Recognized track title.

    Returns:
        Filesystem-safe filename stem.

    Raises:
        ValueError: If both ``artist`` and ``title`` are empty or missing.
    """
    parts = [p for p in (_slugify(artist) if artist else None, _slugify(title) if title else None) if p]
    if not parts:
        raise ValueError("Cannot build track filename: artist and title are both missing")
    return "_".join(parts)


def _natural_sort_key(path: Path) -> list[str | int]:
    """Return a sort key that orders embedded numbers numerically."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def find_matching_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted files in ``directory`` matching ``pattern``.

    ``%%`` in the pattern matches one or more digits, so
    ``full_track_%%.aifc`` matches ``full_track_01.aifc``,
    ``full_track_2.aifc``, etc. Patterns without ``%%`` are matched with
    standard shell glob rules.

    Args:
        directory: Directory to scan.
        pattern: Filename pattern; ``%%`` acts as a digit wildcard.

    Returns:
        Sorted list of matching file paths.

    Raises:
        NotADirectoryError: If ``directory`` is not a directory.
    """
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if "%%" in pattern:
        parts = pattern.split("%%")
        regex = r"(\d+)".join(re.escape(part) for part in parts)
        files = [p for p in directory.iterdir() if p.is_file() and re.fullmatch(regex, p.name)]
    else:
        files = [p for p in directory.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern)]

    files.sort(key=_natural_sort_key)
    return files


def _extract_track_info(result: dict[str, Any]) -> dict[str, Any]:
    """Return a flattened subset of a ShazamIO recognize response.

    Args:
        result: Raw dictionary returned by ``Shazam.recognize_song``.

    Returns:
        Dictionary with ``title``, ``artist``, ``album``, ``genre``,
        ``shazam_url``, ``shazam_id`` and the original ``raw`` payload.
    """
    track = result.get("track", {}) if isinstance(result, dict) else {}
    album: str | None = None
    for section in track.get("sections", []):
        for meta in section.get("metadata", []):
            if meta.get("title") == "Album":
                album = meta.get("text")
                break
        if album:
            break

    return {
        "title": track.get("title"),
        "artist": track.get("subtitle"),
        "subtitle": track.get("subtitle"),
        "album": album,
        "genre": track.get("genres", {}).get("primary"),
        "shazam_url": track.get("url"),
        "shazam_id": track.get("key"),
        "raw": result,
    }


async def recognize_async(
    audio_path: Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Recognize a song with ShazamIO (async).

    ``.aifc`` files and any other formats outside
    :data:`SHAZAMIO_NATIVE_EXTENSIONS` are converted to WAV before recognition.

    Args:
        audio_path: Path to the input audio file.
        config: Optional pipeline config (used for ``ffmpeg_path`` when
            conversion is needed). Defaults are used when omitted.

    Returns:
        Recognition result containing ``title``, ``artist``, ``album`` and the
        raw ShazamIO response.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
        ImportError: If ``shazamio`` is not installed.
        RuntimeError: If recognition fails or no match is returned.
    """
    try:
        from shazamio import Shazam
    except ImportError as exc:
        raise ImportError(
            f"ShazamIO is required for song recognition ({exc}). "
            "Install it with: pip install shazamio"
        ) from exc

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    config = config or PipelineConfig()
    input_path = audio_path
    temp_dir: Path | None = None

    try:
        ext = audio_path.suffix.lower()
        if ext not in SHAZAMIO_NATIVE_EXTENSIONS:
            temp_dir = Path(tempfile.mkdtemp(prefix="reversescore_recognize_"))
            wav_path = temp_dir / f"{safe_stem_name(audio_path.stem)}.wav"
            input_path = convert_to_wav(audio_path, wav_path, config)
            logger.info("Converted %s -> %s for ShazamIO", audio_path, input_path)

        shazam = Shazam()
        result = await shazam.recognize_song(str(input_path))
        info = _extract_track_info(result)
        info["input_path"] = str(audio_path)
        info["recognized_path"] = str(input_path)

        if not info.get("title") and not info.get("artist"):
            logger.warning("ShazamIO did not identify any track for %s", audio_path)
            raise RuntimeError("No match found for the provided audio")

        return info
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def recognize_song(
    audio_path: Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Recognize a song with ShazamIO (sync wrapper).

    Args:
        audio_path: Path to the input audio file.
        config: Optional pipeline config for ffmpeg conversion settings.

    Returns:
        Recognition result containing ``title``, ``artist``, ``album`` and the
        raw ShazamIO response.
    """
    return asyncio.run(recognize_async(audio_path, config))

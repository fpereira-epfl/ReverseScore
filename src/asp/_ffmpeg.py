"""Shared FFmpeg/FFprobe helpers for audio processing commands."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .utils import run_command

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg/FFprobe operation fails."""


@dataclass(frozen=True)
class AudioInfo:
    """Basic audio stream metadata."""

    duration: float
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str | None = None
    sample_format: str | None = None
    bits_per_sample: int | None = None
    bit_rate: int | None = None
    format_name: str | None = None


def require_ffmpeg() -> tuple[str, str]:
    """Return paths to ffmpeg and ffprobe, raising if either is missing."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    missing: list[str] = []
    if ffmpeg is None:
        missing.append("ffmpeg")
    if ffprobe is None:
        missing.append("ffprobe")
    if missing:
        raise FFmpegError(
            f"missing required command{'s' if len(missing) > 1 else ''}: {', '.join(missing)}"
        )
    assert ffmpeg is not None and ffprobe is not None
    return ffmpeg, ffprobe


def run_ffmpeg(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the given arguments."""
    ffmpeg, _ = require_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-nostdin"] + list(args)
    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        return run_command(cmd, check=check)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise FFmpegError(f"ffmpeg failed: {stderr}") from exc
    except FileNotFoundError as exc:
        raise FFmpegError("ffmpeg not found in PATH") from exc


def run_ffprobe(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ffprobe with the given arguments."""
    _, ffprobe = require_ffmpeg()
    cmd = [ffprobe, "-v", "error"] + list(args)
    logger.debug("Running ffprobe: %s", " ".join(cmd))
    try:
        return run_command(cmd, check=check)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise FFmpegError(f"ffprobe failed: {stderr}") from exc
    except FileNotFoundError as exc:
        raise FFmpegError("ffprobe not found in PATH") from exc


def probe_audio(path: Path) -> AudioInfo:
    """Read audio metadata with ffprobe."""
    result = run_ffprobe(
        [
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration,bit_rate,format_long_name:stream=codec_name,sample_rate,channels,channel_layout,sample_fmt,bits_per_raw_sample,bits_per_sample",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
        fmt = data.get("format") or {}

        def _int(key: str) -> int | None:
            value = stream.get(key)
            if value is None or value == "N/A":
                return None
            return int(value)

        def _float(key: str) -> float | None:
            value = fmt.get(key)
            if value is None or value == "N/A":
                return None
            return float(value)

        bits = _int("bits_per_raw_sample") or _int("bits_per_sample") or None
        bit_rate = _int("bit_rate")
        duration = float(fmt.get("duration", 0))

        return AudioInfo(
            duration=duration,
            codec=str(stream.get("codec_name", "unknown")),
            sample_rate=int(stream.get("sample_rate", 0)),
            channels=int(stream.get("channels", 0)),
            channel_layout=stream.get("channel_layout"),
            sample_format=stream.get("sample_fmt"),
            bits_per_sample=bits,
            bit_rate=bit_rate,
            format_name=fmt.get("format_long_name"),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FFmpegError(f"could not parse ffprobe output for {path}") from exc


def has_encoder(encoder: str) -> bool:
    """Return True if ffmpeg includes the named audio encoder."""
    ffmpeg, _ = require_ffmpeg()
    try:
        result = run_command([ffmpeg, "-hide_banner", "-encoders"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    pattern = re.compile(r"^\s*A\S*\s+" + re.escape(encoder) + r"\s", re.MULTILINE)
    return bool(pattern.search(result.stdout))


def has_filter(filter_name: str) -> bool:
    """Return True if ffmpeg includes the named audio filter."""
    ffmpeg, _ = require_ffmpeg()
    try:
        result = run_command([ffmpeg, "-hide_banner", "-filters"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == filter_name:
            return True
    return False


def detect_silences(
    path: Path,
    gap: float,
    threshold_db: str,
    *,
    silence_duration: float | None = None,
) -> list[dict[str, float]]:
    """Return silence regions detected by FFmpeg's silencedetect filter.

    Each region is a dict with ``start``, ``end``, and ``duration`` keys.
    """
    threshold = threshold_db if threshold_db.lower().endswith("db") else f"{threshold_db}dB"
    duration_arg = gap if silence_duration is None else silence_duration
    result = run_ffmpeg(
        [
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={threshold}:duration={duration_arg:.9f}",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "FFmpeg silence detection failed")

    silences: list[dict[str, float]] = []
    pending_start: float | None = None
    start_re = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    end_re = re.compile(
        r"silence_end:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?).*?"
        r"silence_duration:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.DOTALL,
    )
    for line in result.stderr.splitlines():
        start_match = start_re.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
        end_match = end_re.search(line)
        if end_match:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            if pending_start is None:
                pending_start = max(0.0, end - duration)
            start = max(0.0, pending_start)
            end = max(start, end)
            silences.append({"start": start, "end": end, "duration": max(0.0, end - start)})
            pending_start = None

    silences.sort(key=lambda item: (item["start"], item["end"]))
    return silences


def temporary_output(near: Path, stem: str, suffix: str = ".tmp") -> Path:
    """Create a temporary file in the same directory as ``near``."""
    parent = near.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{stem}.", suffix=suffix, dir=parent)
    os.close(fd)
    return Path(name)

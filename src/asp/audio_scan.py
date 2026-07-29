"""Audio file analysis: levels, loudness, dynamics, and integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._ffmpeg import FFmpegError, probe_audio, run_ffmpeg


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel statistics from ffmpeg astats."""

    channel: str
    peak_db: str = "unknown"
    rms_db: str = "unknown"
    dc_offset: str = "unknown"


@dataclass(frozen=True)
class ScanResult:
    """Complete analysis result for an audio file."""

    path: Path
    info: Any
    file_size_bytes: int
    duration_seconds: float
    format_name: str | None
    codec_name: str | None
    codec_short: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    sample_format: str | None
    declared_bits: int | None
    measured_bits: str | None
    samples: str | None
    sample_peak_db: str | None
    sample_headroom_db: str | None
    true_peak_db: str | None
    true_peak_headroom_db: str | None
    rms_level_db: str | None
    rms_peak_db: str | None
    rms_trough_db: str | None
    integrated_loudness_lufs: str | None
    loudness_range_lu: str | None
    loudness_threshold_lufs: str | None
    dc_offset: str | None
    crest_factor_db: str | None
    channel_stats: list[ChannelStats] = field(default_factory=list)
    clipping_assessment: str = "UNKNOWN"


def _json_value(text: str, key: str) -> str:
    pattern = re.compile(rf'^\s*"{re.escape(key)}"\s*:\s*"([^"]*)"', re.MULTILINE)
    matches: list[str] = pattern.findall(text)
    return matches[-1] if matches else ""


def _astats_overall_value(text: str, label: str) -> str:
    overall = False
    wanted = f"] {label}:"
    for line in text.splitlines():
        if "] Overall" in line:
            overall = True
            continue
        if overall and wanted in line:
            value = line.split(wanted, 1)[1].strip()
            return value
    return ""


def _channel_stats(astats_text: str) -> list[ChannelStats]:
    stats: list[ChannelStats] = []
    current: dict[str, str] = {}
    current_channel = ""

    def flush() -> None:
        nonlocal current_channel, current
        if current_channel:
            stats.append(
                ChannelStats(
                    channel=current_channel,
                    peak_db=current.get("Peak level dB", "unknown"),
                    rms_db=current.get("RMS level dB", "unknown"),
                    dc_offset=current.get("DC offset", "unknown"),
                )
            )
        current_channel = ""
        current = {}

    channel_re = re.compile(r"\] Channel: (.+)$")
    for line in astats_text.splitlines():
        match = channel_re.search(line)
        if match:
            flush()
            current_channel = match.group(1).strip()
            continue
        if "] Overall" in line:
            flush()
            continue
        for label in ("Peak level dB", "RMS level dB", "DC offset"):
            prefix = f"] {label}:"
            if prefix in line:
                current[label] = line.split(prefix, 1)[1].strip()
    flush()
    return stats


def _format_bitrate(bit_rate: int | None) -> str:
    if bit_rate is None or bit_rate <= 0:
        return "unknown"
    return f"{bit_rate / 1000:.0f} kb/s"


def _safe_float(value: str | None) -> float | None:
    if value is None or value in ("", "N/A", "-inf", "inf"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _headroom_db(peak_db: str | None) -> str:
    peak = _safe_float(peak_db)
    if peak is None:
        return "unknown"
    return f"{max(0.0, 0.0 - peak):.2f}"


def _crest_factor(sample_peak_db: str | None, rms_db: str | None) -> str:
    peak = _safe_float(sample_peak_db)
    rms = _safe_float(rms_db)
    if peak is None or rms is None:
        return "unknown"
    return f"{peak - rms:.2f}"


def _clipping_assessment(sample_peak_db: str | None, true_peak_db: str | None) -> str:
    sample = _safe_float(sample_peak_db)
    truepeak = _safe_float(true_peak_db)
    if sample is None or truepeak is None:
        return "UNKNOWN"
    if sample > 0.01:
        return "INVALID / ABOVE DIGITAL FULL SCALE"
    if sample >= -0.01 and truepeak > 0.0:
        return "HIGH RISK: full-scale samples and true-peak overs"
    if sample >= -0.01:
        return "WARNING: samples reach digital full scale"
    if truepeak > 0.0:
        return "WARNING: inter-sample true-peak overs"
    if truepeak > -0.50:
        return "CAUTION: less than 0.5 dB true-peak headroom"
    if truepeak > -1.00:
        return "LOW HEADROOM: less than 1 dB true-peak headroom"
    return "NO DIGITAL PEAK OVERLOAD DETECTED"


def scan_audio(path: Path) -> ScanResult:
    """Analyse an audio file without modifying it."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FFmpegError(f"input file not found: {path}")

    info = probe_audio(path)
    if info.duration <= 0:
        raise FFmpegError("could not determine input duration")

    astats_result = run_ffmpeg(
        [
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "astats=metadata=0:reset=0",
            "-f",
            "null",
            "-",
        ]
    )
    loudness_result = run_ffmpeg(
        [
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "loudnorm=I=-23:LRA=7:TP=-2:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )

    astats = astats_result.stderr
    loudness = loudness_result.stderr

    sample_peak = _astats_overall_value(astats, "Peak level dB") or None
    rms_level = _astats_overall_value(astats, "RMS level dB") or None
    rms_peak = _astats_overall_value(astats, "RMS peak dB") or None
    rms_trough = _astats_overall_value(astats, "RMS trough dB") or None
    dc_offset = _astats_overall_value(astats, "DC offset") or None
    measured_bits = _astats_overall_value(astats, "Bit depth") or None
    samples = _astats_overall_value(astats, "Number of samples") or None

    true_peak = _json_value(loudness, "input_tp") or None
    integrated = _json_value(loudness, "input_i") or None
    lra = _json_value(loudness, "input_lra") or None
    threshold = _json_value(loudness, "input_thresh") or None

    return ScanResult(
        path=path,
        info=info,
        file_size_bytes=path.stat().st_size,
        duration_seconds=info.duration,
        format_name=info.format_name,
        codec_name=info.codec,
        codec_short=info.codec,
        sample_rate=info.sample_rate,
        channels=info.channels,
        channel_layout=info.channel_layout,
        sample_format=info.sample_format,
        declared_bits=info.bits_per_sample,
        measured_bits=measured_bits,
        samples=samples,
        sample_peak_db=sample_peak,
        sample_headroom_db=_headroom_db(sample_peak),
        true_peak_db=true_peak,
        true_peak_headroom_db=_headroom_db(true_peak),
        rms_level_db=rms_level,
        rms_peak_db=rms_peak,
        rms_trough_db=rms_trough,
        integrated_loudness_lufs=integrated,
        loudness_range_lu=lra,
        loudness_threshold_lufs=threshold,
        dc_offset=dc_offset,
        crest_factor_db=_crest_factor(sample_peak, rms_level),
        channel_stats=_channel_stats(astats),
        clipping_assessment=_clipping_assessment(sample_peak, true_peak),
    )

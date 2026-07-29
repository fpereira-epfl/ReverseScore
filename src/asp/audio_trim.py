"""Trim leading and trailing silence from an audio file."""

from __future__ import annotations

from pathlib import Path

from ._ffmpeg import FFmpegError, detect_silences, probe_audio, require_ffmpeg, run_ffmpeg

SUPPORTED_CODECS = {"pcm_s16be", "pcm_s24be", "pcm_s32be", "pcm_f32be", "pcm_f64be"}


def trim_silence(
    input_path: Path,
    *,
    min_silence: float = 1.5,
    threshold_db: str = "-45dB",
    padding: float = 0.020,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Trim edge silence from an audio file.

    Args:
        input_path: Source audio file.
        min_silence: Minimum edge silence duration in seconds.
        threshold_db: FFmpeg silencedetect threshold.
        padding: Audio retained on each side of a selected silence.
        output_path: Destination path. Defaults to ``INPUT_trimmed.aifc``.
        overwrite: Replace existing output.

    Returns:
        Path to the trimmed file.
    """
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".aif", ".aiff", ".aifc"}:
        raise FFmpegError("input must have an .aif, .aiff, or .aifc extension")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_trimmed.aifc"
    else:
        output_path = output_path.expanduser().resolve()

    if output_path.exists() and not overwrite:
        raise FFmpegError(f"output already exists: {output_path}")

    info = probe_audio(input_path)
    if info.duration <= 0:
        raise FFmpegError("could not determine input duration")

    threshold = threshold_db if threshold_db.lower().endswith("db") else f"{threshold_db}dB"
    silences = detect_silences(input_path, gap=min_silence, threshold_db=threshold)
    if not silences:
        raise FFmpegError("no silence regions met the current settings")

    leading_end: float | None = None
    trailing_start: float | None = None
    if silences and silences[0]["start"] <= 0.001:
        leading_end = silences[0]["end"]
    if silences and silences[-1]["end"] >= info.duration - 0.050:
        trailing_start = silences[-1]["start"]

    trim_start = max(0.0, leading_end - padding) if leading_end is not None else 0.0
    trim_end = (
        min(info.duration, trailing_start + padding)
        if trailing_start is not None
        else info.duration
    )

    if trim_start <= 0.000001 and trim_end >= info.duration - 0.000001:
        raise FFmpegError("detected silence does not touch either file edge; nothing to trim")

    new_duration = trim_end - trim_start
    if new_duration <= 0.1:
        raise FFmpegError("the proposed output would be empty or implausibly short")

    codec = info.codec if info.codec in SUPPORTED_CODECS else "pcm_f32be"
    run_ffmpeg(
        [
            "-y" if overwrite else "-n",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
            "-af",
            f"atrim=start={trim_start:.9f}:end={trim_end:.9f},asetpts=PTS-STARTPTS",
            "-c:a",
            codec,
            "-f",
            "aiff",
            str(output_path),
        ]
    )

    if not output_path.exists() or output_path.stat().st_size < 1000:
        output_path.unlink(missing_ok=True)
        raise FFmpegError("the generated output appears empty or invalid")

    return output_path

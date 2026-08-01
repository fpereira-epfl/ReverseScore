"""Change audio playback tempo using FFmpeg's rubberband filter."""

from __future__ import annotations

from pathlib import Path

from ._ffmpeg import FFmpegError, has_encoder, has_filter, probe_audio, require_ffmpeg, run_ffmpeg


def _format_settings_from_path(path: Path) -> tuple[str, str, str] | None:
    """Return (extension, muxer, codec) for a known audio output path."""
    ext = path.suffix.lower().lstrip(".")
    mapping: dict[str, tuple[str, str, str]] = {
        "m4a": ("m4a", "ipod", "aac_at"),
        "aac": ("aac", "adts", "aac_at"),
        "mp3": ("mp3", "mp3", "libmp3lame"),
        "flac": ("flac", "flac", "flac"),
        "wav": ("wav", "wav", "pcm_s24le"),
        "aiff": ("aiff", "aiff", "pcm_s24be"),
        "aif": ("aiff", "aiff", "pcm_s24be"),
        "aifc": ("aifc", "aiff", "pcm_s16be"),
    }
    return mapping.get(ext)


def change_tempo(
    input_path: Path,
    factor: float,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Stretch or compress tempo without changing pitch.

    Args:
        input_path: Source audio file.
        factor: Tempo multiplier. ``0.85`` slows to 85 %, ``1.25`` speeds up
            to 125 %.
        output_path: Destination path. Defaults to ``INPUT_{factor}x.EXT``
            beside the input, preserving the input container.
        overwrite: Replace an existing output file.

    Returns:
        Path to the tempo-adjusted file.

    Raises:
        FFmpegError: If the input is missing, the factor is invalid, the
            output format is unsupported, or FFmpeg fails.
    """
    require_ffmpeg()
    if not has_filter("rubberband"):
        raise FFmpegError(
            "this FFmpeg build does not include the rubberband filter; "
            "install FFmpeg built with librubberband"
        )
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")
    if factor <= 0:
        raise FFmpegError(f"tempo factor must be positive, got {factor}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_{factor:g}x{input_path.suffix}"
    else:
        output_path = output_path.expanduser().resolve()

    if output_path == input_path:
        raise FFmpegError("output path is the same as the input path")
    if output_path.exists() and not overwrite:
        raise FFmpegError(f"output already exists: {output_path}")

    info = probe_audio(input_path)
    if info.duration <= 0:
        raise FFmpegError("could not determine input duration")

    format_settings = _format_settings_from_path(output_path)
    if format_settings is None:
        raise FFmpegError(
            f"unsupported output extension '{output_path.suffix}'. "
            "Use one of: .m4a, .aac, .mp3, .flac, .wav, .aiff, .aifc"
        )

    extension, muxer, codec = format_settings
    if codec == "aac_at" and not has_encoder("aac_at"):
        raise FFmpegError("this FFmpeg build does not include the AudioToolbox AAC encoder aac_at")
    if codec == "libmp3lame" and not has_encoder("libmp3lame"):
        raise FFmpegError("this FFmpeg build does not include the libmp3lame MP3 encoder")

    args = [
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-filter:a",
        f"rubberband=tempo={factor:g}",
        "-map_metadata",
        "0",
        "-c:a",
        codec,
        "-f",
        muxer,
    ]
    if codec in {"aac_at", "libmp3lame"}:
        args.extend(["-b:a", "256k"])
    args.append(str(output_path))

    run_ffmpeg(args)

    if not output_path.exists() or output_path.stat().st_size < 1000:
        output_path.unlink(missing_ok=True)
        raise FFmpegError("the generated output appears empty or invalid")

    return output_path

"""Audio format conversion with optional loudness normalization and denoising."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ._console import format_duration, human_size
from ._ffmpeg import (
    AudioInfo,
    FFmpegError,
    has_encoder,
    has_filter,
    probe_audio,
    require_ffmpeg,
    run_ffmpeg,
    temporary_output,
)
from .utils import find_audio_files, safe_stem_name

logger = logging.getLogger(__name__)

DenoisePreset = Literal["none", "soft", "medium", "hard"]
OutputFormat = Literal["m4a", "aac", "mp3", "flac", "wav", "aiff", "aifc"]

_DENOISE_PRESETS: dict[DenoisePreset, str] = {
    "none": "",
    "soft": "adeclick=w=20:o=80:t=1.15,afftdn=nr=8:nf=-38",
    "medium": "adeclick=w=20:o=80:t=1.20,afftdn=nr=12:nf=-34",
    "hard": "adeclick=w=20:o=80:t=1.30,afftdn=nr=15:nf=-30",
}

_FORMAT_SETTINGS: dict[OutputFormat, tuple[str, str, str, bool]] = {
    # extension, muxer, codec, lossy
    "m4a": ("m4a", "ipod", "aac_at", True),
    "aac": ("aac", "adts", "aac_at", True),
    "mp3": ("mp3", "mp3", "libmp3lame", True),
    "flac": ("flac", "flac", "flac", False),
    "wav": ("wav", "wav", "pcm_s24le", False),
    "aiff": ("aiff", "aiff", "pcm_s24be", False),
    "aifc": ("aifc", "aiff", "pcm_s16be", False),
}

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".aac", ".mp3", ".flac", ".wav", ".aiff", ".aif", ".aifc", ".ogg", ".wma"}
)


@dataclass(frozen=True)
class ConversionResult:
    """Result of an audio conversion."""

    output_path: Path
    info: AudioInfo
    applied_gain_db: float | None
    measured_loudness_lufs: float | None
    measured_true_peak_dbtp: float | None


def _format_from_path(path: Path) -> OutputFormat | None:
    """Infer output format from a file extension."""
    ext = path.suffix.lower().lstrip(".")
    mapping: dict[str, OutputFormat] = {
        "m4a": "m4a",
        "aac": "aac",
        "mp3": "mp3",
        "flac": "flac",
        "wav": "wav",
        "aiff": "aiff",
        "aif": "aiff",
        "aifc": "aifc",
    }
    return mapping.get(ext)


def _parse_loudnorm_json(log_text: str) -> dict[str, str]:
    """Extract loudnorm JSON values from ffmpeg stderr."""
    # Find the JSON block emitted by loudnorm.
    start = log_text.find("{")
    end = log_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data: dict[str, str] = json.loads(log_text[start : end + 1])
        return data
    except json.JSONDecodeError:
        return {}


def convert_audio(
    input_path: Path,
    output_path: Path | None,
    *,
    output_format: OutputFormat = "aifc",
    bitrate: str = "256k",
    normalize: bool = False,
    target_loudness_lufs: float = -18.0,
    true_peak_ceiling_dbtp: float | None = None,
    denoise: DenoisePreset = "none",
    custom_filter: str = "",
    sample_rate: int | None = None,
    channels: int | None = None,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert an audio file to another format with optional restoration and normalization.

    Args:
        input_path: Source audio file.
        output_path: Explicit output path. If None, writes beside the input.
        output_format: Target format.
        bitrate: Lossy bitrate, e.g. ``256k``.
        normalize: Apply constant-gain loudness normalization.
        target_loudness_lufs: Target integrated loudness in LUFS.
        true_peak_ceiling_dbtp: True-peak ceiling in dBTP; defaults to -2 for lossy,
            -1 for lossless.
        denoise: Restoration preset.
        custom_filter: Override the audio filter chain entirely.
        sample_rate: Resample to this rate in Hz (None = keep source).
        channels: Remix to this channel count (None = keep source).
        overwrite: Replace an existing output file.

    Returns:
        Conversion result with metadata and applied gain.
    """
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")

    fmt = output_format.lower()
    if fmt not in _FORMAT_SETTINGS:
        raise FFmpegError(f"unsupported output format: {fmt}")
    extension, muxer, codec, lossy = _FORMAT_SETTINGS[cast(OutputFormat, fmt)]

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.{extension}"
    else:
        output_path = output_path.expanduser().resolve()

    if output_path == input_path:
        raise FFmpegError("output path is the same as the input path")
    if output_path.exists() and not overwrite:
        raise FFmpegError(f"output already exists: {output_path}")

    if codec == "aac_at" and not has_encoder("aac_at"):
        raise FFmpegError("this FFmpeg build does not include the AudioToolbox AAC encoder aac_at")
    if codec == "libmp3lame" and not has_encoder("libmp3lame"):
        raise FFmpegError("this FFmpeg build does not include libmp3lame")

    denoise_filter = _DENOISE_PRESETS.get(denoise, "")
    if custom_filter:
        denoise_filter = custom_filter
        denoise = "custom"  # type: ignore[assignment]

    for required in ("adeclick", "afftdn"):
        if required in denoise_filter and not has_filter(required):
            raise FFmpegError(f"this FFmpeg build does not include the {required} filter")

    if true_peak_ceiling_dbtp is None:
        true_peak_ceiling_dbtp = -2.0 if lossy else -1.0

    source_info = probe_audio(input_path)
    if source_info.duration <= 0:
        raise FFmpegError("could not determine input duration")

    tmp = temporary_output(output_path, input_path.stem, suffix=f".{extension}")
    filter_chain = denoise_filter
    applied_gain: float | None = None
    measured_loudness: float | None = None
    measured_true_peak: float | None = None

    try:
        if normalize:
            measurement_filter = denoise_filter + "," if denoise_filter else ""
            measurement_filter += (
                f"loudnorm=I={target_loudness_lufs}:LRA=20:"
                f"TP={true_peak_ceiling_dbtp}:print_format=json"
            )
            result = run_ffmpeg(
                [
                    "-nostats",
                    "-i",
                    str(input_path),
                    "-map",
                    "0:a:0",
                    "-af",
                    measurement_filter,
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
            )
            values = _parse_loudnorm_json(result.stderr)
            measured_loudness = float(values.get("input_i", 0))
            measured_true_peak = float(values.get("input_tp", 0))

            loudness_gain = target_loudness_lufs - measured_loudness
            peak_gain = true_peak_ceiling_dbtp - measured_true_peak
            applied_gain = min(loudness_gain, peak_gain)

            gain_filter = f"volume={applied_gain:.4f}dB"
            filter_chain = (filter_chain + "," if filter_chain else "") + gain_filter

        # Always overwrite the temporary output; the final destination is
        # protected by the existence check above.
        args = [
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
            "-vn",
            "-sn",
            "-dn",
        ]
        if filter_chain:
            args.extend(["-af", filter_chain])
        if sample_rate is not None:
            args.extend(["-ar", str(sample_rate)])
        if channels is not None:
            args.extend(["-ac", str(channels)])
        args.extend(["-c:a", codec])
        if lossy:
            args.extend(["-b:a", bitrate])
        if fmt == "m4a":
            args.extend(["-movflags", "+faststart"])
        args.extend(["-f", muxer, str(tmp)])

        run_ffmpeg(args)

        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise FFmpegError("FFmpeg produced an empty output file")

        output_info = probe_audio(tmp)
        if abs(output_info.duration - source_info.duration) > 1.0:
            raise FFmpegError("output duration differs unexpectedly from the source")

        if normalize:
            verify = run_ffmpeg(
                [
                    "-nostats",
                    "-i",
                    str(tmp),
                    "-map",
                    "0:a:0",
                    "-af",
                    f"loudnorm=I={target_loudness_lufs}:LRA=20:TP={true_peak_ceiling_dbtp}:print_format=json",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
            )
            verify_values = _parse_loudnorm_json(verify.stderr)
            final_tp = float(verify_values.get("input_tp", 0))
            if final_tp > true_peak_ceiling_dbtp + 0.1:
                raise FFmpegError(
                    f"encoded true peak {final_tp:.2f} dBTP exceeds the {true_peak_ceiling_dbtp} dBTP ceiling"
                )

        if output_path.exists() and not overwrite:
            raise FFmpegError(
                f"output appeared during conversion and will not be overwritten: {output_path}"
            )
        tmp.replace(output_path)

        return ConversionResult(
            output_path=output_path,
            info=output_info,
            applied_gain_db=applied_gain,
            measured_loudness_lufs=measured_loudness,
            measured_true_peak_dbtp=measured_true_peak,
        )
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def convert_audio_files(
    input_path: Path,
    output_path: Path | None,
    *,
    output_format: OutputFormat = "aifc",
    bitrate: str = "256k",
    normalize: bool = False,
    target_loudness_lufs: float = -18.0,
    true_peak_ceiling_dbtp: float | None = None,
    denoise: DenoisePreset = "none",
    custom_filter: str = "",
    sample_rate: int | None = None,
    channels: int | None = None,
    overwrite: bool = False,
    extensions: frozenset[str] | None = None,
) -> dict[str, ConversionResult]:
    """Convert one audio file or all audio files in a directory.

    Args:
        input_path: Source audio file or directory.
        output_path: Destination file or directory. If None, outputs are written
            beside the input(s).
        output_format: Target format.
        extensions: File extensions to include when scanning a directory.

    Returns:
        Mapping from original filename stem to conversion result.
    """
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        result = convert_audio(
            input_path,
            output_path,
            output_format=output_format,
            bitrate=bitrate,
            normalize=normalize,
            target_loudness_lufs=target_loudness_lufs,
            true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
            denoise=denoise,
            custom_filter=custom_filter,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
        return {input_path.stem: result}

    if not input_path.is_dir():
        raise FFmpegError(f"input not found: {input_path}")

    out_dir: Path
    if output_path is None:
        out_dir = input_path
    else:
        out_dir = output_path.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(input_path, extensions or _DEFAULT_EXTENSIONS)
    if not files:
        raise FFmpegError(f"no supported audio files found in {input_path}")

    results: dict[str, ConversionResult] = {}
    for file in files:
        out_file = out_dir / f"{safe_stem_name(file.stem)}.{_FORMAT_SETTINGS[output_format][0]}"
        result = convert_audio(
            file,
            out_file,
            output_format=output_format,
            bitrate=bitrate,
            normalize=normalize,
            target_loudness_lufs=target_loudness_lufs,
            true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
            denoise=denoise,
            custom_filter=custom_filter,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
        results[file.stem] = result

    return results


def summarize_conversion(result: ConversionResult) -> list[tuple[str, str]]:
    """Return key/value rows for CLI output."""
    rows: list[tuple[str, str]] = [
        ("File", str(result.output_path)),
        ("Codec", result.info.codec),
        ("Duration", format_duration(result.info.duration)),
        ("Size", human_size(result.output_path.stat().st_size)),
    ]
    if result.applied_gain_db is not None:
        rows.append(("Fixed gain", f"{result.applied_gain_db:+.4f} dB"))
    return rows

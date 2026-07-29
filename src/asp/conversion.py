"""Audio format conversion utilities using ffmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PipelineConfig
from .utils import ensure_dirs, find_program, run_command, safe_stem_name

logger = logging.getLogger(__name__)

# Input extensions ffmpeg can reliably decode to WAV.
SUPPORTED_INPUT_EXTENSIONS: set[str] = {".m4a", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".wav"}


def find_ffmpeg(config: PipelineConfig) -> Path:
    """Locate the ffmpeg executable specified in config."""
    return find_program(config.ffmpeg_path)


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    sample_rate: int = 44100,
    channels: int = 2,
    overwrite: bool = False,
) -> Path:
    """Convert an audio file to WAV using ffmpeg.

    Args:
        input_path: Path to the source audio file.
        output_path: Destination WAV file path.
        config: Pipeline configuration (used for ffmpeg_path).
        sample_rate: Target sample rate in Hz.
        channels: Number of audio channels.
        overwrite: Re-run conversion even if output exists.

    Returns:
        Path to the generated WAV file.

    Raises:
        FileNotFoundError: If the input file or ffmpeg executable is missing.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path.is_file() and not overwrite:
        logger.info("Reusing existing WAV: %s", output_path)
        return output_path

    ffmpeg = find_ffmpeg(config)
    ensure_dirs(output_path.parent)

    cmd: list[str] = [
        str(ffmpeg),
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    logger.info("Converting %s -> %s", input_path, output_path)
    result = run_command(cmd)
    if result.stdout:
        logger.debug("ffmpeg stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("ffmpeg stderr:\n%s", result.stderr)

    if not output_path.is_file():
        raise FileNotFoundError(f"ffmpeg did not produce expected WAV: {output_path}")
    return output_path


def convert_directory_to_wav(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    extensions: set[str] | None = None,
    sample_rate: int = 44100,
    channels: int = 2,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Convert all supported audio files in a directory to WAV.

    Args:
        input_dir: Directory containing source audio files.
        output_dir: Directory for generated WAV files.
        config: Pipeline configuration.
        extensions: File extensions to convert. Defaults to common formats.
        sample_rate: Target sample rate in Hz.
        channels: Number of audio channels.
        overwrite: Re-run conversion even if outputs exist.

    Returns:
        Mapping from original filename stem to output WAV path.
    """
    from .utils import find_audio_files

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ensure_dirs(output_dir)

    if extensions is None:
        extensions = SUPPORTED_INPUT_EXTENSIONS

    files = find_audio_files(input_dir, extensions)
    results: dict[str, Path] = {}
    for file in files:
        out_name = f"{safe_stem_name(file.stem)}.wav"
        output_path = output_dir / out_name
        convert_to_wav(
            file,
            output_path,
            config,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
        results[file.stem] = output_path
    return results

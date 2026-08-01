"""Source separation using Meta's demucs."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

from .config import PipelineConfig
from .utils import ensure_dirs, find_program, run_command, safe_stem_name

logger = logging.getLogger(__name__)

# Tango-relevant stem names. demucs default htdemucs emits: drums, bass, other, vocals.
# htdemucs_6s also emits guitar and piano. We map these to tango-friendly labels.
TANGO_STEM_MAP: dict[str, str] = {
    "other": "bandoneon_violin",
    "vocals": "voice",
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "piano": "piano",
}


def separate_stems(
    audio_path: Path,
    config: PipelineConfig,
    overwrite: bool = False,
    flat_output_dir: Path | None = None,
) -> dict[str, Path]:
    """Separate an audio file into stems using demucs.

    Args:
        audio_path: Path to the input audio file.
        config: Pipeline configuration.
        overwrite: Re-run separation even if stems already exist.
        flat_output_dir: If given, copy the resulting stems directly into this
            directory and return paths pointing there. demucs still writes to
            its normal nested working directory under ``config.separation_dir``,
            but callers of ``asp separate`` receive a clean, flat output folder.

    Returns:
        Mapping from stem label to WAV file path.

    Raises:
        FileNotFoundError: If the audio file or demucs executable is missing.
        subprocess.CalledProcessError: If demucs fails.
    """
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = config.separation_dir / safe_stem_name(audio_path.stem)
    ensure_dirs(out_dir)

    model_dir = out_dir / config.demucs_model
    if model_dir.exists() and not overwrite:
        logger.info("Reusing existing stems in %s", model_dir)
        stems = _collect_stems(model_dir, config.demucs_stems)
    else:
        demucs_exe = find_program("demucs")
        cmd: list[str] = [
            str(demucs_exe),
            "--name",
            config.demucs_model,
            "--out",
            str(out_dir),
        ]
        if config.demucs_segment is not None:
            cmd.extend(["--segment", str(config.demucs_segment)])
        if config.demucs_device:
            cmd.extend(["--device", config.demucs_device])
        cmd.append(str(audio_path))

        logger.info("Running demucs: %s", " ".join(cmd))
        result = run_command(cmd)
        if result.stdout:
            logger.debug("demucs stdout:\n%s", result.stdout)
        if result.stderr:
            logger.debug("demucs stderr:\n%s", result.stderr)

        stems = _collect_stems(model_dir, config.demucs_stems)

    if flat_output_dir is not None:
        flat_output_dir = flat_output_dir.expanduser().resolve()
        ensure_dirs(flat_output_dir)
        stems = _copy_stems_flat(stems, flat_output_dir)

    return stems


def _collect_stems(model_dir: Path, stems: Iterable[str]) -> dict[str, Path]:
    """Collect separated WAV stems from a demucs output directory."""
    stem_dir = model_dir / "audio" if (model_dir / "audio").is_dir() else model_dir
    result: dict[str, Path] = {}
    for stem in stems:
        wav = stem_dir / f"{stem}.wav"
        if not wav.is_file():
            # Older demucs layouts may nest under a track-name folder.
            candidates = list(stem_dir.rglob(f"{stem}.wav"))
            if candidates:
                wav = candidates[0]
        if wav.is_file():
            label = TANGO_STEM_MAP.get(stem, stem)
            result[label] = wav
        else:
            logger.warning("Expected stem not found: %s", wav)
    return result


def _copy_stems_flat(stems: dict[str, Path], flat_output_dir: Path) -> dict[str, Path]:
    """Copy stems into a single flat directory and return the new paths.

    Filenames are written as ``<label>.wav`` so the output folder is easy to
    inspect and consume by downstream tools such as ``asp stems-to-midi``.
    """
    flat_output_dir = flat_output_dir.expanduser().resolve()
    result: dict[str, Path] = {}
    for label, source in stems.items():
        destination = flat_output_dir / f"{label}.wav"
        shutil.copy2(source, destination)
        result[label] = destination
    return result


def separate_two_stems_vocals(
    audio_path: Path,
    config: PipelineConfig,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Separate audio into vocals and accompaniment.

    Useful for vocal tangos where the singer should be isolated before
    orchestral separation.

    Args:
        audio_path: Path to input audio.
        config: Pipeline configuration.
        overwrite: Re-run separation if outputs exist.

    Returns:
        Mapping with ``vocals`` and ``no_vocals`` stem paths.
    """
    audio_path = audio_path.expanduser().resolve()
    out_dir = config.separation_dir / safe_stem_name(audio_path.stem)
    ensure_dirs(out_dir)

    model_dir = out_dir / f"{config.demucs_model}_two_stems"
    if model_dir.exists() and not overwrite:
        logger.info("Reusing existing two-stem separation in %s", model_dir)
        return _collect_two_stems(model_dir)

    demucs_exe = find_program("demucs")
    cmd: list[str] = [
        str(demucs_exe),
        "--name",
        config.demucs_model,
        "--two-stems",
        "vocals",
        "--out",
        str(out_dir),
    ]
    if config.demucs_segment is not None:
        cmd.extend(["--segment", str(config.demucs_segment)])
    if config.demucs_device:
        cmd.extend(["--device", config.demucs_device])
    cmd.append(str(audio_path))

    logger.info("Running demucs (two-stems): %s", " ".join(cmd))
    run_command(cmd)
    return _collect_two_stems(model_dir)


def _collect_two_stems(model_dir: Path) -> dict[str, Path]:
    """Collect vocals/no_vocals stems from a two-stem demucs run."""
    stem_dir = model_dir / "audio" if (model_dir / "audio").is_dir() else model_dir
    result: dict[str, Path] = {}
    for stem in ("vocals", "no_vocals"):
        wav = stem_dir / f"{stem}.wav"
        if wav.is_file():
            result[stem] = wav
        else:
            candidates = list(stem_dir.rglob(f"{stem}.wav"))
            if candidates:
                result[stem] = candidates[0]
    return result

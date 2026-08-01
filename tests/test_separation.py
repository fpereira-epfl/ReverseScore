"""Tests for source separation."""

from pathlib import Path
from unittest import mock

from asp.config import PipelineConfig
from asp.separation import separate_stems


def test_separate_stems_reuses_existing(tmp_path: Path) -> None:
    audio = tmp_path / "tango.wav"
    audio.write_text("fake audio")

    config = PipelineConfig(output_dir=tmp_path / "out")
    model_dir = config.separation_dir / "tango" / config.demucs_model
    stem_dir = model_dir / "audio"
    stem_dir.mkdir(parents=True)
    (stem_dir / "bass.wav").touch()
    (stem_dir / "other.wav").touch()

    with mock.patch("asp.separation.run_command") as mock_run:
        stems = separate_stems(audio, config)
        mock_run.assert_not_called()

    assert "bass" in stems
    assert "bandoneon_violin" in stems


def test_separate_stems_flat_output_dir(tmp_path: Path) -> None:
    audio = tmp_path / "tango.wav"
    audio.write_text("fake audio")

    config = PipelineConfig(output_dir=tmp_path / "out")
    model_dir = config.separation_dir / "tango" / config.demucs_model
    stem_dir = model_dir / "audio"
    stem_dir.mkdir(parents=True)
    (stem_dir / "bass.wav").write_text("bass audio")
    (stem_dir / "other.wav").write_text("other audio")

    flat_dir = tmp_path / "flat"

    with mock.patch("asp.separation.run_command") as mock_run:
        stems = separate_stems(audio, config, flat_output_dir=flat_dir)
        mock_run.assert_not_called()

    assert stems == {
        "bass": flat_dir / "bass.wav",
        "bandoneon_violin": flat_dir / "bandoneon_violin.wav",
    }
    assert (flat_dir / "bass.wav").read_text() == "bass audio"
    assert (flat_dir / "bandoneon_violin.wav").read_text() == "other audio"

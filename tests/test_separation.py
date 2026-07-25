"""Tests for source separation."""

from pathlib import Path
from unittest import mock

from reversescore.config import PipelineConfig
from reversescore.separation import separate_stems


def test_separate_stems_reuses_existing(tmp_path: Path) -> None:
    audio = tmp_path / "tango.wav"
    audio.write_text("fake audio")

    config = PipelineConfig(output_dir=tmp_path / "out")
    model_dir = config.separation_dir / "tango" / config.demucs_model
    stem_dir = model_dir / "audio"
    stem_dir.mkdir(parents=True)
    (stem_dir / "bass.wav").touch()
    (stem_dir / "other.wav").touch()

    with mock.patch("reversescore.separation.run_command") as mock_run:
        stems = separate_stems(audio, config)
        mock_run.assert_not_called()

    assert "bass" in stems
    assert "bandoneon_violin" in stems

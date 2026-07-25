"""Tests for configuration."""

from pathlib import Path

import pytest
from reversescore.config import PipelineConfig


def test_default_config() -> None:
    config = PipelineConfig()
    assert config.demucs_model == "htdemucs"
    assert config.time_signature == "4/4"
    assert config.quantization_grid == 4
    assert config.split_bandoneon is True


def test_config_validates_thresholds() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(onset_threshold=1.5)


def test_config_resolves_output_dir(tmp_path: Path) -> None:
    config = PipelineConfig(output_dir=tmp_path / "foo")
    assert config.output_dir == tmp_path / "foo"
    assert config.separation_dir == tmp_path / "foo" / "stems"
    assert config.midi_dir == tmp_path / "foo" / "midi"
    assert config.score_dir == tmp_path / "foo" / "scores"

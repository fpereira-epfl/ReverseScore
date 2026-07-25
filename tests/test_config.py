"""Tests for configuration."""

from pathlib import Path

import pytest
from reversescore.config import PipelineConfig


@pytest.fixture
def isolated_config():
    """Temporarily disable config.yaml loading so defaults are tested."""
    original = PipelineConfig.model_config["yaml_file"]
    PipelineConfig.model_config["yaml_file"] = "/nonexistent/config.yaml"
    try:
        yield
    finally:
        PipelineConfig.model_config["yaml_file"] = original


def test_default_config(isolated_config: None) -> None:
    config = PipelineConfig()
    assert config.demucs_model == "htdemucs"
    assert config.time_signature == "4/4"
    assert config.quantization_grid == 4
    assert config.split_bandoneon is True
    assert config.ffmpeg_path == "ffmpeg"
    assert config.wav_output_dir == Path("./data/wav").resolve()
    assert config.wav_dir == config.wav_output_dir


def test_config_validates_thresholds(isolated_config: None) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(onset_threshold=1.5)


def test_config_accepts_guitar_hint(isolated_config: None) -> None:
    config = PipelineConfig(
        instrument_hints=["guitar"],
        instrument_exclusions=["drums"],
    )
    assert config.is_instrument_excluded("drums")
    assert not config.is_instrument_excluded("guitar")


def test_config_resolves_paths(isolated_config: None, tmp_path: Path) -> None:
    config = PipelineConfig(
        output_dir=tmp_path / "foo",
        input_dir=tmp_path / "bar",
        wav_output_dir=tmp_path / "baz",
    )
    assert config.output_dir == tmp_path / "foo"
    assert config.input_dir == tmp_path / "bar"
    assert config.wav_output_dir == tmp_path / "baz"
    assert config.separation_dir == tmp_path / "foo" / "stems"
    assert config.midi_dir == tmp_path / "foo" / "midi"
    assert config.score_dir == tmp_path / "foo" / "scores"
    assert config.wav_dir == tmp_path / "baz"


def test_config_loads_from_yaml(isolated_config: None, tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "input_dir: ./data/m4a\n"
        "output_dir: ./scores\n"
        "wav_output_dir: ./data/wav\n"
        "ffmpeg_path: /opt/ffmpeg\n"
        "time_signature: 2/4\n"
    )
    original_yaml_file = PipelineConfig.model_config["yaml_file"]
    PipelineConfig.model_config["yaml_file"] = str(yaml_path)
    try:
        config = PipelineConfig()
        assert config.ffmpeg_path == "/opt/ffmpeg"
        assert config.time_signature == "2/4"
        # Paths in config.yaml are resolved relative to the working directory.
        assert config.wav_output_dir == Path("./data/wav").resolve()
    finally:
        PipelineConfig.model_config["yaml_file"] = original_yaml_file

"""Tests for the ASP CLI commands."""

from pathlib import Path
from unittest import mock

from asp.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_separate_command_uses_flat_output_dir(tmp_path: Path) -> None:
    audio = tmp_path / "tango.wav"
    audio.write_text("fake audio")
    output_dir = tmp_path / "flat"

    fake_stems = {
        "bass": output_dir / "bass.wav",
        "bandoneon_violin": output_dir / "bandoneon_violin.wav",
    }

    with mock.patch("asp.separation.separate_stems") as mock_separate:
        mock_separate.return_value = fake_stems
        result = runner.invoke(app, ["separate", str(audio), "-o", str(output_dir)])

    assert result.exit_code == 0, result.output
    mock_separate.assert_called_once()
    _, kwargs = mock_separate.call_args
    assert kwargs["flat_output_dir"] == output_dir
    assert output_dir.name in result.output

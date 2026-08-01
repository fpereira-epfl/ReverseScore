"""Tests for ShazamIO-based song recognition."""

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest
from asp.cli import app
from asp.config import PipelineConfig
from asp.recognition import (
    find_matching_files,
    format_track_filename,
    recognize_async,
    recognize_song,
)
from typer.testing import CliRunner

runner = CliRunner()

# shazamio is an optional runtime dependency. Ensure it can be mocked during
# tests even when it is not installed in the current environment.
if "shazamio" not in sys.modules:
    sys.modules["shazamio"] = mock.MagicMock()


@pytest.fixture
def fake_shazam_response() -> dict:
    """Minimal successful ShazamIO response payload."""
    return {
        "track": {
            "title": "La Cumparsita",
            "subtitle": "Gerardo Matos Rodríguez",
            "url": "https://www.shazam.com/track/12345",
            "key": "12345",
            "genres": {"primary": "Tango"},
            "sections": [
                {
                    "metadata": [
                        {"title": "Album", "text": "Tangos Inmortales"},
                    ]
                }
            ],
        }
    }


@pytest.fixture
def mock_shazam(fake_shazam_response: dict):
    """Patch ``shazamio.Shazam`` with an async mock."""
    instance = mock.AsyncMock()
    instance.recognize_song.return_value = fake_shazam_response
    with mock.patch("shazamio.Shazam", return_value=instance) as constructor:
        yield constructor, instance


def test_recognize_song_m4a(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")

    _, instance = mock_shazam
    result = recognize_song(audio)

    assert result["title"] == "La Cumparsita"
    assert result["artist"] == "Gerardo Matos Rodríguez"
    assert result["album"] == "Tangos Inmortales"
    assert result["genre"] == "Tango"
    assert result["input_path"] == str(audio)
    instance.recognize_song.assert_awaited_once_with(str(audio))


def test_recognize_song_aifc_converts(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "song.aifc"
    audio.write_text("fake audio")

    def _fake_convert(
        input_path: Path, output_path: Path, cfg: PipelineConfig, **kwargs: object
    ) -> Path:
        output_path.touch()
        return output_path

    _, instance = mock_shazam
    config = PipelineConfig(output_dir=tmp_path)

    with mock.patch("asp.recognition.convert_to_wav", side_effect=_fake_convert) as mock_convert:
        result = recognize_song(audio, config)

    mock_convert.assert_called_once()
    args = mock_convert.call_args[0]
    assert args[0] == audio
    assert args[1].suffix == ".wav"
    instance.recognize_song.assert_awaited_once_with(str(args[1]))
    assert result["title"] == "La Cumparsita"


def test_recognize_song_no_match(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")

    _, instance = mock_shazam
    instance.recognize_song.return_value = {"track": {}}

    with pytest.raises(RuntimeError, match="No match found"):
        recognize_song(audio)


def test_recognize_song_missing_file(tmp_path: Path, mock_shazam: tuple) -> None:
    missing = tmp_path / "missing.m4a"

    with pytest.raises(FileNotFoundError):
        recognize_song(missing)


def test_recognize_async_is_async(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")

    _, instance = mock_shazam
    result = asyncio.run(recognize_async(audio))

    assert result["title"] == "La Cumparsita"
    instance.recognize_song.assert_awaited_once()


def test_format_track_filename() -> None:
    assert (
        format_track_filename("Gerardo Matos Rodríguez", "La Cumparsita")
        == "gerardo-matos-rodríguez_la-cumparsita"
    )
    assert format_track_filename(None, "La Cumparsita") == "la-cumparsita"
    assert format_track_filename("Donato", None) == "donato"


def test_format_track_filename_missing() -> None:
    with pytest.raises(ValueError, match="artist and title are both missing"):
        format_track_filename(None, None)


def test_cli_identify_rename(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "full_track_01.m4a"
    audio.write_text("fake audio")

    result = runner.invoke(app, ["identify", str(audio), "--rename"])
    assert result.exit_code == 0, result.output

    expected_name = "gerardo-matos-rodríguez_la-cumparsita.m4a"
    assert (tmp_path / expected_name).is_file()
    assert not audio.exists()
    assert expected_name in result.output


def test_cli_identify_rename_collision(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "full_track_01.m4a"
    audio.write_text("fake audio")
    (tmp_path / "gerardo-matos-rodríguez_la-cumparsita.m4a").touch()

    result = runner.invoke(app, ["identify", str(audio), "--rename"])
    assert result.exit_code == 0, result.output

    expected_name = "gerardo-matos-rodríguez_la-cumparsita_1.m4a"
    assert (tmp_path / expected_name).is_file()
    assert not audio.exists()
    assert expected_name in result.output


def test_find_matching_files_digit_wildcard(tmp_path: Path) -> None:
    (tmp_path / "full_track_01.aifc").touch()
    (tmp_path / "full_track_2.aifc").touch()
    (tmp_path / "full_track_abc.aifc").touch()
    (tmp_path / "other.aifc").touch()

    files = find_matching_files(tmp_path, "full_track_%%.aifc")
    names = [p.name for p in files]

    assert names == ["full_track_01.aifc", "full_track_2.aifc"]


def test_find_matching_files_glob(tmp_path: Path) -> None:
    (tmp_path / "song_01.m4a").touch()
    (tmp_path / "song_02.m4a").touch()
    (tmp_path / "notes.txt").touch()

    files = find_matching_files(tmp_path, "song_*.m4a")
    assert len(files) == 2


def test_find_matching_files_natural_sort(tmp_path: Path) -> None:
    (tmp_path / "full_track_10.aifc").touch()
    (tmp_path / "full_track_2.aifc").touch()
    (tmp_path / "full_track_1.aifc").touch()

    files = find_matching_files(tmp_path, "full_track_%%.aifc")
    assert [p.name for p in files] == [
        "full_track_1.aifc",
        "full_track_2.aifc",
        "full_track_10.aifc",
    ]


def test_cli_identify_folder_rename(tmp_path: Path, mock_shazam: tuple) -> None:
    (tmp_path / "full_track_01.m4a").write_text("fake")
    (tmp_path / "full_track_02.m4a").write_text("fake")

    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "full_track_%%.m4a",
            "--rename",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    assert not (tmp_path / "full_track_01.m4a").exists()
    assert not (tmp_path / "full_track_02.m4a").exists()
    assert (tmp_path / "gerardo-matos-rodríguez_la-cumparsita.m4a").is_file()
    assert (tmp_path / "gerardo-matos-rodríguez_la-cumparsita_1.m4a").is_file()


def test_cli_identify_folder_delay(tmp_path: Path, mock_shazam: tuple) -> None:
    (tmp_path / "full_track_01.m4a").write_text("fake")
    (tmp_path / "full_track_02.m4a").write_text("fake")

    with mock.patch("asp.cli.time.sleep") as mock_sleep:
        result = runner.invoke(
            app,
            [
                "identify",
                "--input-dir",
                str(tmp_path),
                "--pattern",
                "full_track_%%.m4a",
                "--delay",
                "1.5",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_sleep.assert_called_once_with(1.5)


def test_cli_identify_folder_no_matches(tmp_path: Path, mock_shazam: tuple) -> None:
    result = runner.invoke(
        app,
        ["identify", "--input-dir", str(tmp_path), "--pattern", "full_track_%%.m4a"],
    )
    assert result.exit_code == 1, result.output
    assert "No files matching" in result.output


def test_cli_identify_audio_and_folder_error(tmp_path: Path, mock_shazam: tuple) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")

    result = runner.invoke(app, ["identify", str(audio), "--input-dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "not both" in result.output


def test_cli_identify_cache_persists_across_runs(tmp_path: Path, mock_shazam: tuple) -> None:
    (tmp_path / "track_01.m4a").write_text("fake")
    (tmp_path / "track_02.m4a").write_text("fake")

    _, instance = mock_shazam
    instance.recognize_song.reset_mock()

    # First run: identify both files and create the cache.
    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*.m4a",
            "--use-cache",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert instance.recognize_song.call_count == 2
    assert (tmp_path / "local_cache.json").is_file()

    # Second run: cache should be reused, no extra recognition calls.
    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*.m4a",
            "--use-cache",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert instance.recognize_song.call_count == 2
    assert "Skipping track_01.m4a (cached)" in result.output
    assert "Skipping track_02.m4a (cached)" in result.output

    # Add a third file and run again: only the new file should be identified,
    # while the original two remain cached.
    (tmp_path / "track_03.m4a").write_text("fake")
    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*.m4a",
            "--use-cache",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert instance.recognize_song.call_count == 3
    assert "Skipping track_01.m4a (cached)" in result.output
    assert "Skipping track_02.m4a (cached)" in result.output
    assert "Identifying track_03.m4a" in result.output


def test_cli_identify_always_writes_cache_even_without_flag(
    tmp_path: Path, mock_shazam: tuple
) -> None:
    (tmp_path / "track_01.m4a").write_text("fake")

    _, instance = mock_shazam
    instance.recognize_song.reset_mock()

    # Run without --use-cache: it should still create/update the cache.
    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*.m4a",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert instance.recognize_song.call_count == 1
    assert (tmp_path / "local_cache.json").is_file()

    # Run again with --use-cache: the previously created cache should be used.
    instance.recognize_song.reset_mock()
    result = runner.invoke(
        app,
        [
            "identify",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*.m4a",
            "--use-cache",
            "--delay",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert instance.recognize_song.call_count == 0
    assert "Skipping track_01.m4a (cached)" in result.output

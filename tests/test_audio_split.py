"""Tests for track splitting logic."""

from pathlib import Path
from unittest import mock

import pytest
from asp.audio_split import Silence, _parse_time, _select_boundaries, _split_by_added_silence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("45", 45.0),
        ("0:45", 45.0),
        ("1:23", 83.0),
        ("1:05:30", 3930.0),
        ("10:00:00", 36000.0),
        ("0:02:30.5", 150.5),
    ],
)
def test_parse_time_formats(text: str, expected: float) -> None:
    assert _parse_time(text) == pytest.approx(expected)


def test_select_boundaries_prefers_hints() -> None:
    """Hints near candidates should bias boundary selection."""
    # Two candidate silences; only one boundary needed (2 tracks).
    # Candidate A is at 40s; candidate B is exactly at the 60s target. Without a
    # hint candidate B is preferred because it produces a perfect 60s track. A
    # hint at 40s should flip the selection to candidate A.
    candidates = [
        Silence(start=38.0, end=42.0, duration=4.0),  # midpoint 40s
        Silence(start=58.0, end=62.0, duration=4.0),  # midpoint 60s
    ]

    selected_no_hint = _select_boundaries(
        candidates,
        track_count=2,
        content_start=0.0,
        content_end=120.0,
        total_duration=120.0,
        minimum=30.0,
        maximum=90.0,
        target=60.0,
        padding=0.0,
    )

    selected_with_hint = _select_boundaries(
        candidates,
        track_count=2,
        content_start=0.0,
        content_end=120.0,
        total_duration=120.0,
        minimum=30.0,
        maximum=90.0,
        target=60.0,
        padding=0.0,
        hints=[40.0],
    )

    assert selected_no_hint[0].start == 58.0
    assert selected_with_hint[0].start == 38.0


def test_split_by_added_silence_filters_and_trims(tmp_path: Path) -> None:
    """Added-silence mode builds tracks between long silences and trims them."""
    audio = tmp_path / "song.aifc"
    audio.write_text("fake audio")
    out_dir = tmp_path / "out"

    info = mock.MagicMock()
    info.duration = 250.0
    info.codec = "pcm_s16be"

    # Two 9-second silences in a 250s file (added_silence=10 => min duration 9).
    # Expected tracks:
    #   0-70   (content before first silence)
    #   79-180 (content between silences)
    #   189-250 (content after second silence)
    # The middle segment is dropped because it is too quiet.
    detected = [
        {"start": 70.0, "end": 79.0, "duration": 9.0},
        {"start": 180.0, "end": 189.0, "duration": 9.0},
    ]

    def _fake_run_ffmpeg(args: list[str], *, check: bool = True) -> mock.MagicMock:
        # The output path is the last argument to ffmpeg.
        Path(args[-1]).write_bytes(b"x" * 1500)
        return mock.MagicMock(returncode=0)

    with (
        mock.patch("asp.audio_split.probe_audio", return_value=info),
        mock.patch("asp.audio_split.detect_silences", return_value=detected),
        mock.patch("asp.audio_split._segment_max_volume", side_effect=[-30.0, -60.0, -25.0]),
        mock.patch("asp.audio_split.run_ffmpeg", side_effect=_fake_run_ffmpeg) as mock_run,
    ):
        result = _split_by_added_silence(
            audio,
            added_silence=10,
            output_dir=out_dir,
            prefix=None,
            overwrite=False,
            dry_run=False,
        )

    assert len(result.tracks) == 2
    assert result.tracks[0].start == 0.0
    assert result.tracks[0].end == 70.0
    assert result.tracks[1].start == 189.0
    assert result.tracks[1].end == 250.0
    assert mock_run.call_count == 2

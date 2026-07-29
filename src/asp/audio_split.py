"""Split a long recording into tracks using detected silences."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._ffmpeg import FFmpegError, detect_silences, probe_audio, require_ffmpeg, run_ffmpeg


@dataclass(frozen=True)
class SplitResult:
    """Result of a track-split operation."""

    tracks: list[Track]
    output_paths: list[Path]


@dataclass(frozen=True)
class Silence:
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class Track:
    number: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _parse_time(value: str) -> float:
    text = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        seconds = float(text)
    else:
        match = re.fullmatch(r"(?:(\d+):)?([0-5]?\d):([0-5]\d(?:\.\d+)?)", text)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2))
            seconds_part = float(match.group(3))
            seconds = hours * 3600 + minutes * 60 + seconds_part
        else:
            match = re.fullmatch(r"(\d+):([0-5]\d(?:\.\d+)?)", text)
            if not match:
                raise ValueError(f"invalid time {value!r}; use MM:SS, HH:MM:SS, or seconds")
            minutes = int(match.group(1))
            seconds_part = float(match.group(2))
            seconds = minutes * 60 + seconds_part
    if seconds <= 0:
        raise ValueError("time must be greater than zero")
    return seconds


def _format_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
    return f"{minutes:d}:{whole_seconds:02d}.{millis:03d}"


def _identify_content_edges(
    silences: Sequence[Silence],
    total_duration: float,
    edge_guard: float,
) -> tuple[float, float]:
    content_start = 0.0
    content_end = total_duration
    for silence in silences:
        if silence.start <= edge_guard:
            content_start = max(content_start, silence.end)
        else:
            break
    for silence in reversed(silences):
        if silence.end >= total_duration - edge_guard:
            content_end = min(content_end, silence.start)
        else:
            break
    if content_end <= content_start:
        return 0.0, total_duration
    return content_start, content_end


def _internal_candidates(
    silences: Sequence[Silence],
    content_start: float,
    content_end: float,
) -> list[Silence]:
    return [
        silence
        for silence in silences
        if silence.start > content_start
        and silence.end < content_end
        and silence.end > silence.start
    ]


def _boundary_track_duration(
    previous: Silence | None,
    current: Silence,
    content_start: float,
    padding: float,
) -> float:
    start = max(0.0, previous.end - padding) if previous else max(0.0, content_start - padding)
    end = current.start + padding
    return end - start


def _final_track_duration(
    previous: Silence | None,
    content_start: float,
    content_end: float,
    padding: float,
    total_duration: float,
) -> float:
    start = max(0.0, previous.end - padding) if previous else max(0.0, content_start - padding)
    end = min(total_duration, content_end + padding)
    return end - start


def _track_score(duration: float, target: float, silence: Silence | None) -> float:
    deviation = (duration - target) / max(target, 1.0)
    duration_penalty = deviation * deviation * 100.0
    silence_reward = 5.0 * math.log1p(max(0.0, silence.duration)) if silence else 0.0
    return silence_reward - duration_penalty


def _select_boundaries(
    candidates: Sequence[Silence],
    track_count: int,
    content_start: float,
    content_end: float,
    total_duration: float,
    minimum: float,
    maximum: float,
    target: float,
    padding: float,
) -> list[Silence]:
    boundaries_needed = track_count - 1
    if boundaries_needed == 0:
        duration = _final_track_duration(None, content_start, content_end, padding, total_duration)
        if not minimum <= duration <= maximum:
            raise FFmpegError(
                f"the only track would be {_format_time(duration)}, outside "
                f"{_format_time(minimum)}–{_format_time(maximum)}"
            )
        return []

    if len(candidates) < boundaries_needed:
        raise FFmpegError(
            f"need {boundaries_needed} internal silence gaps, but only {len(candidates)} were detected"
        )

    state_scores: list[dict[int, float]] = [{} for _ in range(boundaries_needed + 1)]
    parents: list[dict[int, int | None]] = [{} for _ in range(boundaries_needed + 1)]

    for index, silence in enumerate(candidates):
        duration = _boundary_track_duration(None, silence, content_start, padding)
        if minimum <= duration <= maximum:
            state_scores[1][index] = _track_score(duration, target, silence)
            parents[1][index] = None

    for chosen_count in range(2, boundaries_needed + 1):
        for current_index, current in enumerate(candidates):
            best_score: float | None = None
            best_previous: int | None = None
            for previous_index, previous_score in state_scores[chosen_count - 1].items():
                if previous_index >= current_index:
                    continue
                previous = candidates[previous_index]
                duration = _boundary_track_duration(previous, current, content_start, padding)
                if not minimum <= duration <= maximum:
                    continue
                score = previous_score + _track_score(duration, target, current)
                if best_score is None or score > best_score:
                    best_score = score
                    best_previous = previous_index
            if best_score is not None:
                state_scores[chosen_count][current_index] = best_score
                parents[chosen_count][current_index] = best_previous

    best_final_score: float | None = None
    best_final_index: int | None = None
    for previous_index, previous_score in state_scores[boundaries_needed].items():
        previous = candidates[previous_index]
        duration = _final_track_duration(
            previous, content_start, content_end, padding, total_duration
        )
        if not minimum <= duration <= maximum:
            continue
        score = previous_score + _track_score(duration, target, None)
        if best_final_score is None or score > best_final_score:
            best_final_score = score
            best_final_index = previous_index

    if best_final_index is None:
        raise FFmpegError(
            f"no combination of detected silences can create exactly {track_count} tracks "
            f"while keeping every track between {_format_time(minimum)} and {_format_time(maximum)}"
        )

    selected_indices: list[int] = []
    reverse_index: int | None = best_final_index
    for chosen_count in range(boundaries_needed, 0, -1):
        if reverse_index is None:
            raise FFmpegError("internal boundary-selection error")
        selected_indices.append(reverse_index)
        reverse_index = parents[chosen_count][reverse_index]
    selected_indices.reverse()
    return [candidates[index] for index in selected_indices]


def _build_tracks(
    selected: Sequence[Silence],
    content_start: float,
    content_end: float,
    total_duration: float,
    padding: float,
) -> list[Track]:
    starts = [max(0.0, content_start - padding)]
    ends: list[float] = []
    for silence in selected:
        ends.append(min(total_duration, silence.start + padding))
        starts.append(max(0.0, silence.end - padding))
    ends.append(min(total_duration, content_end + padding))

    tracks = [
        Track(number=index + 1, start=start, end=end)
        for index, (start, end) in enumerate(zip(starts, ends))
    ]
    for track in tracks:
        if track.end <= track.start:
            raise FFmpegError(
                f"track {track.number} has invalid boundaries {track.start:.6f}–{track.end:.6f}"
            )
    return tracks


def _output_codec(source_codec: str) -> str:
    supported = {
        "pcm_s8",
        "pcm_s16be",
        "pcm_s24be",
        "pcm_s32be",
        "pcm_f32be",
        "pcm_f64be",
    }
    return source_codec if source_codec in supported else "pcm_f32be"


def _output_extension(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    return suffix if suffix in {".aif", ".aiff", ".aifc"} else ".aifc"


def split_recording(
    input_path: Path,
    track_count: int,
    *,
    min_track: float,
    max_track: float,
    target_track: float | None = None,
    gap: float = 0.5,
    threshold_db: str = "-40dB",
    padding: float = 0.02,
    edge_guard: float = 10.0,
    output_dir: Path | None = None,
    prefix: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> SplitResult:
    """Split a recording into tracks using silence detection and dynamic programming."""
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".aif", ".aiff", ".aifc"}:
        raise FFmpegError("input must have an .aif, .aiff, or .aifc extension")

    if track_count <= 0:
        raise FFmpegError("--tracks must be greater than zero")
    if min_track > max_track:
        raise FFmpegError("--min-track must not exceed --max-track")

    info = probe_audio(input_path)
    absolute_minimum = track_count * min_track
    if info.duration < absolute_minimum:
        raise FFmpegError(
            f"input duration {_format_time(info.duration)} is shorter than "
            f"{track_count} × {_format_time(min_track)}"
        )

    raw_silences = detect_silences(input_path, gap=gap, threshold_db=threshold_db)
    silences = [Silence(**region) for region in raw_silences]
    if not silences:
        raise FFmpegError(
            "FFmpeg detected no silence regions; try a shorter gap or a less-negative threshold"
        )

    content_start, content_end = _identify_content_edges(silences, info.duration, edge_guard)
    candidates = _internal_candidates(silences, content_start, content_end)
    usable_duration = content_end - content_start
    target = target_track or usable_duration / track_count
    target = min(max(target, min_track), max_track)

    selected = _select_boundaries(
        candidates,
        track_count,
        content_start,
        content_end,
        info.duration,
        min_track,
        max_track,
        target,
        padding,
    )
    tracks = _build_tracks(selected, content_start, content_end, info.duration, padding)

    if len(tracks) != track_count:
        raise FFmpegError(
            f"internal error: produced {len(tracks)} tracks for {track_count} requested"
        )

    if dry_run:
        return SplitResult(tracks=tracks, output_paths=[])

    out_dir = output_dir.expanduser().resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = prefix or f"{input_path.stem}_track"
    extension = _output_extension(input_path)
    width = max(2, len(str(track_count)))
    paths = [
        out_dir / f"{base}_{number:0{width}d}{extension}" for number in range(1, track_count + 1)
    ]

    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FFmpegError(
            "output files already exist; use --overwrite to replace them:\n"
            + "\n".join(f"  {path}" for path in existing[:10])
        )

    codec = _output_codec(info.codec)
    written: list[Path] = []
    try:
        for track, path in zip(tracks, paths):
            run_ffmpeg(
                [
                    "-loglevel",
                    "error",
                    "-y" if overwrite else "-n",
                    "-i",
                    str(input_path),
                    "-map",
                    "0:a:0",
                    "-map_metadata",
                    "0",
                    "-af",
                    f"atrim=start={track.start:.9f}:end={track.end:.9f},asetpts=PTS-STARTPTS",
                    "-c:a",
                    codec,
                    "-f",
                    "aiff",
                    str(path),
                ]
            )
            if not path.exists() or path.stat().st_size < 1000:
                path.unlink(missing_ok=True)
                raise FFmpegError(f"{path.name} is missing, empty, or invalid")
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise

    return SplitResult(tracks=tracks, output_paths=written)

"""Split a long recording into tracks using detected silences."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._ffmpeg import FFmpegError, detect_silences, probe_audio, require_ffmpeg, run_ffmpeg


class SplitError(FFmpegError):
    """Splitting failed with attached diagnostic data for user guidance."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


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


def _silence_diagnostics(
    silences: Sequence[Silence],
    candidates: Sequence[Silence],
    track_count: int,
    min_track: float,
    max_track: float,
    gap: float,
    threshold_db: str,
    content_start: float,
    content_end: float,
    total_duration: float,
) -> dict[str, object]:
    """Build a dictionary of human-readable split diagnostics."""
    boundaries_needed = track_count - 1
    usable_duration = content_end - content_start
    target = min(max(usable_duration / track_count, min_track), max_track)

    stats: dict[str, object] = {
        "total_duration": _format_time(total_duration),
        "content_duration": _format_time(usable_duration),
        "requested_tracks": track_count,
        "boundaries_needed": boundaries_needed,
        "min_track": _format_time(min_track),
        "max_track": _format_time(max_track),
        "target_track": _format_time(target),
        "threshold_db": threshold_db,
        "gap": gap,
        "total_silences": len(silences),
        "usable_candidates": len(candidates),
    }

    if candidates:
        durations = [s.duration for s in candidates]
        gaps = [
            candidates[i + 1].start - candidates[i].end
            for i in range(len(candidates) - 1)
        ]
        stats["shortest_silence"] = min(durations)
        stats["longest_silence"] = max(durations)
        stats["average_silence"] = sum(durations) / len(durations)
        if gaps:
            stats["shortest_gap"] = min(gaps)
            stats["longest_gap"] = max(gaps)
            stats["average_gap"] = sum(gaps) / len(gaps)

    # Determine likely cause and recommendations.
    recommendations: list[str] = []
    if not candidates:
        recommendations.append(
            "No usable internal silences were found. The threshold may be too strict, "
            "or the recording may have no clear gaps."
        )
    elif len(candidates) < boundaries_needed:
        recommendations.append(
            f"Detected {len(candidates)} usable silence(s), but {boundaries_needed} "
            f"boundaries are needed for {track_count} tracks."
        )
    else:
        recommendations.append(
            f"Detected {len(candidates)} usable silence(s), but no combination keeps "
            f"every track within {_format_time(min_track)}–{_format_time(max_track)}."
        )

    # Threshold guidance.
    try:
        threshold_value = float(threshold_db.lower().rstrip("db"))
    except ValueError:
        threshold_value = -40.0

    if len(candidates) < boundaries_needed:
        if threshold_value <= -50.0:
            recommendations.append(
                "Threshold is already very sensitive (-50 dB). Try a shorter --gap "
                f"(currently {gap}s) to detect shorter silences, or reduce --tracks."
            )
        else:
            recommendations.append(
                "Try a more sensitive threshold (less negative, e.g. -30dB or -24dB) "
                f"or a shorter --gap (currently {gap}s) to detect more silences."
            )
    elif len(candidates) >= boundaries_needed:
        # Enough silences exist, but lengths don't fit. Suggest relaxing min/max.
        recommendations.append(
            "The silences are in the wrong places for the requested track count. "
            "Try widening --min-track / --max-track, or check whether --tracks matches the recording."
        )

    # Check total duration feasibility.
    if total_duration < track_count * min_track:
        recommendations.append(
            f"The recording is too short for {track_count} tracks of at least "
            f"{_format_time(min_track)} each. Reduce --tracks or --min-track."
        )

    stats["recommendations"] = recommendations
    return stats


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


def _track_score(
    duration: float, target: float, silence: Silence | None, hint_bonus: float = 0.0
) -> float:
    deviation = (duration - target) / max(target, 1.0)
    duration_penalty = deviation * deviation * 100.0
    silence_reward = 5.0 * math.log1p(max(0.0, silence.duration)) if silence else 0.0
    return silence_reward - duration_penalty + hint_bonus


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
    hints: Sequence[float] | None = None,
    hint_tolerance: float = 5.0,
) -> list[Silence]:
    boundaries_needed = track_count - 1
    if boundaries_needed == 0:
        duration = _final_track_duration(None, content_start, content_end, padding, total_duration)
        if not minimum <= duration <= maximum:
            raise SplitError(
                f"the only track would be {_format_time(duration)}, outside "
                f"{_format_time(minimum)}–{_format_time(maximum)}"
            )
        return []

    if len(candidates) < boundaries_needed:
        raise SplitError(
            f"need {boundaries_needed} internal silence gaps, but only {len(candidates)} were detected"
        )

    hint_bonuses = [0.0] * len(candidates)
    if hints:
        for hint in hints:
            for index, candidate in enumerate(candidates):
                # Use the silence midpoint as the boundary anchor.
                midpoint = (candidate.start + candidate.end) / 2.0
                distance = abs(midpoint - hint)
                if distance <= hint_tolerance:
                    # Closer hints get a larger bonus; max bonus at exact match.
                    hint_bonuses[index] += 100.0 * (1.0 - distance / hint_tolerance)

    state_scores: list[dict[int, float]] = [{} for _ in range(boundaries_needed + 1)]
    parents: list[dict[int, int | None]] = [{} for _ in range(boundaries_needed + 1)]

    for index, silence in enumerate(candidates):
        duration = _boundary_track_duration(None, silence, content_start, padding)
        if minimum <= duration <= maximum:
            state_scores[1][index] = _track_score(
                duration, target, silence, hint_bonus=hint_bonuses[index]
            )
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
                score = previous_score + _track_score(
                    duration, target, current, hint_bonus=hint_bonuses[current_index]
                )
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
        raise SplitError(
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


def _snap_hint_to_silence(
    hint: float, silences: Sequence[Silence], tolerance: float = 5.0
) -> Silence:
    """Return the detected silence closest to ``hint``, or a synthetic one."""
    best: Silence | None = None
    best_distance = float("inf")
    for silence in silences:
        midpoint = (silence.start + silence.end) / 2.0
        distance = abs(midpoint - hint)
        if distance <= tolerance and distance < best_distance:
            best = silence
            best_distance = distance
    if best is not None:
        return best
    # No nearby silence: create a tiny synthetic silence centred on the hint.
    half = 0.001
    return Silence(start=hint - half, end=hint + half, duration=half * 2)


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


def _segment_max_volume(path: Path, start: float, end: float) -> float:
    """Return the max volume in dB for an audio segment using volumedetect."""
    result = run_ffmpeg(
        [
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"atrim=start={start:.9f}:end={end:.9f},volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise FFmpegError(f"volume detection failed: {result.stderr.strip()}")
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    if not match:
        raise FFmpegError("could not parse max_volume from ffmpeg output")
    return float(match.group(1))


def _split_by_added_silence(
    input_path: Path,
    added_silence: int,
    *,
    output_dir: Path | None,
    prefix: str | None,
    overwrite: bool,
    dry_run: bool,
    content_threshold_db: str = "-50dB",
    min_segment_duration: float = 60.0,
) -> SplitResult:
    """Split a recording that has ``added_silence`` seconds of silence between tracks.

    Steps:
    1. Detect silences of at least ``added_silence - 1`` seconds.
    2. Build segments between consecutive long silences.
    3. Drop segments shorter than ``min_segment_duration`` or too quiet.
    4. Trim the added silence out of each segment by using the silence edges as
       boundaries.
    5. Write the resulting tracks.
    """
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".aif", ".aiff", ".aifc"}:
        raise FFmpegError("input must have an .aif, .aiff, or .aifc extension")

    info = probe_audio(input_path)
    min_silence_duration = added_silence - 1
    raw_silences = detect_silences(
        input_path,
        gap=min_silence_duration,
        threshold_db=content_threshold_db,
    )
    silences = [Silence(**region) for region in raw_silences]
    long_silences = [s for s in silences if s.duration >= min_silence_duration]
    if not long_silences:
        raise SplitError(
            f"no silences of at least {min_silence_duration}s detected; "
            "check the --added-silence value or recording quality"
        )

    # Build candidate tracks bounded by the edges of the long silences.
    boundaries: list[tuple[float, float]] = []
    prev_end = 0.0
    for silence in long_silences:
        start = prev_end
        end = silence.start
        boundaries.append((start, end))
        prev_end = silence.end
    boundaries.append((prev_end, info.duration))

    # Filter by duration and loudness.
    filtered: list[tuple[float, float]] = []
    for start, end in boundaries:
        duration = end - start
        if duration < min_segment_duration:
            continue
        max_db = _segment_max_volume(input_path, start, end)
        if max_db < -50.0:
            continue
        filtered.append((start, end))

    if not filtered:
        raise SplitError(
            "no valid tracks found between the detected silences; "
            "segments may all be too short or too quiet"
        )

    tracks = [
        Track(number=index + 1, start=start, end=end)
        for index, (start, end) in enumerate(filtered)
    ]

    if dry_run:
        return SplitResult(tracks=tracks, output_paths=[])

    out_dir = output_dir.expanduser().resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = prefix or f"{input_path.stem}_track"
    extension = _output_extension(input_path)
    track_count = len(tracks)
    width = max(2, len(str(track_count)))
    paths = [
        out_dir / f"{base}_{number:0{width}d}{extension}"
        for number in range(1, track_count + 1)
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
    hints: Sequence[float] | None = None,
    added_silence: int = 0,
) -> SplitResult:
    """Split a recording into tracks using silence detection and dynamic programming."""
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FFmpegError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".aif", ".aiff", ".aifc"}:
        raise FFmpegError("input must have an .aif, .aiff, or .aifc extension")

    if added_silence > 0:
        if added_silence < 2:
            raise FFmpegError("--added-silence must be at least 2 seconds")
        return _split_by_added_silence(
            input_path,
            added_silence,
            output_dir=output_dir,
            prefix=prefix,
            overwrite=overwrite,
            dry_run=dry_run,
        )

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

    content_start, content_end = _identify_content_edges(silences, info.duration, edge_guard)
    candidates = _internal_candidates(silences, content_start, content_end)

    def _build_diagnostics(message: str) -> SplitError:
        return SplitError(
            message,
            diagnostics=_silence_diagnostics(
                silences,
                candidates,
                track_count,
                min_track,
                max_track,
                gap,
                threshold_db,
                content_start,
                content_end,
                info.duration,
            ),
        )

    if not silences:
        raise _build_diagnostics(
            "FFmpeg detected no silence regions; try a shorter gap or a less-negative threshold"
        )

    usable_duration = content_end - content_start
    target = target_track or usable_duration / track_count
    target = min(max(target, min_track), max_track)

    # Fast path: when the user supplies exactly the required number of boundary
    # hints, snap each hint to the closest detected silence and build tracks
    # directly without running the constrained dynamic-programming search.
    boundaries_needed = track_count - 1
    if hints and len(hints) == boundaries_needed:
        sorted_hints = sorted(hints)
        selected = [_snap_hint_to_silence(h, silences) for h in sorted_hints]
        tracks = _build_tracks(selected, content_start, content_end, info.duration, padding)
    else:
        try:
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
                hints=hints,
            )
        except SplitError as exc:
            raise _build_diagnostics(str(exc)) from exc
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

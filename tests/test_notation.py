"""Tests for notation cleanup and score assembly."""

from pathlib import Path

from music21 import note, stream
from reversescore.config import PipelineConfig
from reversescore.notation import (
    _rebuild_parts,
    _simplify_durations,
    assign_instrument,
    build_score,
    export_score,
    quantize_score,
    set_time_signature,
    split_bandoneon_part,
)


def test_set_time_signature() -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(note.Note("C4", quarterLength=1))
    score.insert(0, part)
    score = set_time_signature(score, "2/4")
    ts = score.parts[0].getElementsByClass("TimeSignature")[0]
    assert ts.ratioString == "2/4"


def test_quantize_score() -> None:
    part = stream.Part()
    # A note placed off the 16th-note grid with a messy duration.
    n = note.Note("C4", quarterLength=0.24)
    part.insert(0.07, n)
    score = stream.Score()
    score.insert(0, part)
    quantized = quantize_score(score)
    qn = list(quantized.parts[0].recurse().notes)[0]
    # Quantization should snap the note onto a 16th-note grid and clean
    # the duration (0.24 -> 0.25) rather than preserve the raw values.
    assert qn.offset in (0.0, 0.0625)
    assert float(qn.duration.quarterLength) == 0.25


def test_assign_instrument() -> None:
    part = stream.Part()
    part.append(note.Note("C4"))
    config = PipelineConfig()
    assign_instrument(part, "bass", config)
    instruments = list(part.getElementsByClass("Instrument"))
    assert len(instruments) == 1


def test_assign_instrument_with_hints() -> None:
    part = stream.Part()
    part.append(note.Note("C4"))
    config = PipelineConfig(instrument_hints=["violin"])
    assign_instrument(part, "bandoneon_violin", config)
    instruments = list(part.getElementsByClass("Instrument"))
    assert instruments[0].instrumentName == "Violin"


def test_build_score_skips_excluded_stems(tmp_path: Path) -> None:
    midi_paths: dict[str, Path] = {}
    for label in ("drums", "bass", "guitar"):
        score = stream.Score()
        part = stream.Part()
        part.append(note.Note("C4", quarterLength=4))
        score.insert(0, part)
        path = tmp_path / f"{label}.mid"
        score.write("midi", fp=str(path))
        midi_paths[label] = path

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        instrument_exclusions=["drums", "guitar"],
        split_bandoneon=False,
    )
    score = build_score(midi_paths, config, title="Test")
    assert len(score.parts) == 1
    assert score.parts[0].partAbbreviation == "bass"


def test_split_bandoneon_part(tmp_path: Path) -> None:
    part = stream.Part()
    part.partAbbreviation = "bandoneon"
    m = stream.Measure(number=1)
    m.insert(0, note.Note("C6", quarterLength=1))  # treble
    m.insert(1, note.Note("C3", quarterLength=1))  # bass
    part.append(m)
    score = stream.Score()
    score.insert(0, part)

    split = split_bandoneon_part(score)
    assert len(split.parts) == 2


def test_build_score_and_export(tmp_path: Path) -> None:
    # Create a tiny MIDI file for each stem.
    midi_paths: dict[str, Path] = {}
    for label in ("bass", "vocals"):
        score = stream.Score()
        part = stream.Part()
        part.append(note.Note("C4", quarterLength=4))
        score.insert(0, part)
        path = tmp_path / f"{label}.mid"
        score.write("midi", fp=str(path))
        midi_paths[label] = path

    config = PipelineConfig(output_dir=tmp_path / "out", split_bandoneon=False)
    score = build_score(midi_paths, config, title="Test")
    assert len(score.parts) == 2

    musicxml_path = tmp_path / "out" / "test.musicxml"
    export_score(score, musicxml_path, fmt="musicxml")
    assert musicxml_path.exists()


def test_simplify_complex_durations() -> None:
    part = stream.Part()
    m = stream.Measure(number=1)
    m.insert(0, note.Note("C4", quarterLength=2.5))
    part.append(m)
    score = stream.Score()
    score.insert(0, part)

    score = _simplify_durations(score)
    notes = list(score.parts[0].recurse().notes)
    assert len(notes) == 2
    assert notes[0].duration.type == "half"
    assert notes[1].duration.type == "eighth"


def test_rebuild_parts_avoids_export_crash(tmp_path: Path) -> None:
    # A part with gaps and a complex duration would previously crash MusicXML
    # export with makeNotation=True due to duplicate measures.
    part = stream.Part()
    part.insert(0, note.Note("C4", quarterLength=1.5))
    part.insert(2, note.Note("D4", quarterLength=2.5))
    part.insert(5, note.Note("E4", quarterLength=0.5))
    score = stream.Score()
    score.insert(0, part)

    score = _rebuild_parts(score, "4/4")
    musicxml_path = tmp_path / "rebuild.musicxml"
    export_score(score, musicxml_path, fmt="musicxml")
    assert musicxml_path.exists()


def test_build_score_handles_complex_durations(tmp_path: Path) -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(note.Note("C4", quarterLength=2.5))
    score.insert(0, part)
    path = tmp_path / "complex.mid"
    score.write("midi", fp=str(path))

    config = PipelineConfig(output_dir=tmp_path / "out", split_bandoneon=False)
    built = build_score({"bass": path}, config)
    assert len(built.parts) == 1

    musicxml_path = tmp_path / "out" / "complex.musicxml"
    export_score(built, musicxml_path, fmt="musicxml")
    assert musicxml_path.exists()

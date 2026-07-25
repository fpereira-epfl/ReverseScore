"""Notation cleanup, quantization, and score assembly with music21."""

from __future__ import annotations

import logging
from pathlib import Path

import music21
from music21 import articulations, converter, instrument, meter, note, stream

from .config import PipelineConfig
from .utils import ensure_dirs

logger = logging.getLogger(__name__)


def _bandoneon_instrument() -> instrument.Instrument:
    """Return a music21 Instrument named Bandoneon.

    music21 does not provide a dedicated Bandoneon class, so we build one from
    the generic ``Instrument`` base and set the part name.
    """
    return instrument.Instrument(
        instrumentName="Bandoneon",
        instrumentAbbreviation="Band.",
        midiProgram=21,  # Accordion sound is the closest General MIDI match.
    )


# Instrument assignment for tango stem labels.
STEM_INSTRUMENTS: dict[str, instrument.Instrument] = {
    "bandoneon_piano_violin": _bandoneon_instrument(),
    "bandoneon": _bandoneon_instrument(),
    "piano": instrument.Piano(),
    "violin": instrument.Violin(),
    "bass": instrument.Contrabass(),
    "vocals": instrument.Vocalist(),
    "drums": instrument.Percussion(),
    "other": _bandoneon_instrument(),
}

# Default clef hints per stem label.
CLEF_HINTS: dict[str, str] = {
    "bass": "bass",
    "bandoneon_piano_violin": "treble",
    "bandoneon": "treble",
    "piano": "treble",
    "violin": "treble",
    "vocals": "treble",
    "drums": "percussion",
}


def load_midi(path: Path) -> stream.Score:
    """Load a MIDI file into a music21 Score."""
    logger.debug("Loading MIDI: %s", path)
    return converter.parse(str(path))


def set_time_signature(score: stream.Score, time_signature: str) -> stream.Score:
    """Insert the requested time signature at the start of every part."""
    ts = meter.TimeSignature(time_signature)
    for part in score.parts:
        measure_zero = part.measure(0)
        if measure_zero is None:
            part.insert(0, ts)
        else:
            existing = list(measure_zero.getElementsByClass("TimeSignature"))
            if existing:
                measure_zero.remove(existing[0])
            measure_zero.insert(0, ts)
    return score


def quantize_score(
    score: stream.Score,
    quantization_grid: int = 4,
    min_quarter_length: float = 0.0625,
    recurse: bool = True,
) -> stream.Score:
    """Quantize note/rest offsets and durations to a fixed grid.

    Args:
        score: music21 Score to quantize.
        quantization_grid: music21 tuple denominator; 4 = 16th notes.
        min_quarter_length: Smallest note value to retain (quarter-note units).
        recurse: Apply recursively to all contained streams.

    Returns:
        Quantized score.
    """
    logger.info(
        "Quantizing to 1/%d quarter-note grid (min length %.4f)",
        quantization_grid,
        min_quarter_length,
    )
    score.quantize(
        (quantization_grid,),
        processOffsets=True,
        processDurations=True,
        recurse=recurse,
        inPlace=True,
    )

    # Remove notes that are too short after quantization.
    for el in score.recurse().notesAndRests:
        if el.duration.quarterLength < min_quarter_length:
            el.duration.quarterLength = min_quarter_length
    return score


def strip_pitch_bends(score: stream.Score) -> stream.Score:
    """Remove pitch-bend events; basic-pitch writes them as microtonal accidentals.

    Tango transcriptions read more cleanly in MuseScore without microtones.
    Future work can convert pitch bends to grace-note glissandi.
    """
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            for pb in list(measure.getElementsByClass("PitchBend")):
                measure.remove(pb)
    return score


def assign_instrument(part: stream.Part, label: str) -> stream.Part:
    """Set a sensible instrument/clef for a part based on its stem label."""
    inst = STEM_INSTRUMENTS.get(label, _bandoneon_instrument())
    part.insert(0, inst)
    clef_name = CLEF_HINTS.get(label, "treble")
    try:
        clef_obj = music21.clef.clefFromString(clef_name)
    except Exception:  # pragma: no cover - defensive fallback
        clef_obj = music21.clef.TrebleClef()
    part.insert(0, clef_obj)
    return part


def add_marcato_accents(part: stream.Part) -> stream.Part:
    """Accent the first attack of each measure for bass and drum parts.

    This captures the tango marcato feel without over-marking every note.
    """
    for measure in part.getElementsByClass("Measure"):
        notes = list(measure.notes)
        if notes:
            first = notes[0]
            if not any(isinstance(a, articulations.Accent) for a in first.articulations):
                first.articulations.append(articulations.Accent())
    return part


def merge_stem_midis(
    midi_paths: dict[str, Path],
    config: PipelineConfig,
    title: str | None = None,
) -> stream.Score:
    """Merge per-stem MIDI files into a single quantized score.

    Args:
        midi_paths: Mapping from stem label to MIDI path.
        config: Pipeline configuration.
        title: Optional score title.

    Returns:
        A music21 Score with one Part per stem.
    """
    score = stream.Score()
    if title:
        score.insert(0, music21.metadata.Metadata(title=title))

    for label, midi_path in midi_paths.items():
        stem_score = load_midi(midi_path)
        for part in stem_score.parts:
            assign_instrument(part, label)
            # Tag the part with the stem label for easier cleanup.
            part.partAbbreviation = label[:8]
            if label in ("bass", "drums"):
                add_marcato_accents(part)
            score.insert(0, part)

    score = set_time_signature(score, config.time_signature)
    score = quantize_score(
        score,
        quantization_grid=config.quantization_grid,
        min_quarter_length=config.min_quarter_length,
    )
    score = strip_pitch_bends(score)
    return score


def split_bandoneon_part(score: stream.Score) -> stream.Score:
    """Split a combined bandoneon/piano/violin part into treble and bass staves.

    This is a heuristic split based on pitch range. Notes below B3 go to the
    bass staff, others to the treble staff. In MuseScore the user can refine
    this further.

    Args:
        score: A music21 Score.

    Returns:
        Score with split staves.
    """
    B3 = music21.pitch.Pitch("B3")
    new_score = stream.Score()

    for part in score.parts:
        if part.partAbbreviation not in ("bandone", "bandoneon", "bandoneon_piano_violin"):
            new_score.insert(0, part)
            continue

        treble_part = stream.Part()
        treble_part.id = f"{part.id}_treble"
        treble_part.partAbbreviation = "band_tre"
        treble_part.insert(0, _bandoneon_instrument())
        treble_part.insert(0, music21.clef.TrebleClef())

        bass_part = stream.Part()
        bass_part.id = f"{part.id}_bass"
        bass_part.partAbbreviation = "band_bas"
        bass_part.insert(0, _bandoneon_instrument())
        bass_part.insert(0, music21.clef.BassClef())

        for measure in part.getElementsByClass("Measure"):
            treble_measure = stream.Measure(number=measure.number)
            bass_measure = stream.Measure(number=measure.number)
            for el in measure:
                if isinstance(el, note.Note):
                    target = bass_measure if el.pitch <= B3 else treble_measure
                    target.insert(el.offset, el)
                elif isinstance(el, note.Rest):
                    # Place rests in both staves to preserve measure alignment.
                    treble_measure.insert(el.offset, note.Rest(quarterLength=el.duration.quarterLength))
                    bass_measure.insert(el.offset, note.Rest(quarterLength=el.duration.quarterLength))
                elif isinstance(el, meter.TimeSignature):
                    treble_measure.insert(0, el)
                    bass_measure.insert(0, el)

            if list(treble_measure.notes):
                treble_part.append(treble_measure)
            if list(bass_measure.notes):
                bass_part.append(bass_measure)

        if list(treble_part.flatten().notes):
            new_score.insert(0, treble_part)
        if list(bass_part.flatten().notes):
            new_score.insert(0, bass_part)

    return new_score


def export_score(
    score: stream.Score,
    output_path: Path,
    fmt: str = "musicxml",
) -> Path:
    """Export a music21 Score to MusicXML or MIDI.

    Args:
        score: music21 Score.
        output_path: Destination path (extension controls format if ``fmt`` is not given).
        fmt: music21 format string (``musicxml`` or ``midi``).

    Returns:
        Path to the exported file.
    """
    ensure_dirs(output_path.parent)
    logger.info("Exporting %s score to %s", fmt, output_path)
    written = score.write(fmt=fmt, fp=str(output_path))
    return Path(written)


def build_score(
    midi_paths: dict[str, Path],
    config: PipelineConfig,
    title: str | None = None,
    split_bandoneon: bool | None = None,
) -> stream.Score:
    """High-level helper: merge, quantize, optionally split, and return a score."""
    score = merge_stem_midis(midi_paths, config, title=title)
    if split_bandoneon is None:
        split_bandoneon = config.split_bandoneon
    if split_bandoneon:
        score = split_bandoneon_part(score)
    return score

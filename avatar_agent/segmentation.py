from __future__ import annotations

import re
from typing import Iterable, List

from .models import AudioSegment, WordTimestamp


_BOUNDARY = re.compile(r"[.!?;:]$")


def _text(words: Iterable[WordTimestamp]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip()).strip()


def segment_words(
    words: List[WordTimestamp],
    min_seconds: float = 4.0,
    max_seconds: float = 6.0,
    short_seconds: float = 3.0,
) -> List[AudioSegment]:
    """Create semantic segments while preserving short transitional phrases.

    The splitter prefers punctuation and pauses after the 4-second target,
    forces a boundary at 6 seconds, and keeps a final remainder below 3 seconds
    as a short segment instead of stretching its audio.
    """
    if not words:
        return []
    result: List[AudioSegment] = []
    current: List[WordTimestamp] = []
    for word in words:
        # Split before adding a word that would push the segment beyond the
        # hard limit. This keeps normal segments at or below six seconds.
        if current and word.end - current[0].start > max_seconds:
            result.append(_make_segment(len(result), current))
            current = []
        current.append(word)
        duration = current[-1].end - current[0].start
        boundary = bool(_BOUNDARY.search(word.text.strip()))
        pause_after = False
        if len(current) > 1:
            previous = current[-2]
            pause_after = word.start - previous.end >= 0.35
        if duration >= max_seconds or (duration >= min_seconds and (boundary or pause_after)):
            result.append(_make_segment(len(result), current))
            current = []
    if current:
        remainder = current[-1].end - current[0].start
        if result and remainder >= short_seconds and result[-1].duration + remainder <= max_seconds:
            previous_words = result.pop()
            previous_start = previous_words.start
            previous_end = previous_words.end
            combined_text = (previous_words.text + " " + _text(current)).strip()
            result.append(AudioSegment(len(result), previous_start, previous_end if previous_end > current[-1].end else current[-1].end, combined_text))
        else:
            result.append(_make_segment(len(result), current))
    return [AudioSegment(i, segment.start, segment.end, segment.text, segment.energy_peak) for i, segment in enumerate(result)]


def _make_segment(index: int, words: List[WordTimestamp]) -> AudioSegment:
    if not words:
        raise ValueError("cannot create an empty segment")
    peak = 0.5
    emphatic = {"important", "key", "注意", "核心", "must", "first", "finally"}
    if any(word.text.lower().strip(".,!?;:") in emphatic for word in words):
        peak = 0.8
    return AudioSegment(index, words[0].start, words[-1].end, _text(words), peak)

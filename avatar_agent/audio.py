from __future__ import annotations

from array import array
from typing import List

from .models import AudioSegment


def attach_energy_peaks(audio_path: str, segments: List[AudioSegment], sample_rate: int = 16000) -> List[AudioSegment]:
    """Compute one normalized RMS peak value per segment with ffmpeg."""
    import subprocess

    command = [
        "ffmpeg", "-v", "error", "-i", audio_path, "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "-"
    ]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    samples = array("h")
    samples.frombytes(raw)
    values: List[float] = []
    for segment in segments:
        start = max(0, int(segment.start * sample_rate))
        end = min(len(samples), max(start + 1, int(segment.end * sample_rate)))
        window = samples[start:end]
        if not window:
            values.append(0.0)
            continue
        values.append((sum(float(sample) * sample for sample in window) / len(window)) ** 0.5)
    if not values:
        return segments
    low, high = min(values), max(values)
    if high == low:
        normalized = [0.5] * len(values)
    else:
        normalized = [(value - low) / (high - low) for value in values]
    return [
        AudioSegment(segment.index, segment.start, segment.end, segment.text, normalized[index])
        for index, segment in enumerate(segments)
    ]

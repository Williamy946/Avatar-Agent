from __future__ import annotations

import json
from typing import Any, Dict, List

from .models import WordTimestamp


def load_transcript(path: str) -> List[WordTimestamp]:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    raw = payload.get("words", payload) if isinstance(payload, dict) else payload
    words = [WordTimestamp.from_dict(item) for item in raw]
    if not words:
        raise ValueError("transcript contains no word timestamps")
    previous = -1.0
    for word in words:
        if word.end < word.start or word.start < previous:
            raise ValueError("word timestamps must be ordered and non-negative")
        previous = word.start
    return words


class WhisperTimestampedASR:
    """Lazy adapter for whisper-timestamped; never imports it for offline use."""

    def __init__(self, model: str = "NbAiLab/whisper-large-v2-nob", language: str | None = "en", device: str = "cpu"):
        self.model_name = model
        self.language = language
        self.device = device

    def transcribe(self, audio_path: str) -> List[WordTimestamp]:
        try:
            import whisper_timestamped as whisper
        except ImportError as exc:
            raise RuntimeError("install whisper-timestamped or provide --transcript") from exc
        audio = whisper.load_audio(audio_path)
        model = whisper.load_model(self.model_name, device=self.device)
        result = whisper.transcribe(model, audio, language=self.language)
        words: List[WordTimestamp] = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append(WordTimestamp.from_dict(word))
        if not words:
            raise ValueError("Whisper returned no word timestamps")
        return words

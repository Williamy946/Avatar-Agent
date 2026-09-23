"""Inference-only Avatar-Agent reference implementation."""

from .models import AudioSegment, GestureClip, WordTimestamp
from .pipeline import Stage2Pipeline

__all__ = ["AudioSegment", "GestureClip", "WordTimestamp", "Stage2Pipeline"]

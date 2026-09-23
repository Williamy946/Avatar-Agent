from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "WordTimestamp":
        return cls(str(value["text"]), float(value["start"]), float(value["end"]))


@dataclass(frozen=True)
class AudioSegment:
    index: int
    start: float
    end: float
    text: str
    energy_peak: float = 0.5

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GestureClip:
    clip_id: str
    category: str
    caption: str
    path: str
    duration: float = 5.0
    motion_peak: float = 0.5
    embedding: List[float] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    verified: bool = True
    static: bool = False

    @classmethod
    def from_dict(cls, value: Dict[str, Any], base_dir: str = "") -> "GestureClip":
        path = str(value.get("path", value.get("file", "")))
        if base_dir and path and not path.startswith(("/", "\\")):
            import os

            path = os.path.normpath(os.path.join(base_dir, path))
        return cls(
            clip_id=str(value.get("clip_id", value.get("id", ""))),
            category=str(value.get("category", value.get("type", "beat"))),
            caption=str(value.get("caption", value.get("description", ""))),
            path=path,
            duration=float(value.get("duration", 5.0)),
            motion_peak=float(value.get("motion_peak", value.get("motionPeak", 0.5))),
            embedding=[float(x) for x in value.get("embedding", [])],
            keywords=[str(x).lower() for x in value.get("keywords", [])],
            verified=bool(value.get("verified", True)),
            static=bool(value.get("static", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

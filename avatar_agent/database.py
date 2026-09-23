from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List

from .models import GestureClip


def load_database(path: str) -> List[GestureClip]:
    """Load a flat clip manifest or the paper's category-based manifest."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base_dir = os.path.dirname(os.path.abspath(path))
    if isinstance(payload, list):
        raw = payload
    elif "clips" in payload:
        raw = payload["clips"]
    else:
        raw = []
        for category in payload.get("categories", []):
            category_id = str(category.get("id", category.get("name", "unknown")))
            category_name = str(category.get("name", category_id))
            for gesture in category.get("gestures", []):
                common = {
                    "clip_id": gesture.get("id", category_id),
                    "category": category_name,
                    "caption": gesture.get("caption", gesture.get("name", "")),
                    "path": gesture.get("path", ""),
                    "duration": gesture.get("duration", 5.0),
                    "motion_peak": gesture.get("motion_peak", 0.5),
                    "keywords": gesture.get("keywords", gesture.get("use_cases", [])),
                    "verified": gesture.get("verified", True),
                    "static": category_id == "L0",
                }
                for key in ("description_5s", "description"):
                    if not common["caption"] and gesture.get(key):
                        common["caption"] = gesture[key]
                raw.append(common)
    clips = [GestureClip.from_dict(value, base_dir=base_dir) for value in raw]
    if not clips:
        raise ValueError("gesture database contains no clips")
    ids = [clip.clip_id for clip in clips]
    if any(not clip.clip_id for clip in clips) or len(ids) != len(set(ids)):
        raise ValueError("gesture database clip_id values must be non-empty and unique")
    return clips


def save_database(path: str, clips: Iterable[GestureClip], metadata: Dict[str, object] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"version": "1.0", "metadata": metadata or {}, "clips": [asdict(c) for c in clips]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

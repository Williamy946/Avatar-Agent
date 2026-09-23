from __future__ import annotations

import json
import math
import re
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import AudioSegment, GestureClip


TOKEN = re.compile(r"[A-Za-z0-9']+")
CATEGORY_HINTS = {
    "iconic": {"size", "shape", "show", "open", "wide", "structure", "path"},
    "metaphoric": {"system", "growth", "connect", "combine", "process", "future"},
    "conduit": {"explain", "say", "language", "communication", "idea"},
    "beat": {"important", "key", "must", "attention", "first", "finally", "warning"},
}


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN.findall(value)}


def _normalise(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class SemanticAgent:
    name = "semantic"

    def rank(self, segment: AudioSegment, candidates: Sequence[GestureClip]) -> List[Tuple[str, float]]:
        query = _tokens(segment.text)
        values: Dict[str, float] = {}
        for clip in candidates:
            caption = _tokens(clip.caption) | set(clip.keywords)
            overlap = len(query & caption) / max(1, len(query))
            hints = CATEGORY_HINTS.get(clip.category.lower(), set())
            hint_score = len(query & hints) / max(1, len(hints))
            values[clip.clip_id] = overlap + 0.25 * hint_score
        return sorted(values.items(), key=lambda item: (-item[1], item[0]))


class BeatAgent:
    name = "beat"

    def rank(self, segment: AudioSegment, candidates: Sequence[GestureClip]) -> List[Tuple[str, float]]:
        values = {clip.clip_id: 1.0 - abs(float(clip.motion_peak) - segment.energy_peak) for clip in candidates}
        return sorted(values.items(), key=lambda item: (-item[1], item[0]))


def _distance(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


class DiversityAgent:
    name = "diversity"

    def rank(self, segment: AudioSegment, candidates: Sequence[GestureClip], history: Sequence[GestureClip]) -> List[Tuple[str, float]]:
        recent = list(history)[-4:]
        values: Dict[str, float] = {}
        for clip in candidates:
            if not recent:
                values[clip.clip_id] = 1.0
                continue
            distances = [_distance(clip.embedding, old.embedding) for old in recent]
            category_penalty = sum(1.0 for old in recent if old.category == clip.category) / len(recent)
            values[clip.clip_id] = (sum(distances) / len(distances)) + (1.0 - category_penalty)
        return sorted(values.items(), key=lambda item: (-item[1], item[0]))


class DirectorAgent:
    """Resolve expert rankings with an explicit semantic > beat > diversity policy."""

    priority = (0.50, 0.30, 0.20)

    def select(
        self,
        segment: AudioSegment,
        candidates: Sequence[GestureClip],
        semantic: Sequence[Tuple[str, float]],
        beat: Sequence[Tuple[str, float]],
        diversity: Sequence[Tuple[str, float]],
        history: Sequence[GestureClip],
    ) -> GestureClip:
        allowed = [clip for clip in candidates if clip.verified]
        if not allowed:
            raise ValueError("no verified gesture candidates")
        maps = [_normalise(dict(ranking)) for ranking in (semantic, beat, diversity)]
        previous = history[-1].clip_id if history else None
        scored = []
        for clip in allowed:
            score = sum(weight * mapping.get(clip.clip_id, 0.0) for weight, mapping in zip(self.priority, maps))
            repeat_penalty = 0.25 if clip.clip_id == previous and len(allowed) > 1 else 0.0
            scored.append((score - repeat_penalty, clip.clip_id, clip))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]


class GestureNarrator:
    def __init__(self, semantic: SemanticAgent | None = None, beat: BeatAgent | None = None, diversity: DiversityAgent | None = None, director: DirectorAgent | None = None):
        self.semantic = semantic or SemanticAgent()
        self.beat = beat or BeatAgent()
        self.diversity = diversity or DiversityAgent()
        self.director = director or DirectorAgent()
        self.history: deque[GestureClip] = deque(maxlen=8)

    def choose(self, segment: AudioSegment, candidates: Sequence[GestureClip]) -> GestureClip:
        semantic = self.semantic.rank(segment, candidates)
        beat = self.beat.rank(segment, candidates)
        diversity = self.diversity.rank(segment, candidates, list(self.history))
        selected = self.director.select(segment, candidates, semantic, beat, diversity, list(self.history))
        self.history.append(selected)
        return selected

    @classmethod
    def from_llm(cls, client: object) -> "GestureNarrator":
        return cls(
            semantic=LLMRankingAgent(client, "semantic"),
            beat=LLMRankingAgent(client, "beat"),
            diversity=LLMRankingAgent(client, "diversity"),
            director=LLMDirectorAgent(client),
        )


class LLMRankingAgent:
    """Use the same candidate list for an MLLM expert ranking."""

    def __init__(self, client: object, role: str):
        self.client = client
        self.role = role
        self.name = role

    def rank(self, segment: AudioSegment, candidates: Sequence[GestureClip], history: Sequence[GestureClip] | None = None) -> List[Tuple[str, float]]:
        items = [{"clip_id": c.clip_id, "category": c.category, "caption": c.caption, "motion_peak": c.motion_peak} for c in candidates]
        prompt = (
            f"Role: {self.role} expert. Rank every candidate from 0 to 1 for the audio segment. "
            "Return JSON only as {\"scores\":[{\"clip_id\":\"...\",\"score\":0.0}]}\n"
            f"Segment: {segment.text}\nAudio peak: {segment.energy_peak}\nCandidates: {json.dumps(items)}"
        )
        response = self.client.request("You are one expert in a multi-agent gesture narrator.", prompt)
        scores = response.get("scores", []) if isinstance(response, dict) else []
        values = {str(item["clip_id"]): float(item["score"]) for item in scores if "clip_id" in item and "score" in item}
        return [(candidate.clip_id, values.get(candidate.clip_id, 0.0)) for candidate in candidates]


class LLMDirectorAgent(DirectorAgent):
    def __init__(self, client: object):
        super().__init__()
        self.client = client

    def select(self, segment, candidates, semantic, beat, diversity, history):
        allowed = [clip for clip in candidates if clip.verified]
        if not allowed:
            raise ValueError("no verified gesture candidates")
        prompt = (
            "Resolve the expert rankings. Prefer semantic compatibility first, then beat alignment, "
            "then motion diversity. Reject unverified clips, avoid an immediate repetition when an "
            "alternative exists, and return JSON only as {\"clip_id\":\"...\"}.\n"
            f"Segment: {segment.text}\n"
            f"Semantic: {json.dumps(list(semantic))}\nBeat: {json.dumps(list(beat))}\n"
            f"Diversity: {json.dumps(list(diversity))}\nHistory: {json.dumps([c.clip_id for c in history])}"
        )
        response = self.client.request("You are the Director agent for a lecture video.", prompt)
        selected_id = response.get("clip_id") if isinstance(response, dict) else None
        for clip in allowed:
            if clip.clip_id == selected_id and (not history or clip.clip_id != history[-1].clip_id or len(allowed) == 1):
                return clip
        return super().select(segment, allowed, semantic, beat, diversity, history)

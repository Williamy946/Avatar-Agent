from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .database import save_database
from .media import probe_duration
from .models import GestureClip


CATEGORIES = ("iconic", "metaphoric", "conduit", "beat")


class ClipGenerator(Protocol):
    def generate(self, portrait: str, caption: str, output: str, duration: float) -> None: ...


class ClipVerifier(Protocol):
    def verify(self, clip_path: str, caption: str) -> bool: ...


class CommandClipGenerator:
    def __init__(self, command: str):
        self.command = command

    def generate(self, portrait: str, caption: str, output: str, duration: float) -> None:
        values = {
            "portrait": shlex.quote(portrait),
            "caption": shlex.quote(caption),
            "output": shlex.quote(output),
            "duration": f"{duration:.3f}",
        }
        command = self.command.format(**values)
        subprocess.run(shlex.split(command), check=True)


class FfprobeVerifier:
    def verify(self, clip_path: str, caption: str) -> bool:
        try:
            return os.path.isfile(clip_path) and probe_duration(clip_path) >= 2.5
        except (OSError, ValueError, subprocess.CalledProcessError):
            return False


class CommandVerifier:
    """Run an optional visual verifier; exit status zero accepts the clip."""

    def __init__(self, command: str):
        self.command = command

    def verify(self, clip_path: str, caption: str) -> bool:
        values = {"clip": shlex.quote(clip_path), "caption": shlex.quote(caption)}
        command = self.command.format(**values)
        try:
            subprocess.run(shlex.split(command), check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False


class LLMCaptionProvider:
    def __init__(self, client: object):
        self.client = client

    def generate(self) -> List[dict]:
        response = self.client.request(
            "You generate safe, culturally neutral lecture gestures.",
            "Generate exactly ten concise gesture captions covering iconic, metaphoric, conduit, "
            "and beat categories. Do not use a separate deictic category; express pointing as an "
            "iconic action tied to a concrete referent. Return JSON only as "
            "{\"captions\":[{\"category\":\"iconic\",\"caption\":\"...\"}]}.",
        )
        captions = response.get("captions", []) if isinstance(response, dict) else []
        valid = [item for item in captions if item.get("category") in CATEGORIES and item.get("caption")]
        if len(valid) < 4:
            raise ValueError("LLM returned fewer than four valid gesture captions")
        return valid[:10]


def default_captions() -> List[dict]:
    return [
        {"id": "gen_0", "category": "iconic", "caption": "opens both palms upward slowly from the center of the abdomen to waist-chest height, holds, and returns, presenting a concept", "keywords": ["present", "show", "introduce"]},
        {"id": "gen_1", "category": "metaphoric", "caption": "opens both palms upward quickly and widely to chest-shoulder height, holds, and returns, symbolizing openness, possibility, or welcome", "keywords": ["open", "possibility", "welcome"]},
        {"id": "gen_2", "category": "iconic", "caption": "moves the right hand horizontally from the center toward the right with the palm turning forward, then makes two or three small waves to browse content or signal", "keywords": ["browse", "wave", "sequence"]},
        {"id": "gen_3", "category": "iconic", "caption": "spreads both palms upward from the abdomen toward both sides near shoulder height, holds briefly, and returns in a presenting gesture", "keywords": ["show", "display", "present"]},
        {"id": "gen_4", "category": "metaphoric", "caption": "opens both hands forward and upward, brings the palms together at the chest, holds, and returns, symbolizing fusion, combination, or completeness", "keywords": ["fuse", "combine", "integrate"]},
        {"id": "gen_5", "category": "conduit", "caption": "opens both palms upward from the abdomen toward both sides at waist-chest height, holds, and returns in an explanatory open-hand gesture", "keywords": ["explain", "describe", "idea"]},
        {"id": "gen_6", "category": "beat", "caption": "opens both palms symmetrically to the sides and makes small upward-and-downward beats while holding the pose to emphasize the current point", "keywords": ["emphasize", "important", "point"]},
        {"id": "gen_7", "category": "beat", "caption": "raises the right hand from the abdomen to shoulder height with the palm forward and makes two or three small side-to-side waves for emphasis while the left hand stays still", "keywords": ["emphasize", "attention", "signal"]},
        {"id": "gen_8", "category": "conduit", "caption": "keeps both hands naturally clasped in front of the abdomen while standing upright and attentively explaining, with no hand motion", "keywords": ["explain", "rest", "transition"], "static": True},
        {"id": "gen_9", "category": "beat", "caption": "raises the left hand from the abdomen to shoulder height with the palm forward and makes small side-to-side waves for emphasis while the right hand stays still", "keywords": ["emphasize", "attention", "signal"]},
    ]


def build_database(
    portrait: str,
    output_dir: str,
    manifest_path: str,
    generator: ClipGenerator,
    verifier: ClipVerifier | None = None,
    captions: Iterable[dict] | None = None,
    duration: float = 5.0,
) -> List[GestureClip]:
    os.makedirs(output_dir, exist_ok=True)
    verifier = verifier or FfprobeVerifier()
    clips: List[GestureClip] = []
    for index, item in enumerate(captions or default_captions()):
        category = str(item["category"])
        caption = str(item["caption"])
        clip_id = str(item.get("id", f"{category}_{index:02d}"))
        output = os.path.join(output_dir, f"{clip_id}.mp4")
        generator.generate(portrait, caption, output, duration)
        verified = verifier.verify(output, caption)
        if not verified:
            continue
        clips.append(
            GestureClip(
                clip_id,
                category,
                caption,
                output,
                duration=probe_duration(output),
                verified=True,
                static=bool(item.get("static", False)),
                keywords=[str(value).lower() for value in item.get("keywords", [])],
            )
        )
    if not clips:
        raise RuntimeError("Stage 1 produced no verified clips")
    save_database(manifest_path, clips, {"portrait": os.path.abspath(portrait), "categories": list(CATEGORIES)})
    return clips

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
        {"category": "iconic", "caption": "opens both hands to show the size of an object"},
        {"category": "iconic", "caption": "moves one hand along an imagined path between two ideas"},
        {"category": "metaphoric", "caption": "brings both hands together to express connection and integration"},
        {"category": "metaphoric", "caption": "raises one hand gradually to express growth or progression"},
        {"category": "conduit", "caption": "opens one palm while explaining an idea to the audience"},
        {"category": "conduit", "caption": "gestures gently toward the audience while inviting attention"},
        {"category": "beat", "caption": "makes a short downward beat to emphasize a key point"},
        {"category": "beat", "caption": "makes two small rhythmic beats while stressing a conclusion"},
        {"category": "beat", "caption": "raises one hand briefly to mark the first item in a list"},
        {"category": "iconic", "caption": "frames a broad shape with both hands while describing it"},
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
        clip_id = f"{category}_{index:02d}"
        output = os.path.join(output_dir, f"{clip_id}.mp4")
        generator.generate(portrait, caption, output, duration)
        verified = verifier.verify(output, caption)
        if not verified:
            continue
        clips.append(GestureClip(clip_id, category, caption, output, duration=probe_duration(output), verified=True))
    if not clips:
        raise RuntimeError("Stage 1 produced no verified clips")
    save_database(manifest_path, clips, {"portrait": os.path.abspath(portrait), "categories": list(CATEGORIES)})
    return clips

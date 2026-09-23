from __future__ import annotations

import json
import os
import tempfile
import unittest

from avatar_agent.agents import GestureNarrator
from avatar_agent.duration import align_clip
from avatar_agent.models import AudioSegment, GestureClip, WordTimestamp
from avatar_agent.segmentation import segment_words
from avatar_agent.pipeline import Stage2Pipeline


class SegmentationTests(unittest.TestCase):
    def test_short_transitional_phrase_is_preserved(self):
        words = [WordTimestamp("Now", 0.0, 0.4), WordTimestamp("we", 0.4, 0.7), WordTimestamp("begin", 0.7, 1.2)]
        result = segment_words(words)
        self.assertEqual(len(result), 1)
        self.assertLess(result[0].duration, 3.0)

    def test_segments_are_bounded(self):
        words = [WordTimestamp(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(20)]
        result = segment_words(words)
        self.assertTrue(result)
        self.assertTrue(all(segment.duration <= 6.01 for segment in result))


class NarratorTests(unittest.TestCase):
    def test_all_agents_rank_same_candidates_and_director_avoids_repeat(self):
        candidates = [
            GestureClip("a", "iconic", "show size", "a.mp4", keywords=["size"], embedding=[1, 0]),
            GestureClip("b", "beat", "emphasize key", "b.mp4", motion_peak=0.8, keywords=["key"], embedding=[0, 1]),
        ]
        narrator = GestureNarrator()
        first = narrator.choose(AudioSegment(0, 0, 5, "show size"), candidates)
        second = narrator.choose(AudioSegment(1, 5, 10, "emphasize key"), candidates)
        self.assertNotEqual(first.clip_id, second.clip_id)

    def test_llm_adapter_uses_explicit_director_rule(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def request(self, system, user):
                self.calls.append((system, user))
                if "Resolve the expert rankings" in user:
                    return {"clip_id": "b"}
                return {"scores": [{"clip_id": "a", "score": 0.9}, {"clip_id": "b", "score": 0.1}]}

        candidates = [
            GestureClip("a", "iconic", "show size", "a.mp4", keywords=["size"]),
            GestureClip("b", "beat", "emphasize key", "b.mp4", keywords=["key"]),
        ]
        fake = FakeClient()
        narrator = GestureNarrator.from_llm(fake)
        chosen = narrator.choose(AudioSegment(0, 0, 5, "show size"), candidates)
        self.assertEqual(chosen.clip_id, "b")
        self.assertEqual(len(fake.calls), 4)


class DurationPolicyTests(unittest.TestCase):
    @staticmethod
    def _ffmpeg_assets(directory: str) -> tuple[str, str]:
        import subprocess

        image = os.path.join(directory, "rest.png")
        motion = os.path.join(directory, "motion.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", image], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", image, "-t", "5", "-r", "24", "-pix_fmt", "yuv420p", motion], check=True)
        return image, motion

    def test_three_duration_branches(self):
        from avatar_agent.media import probe_duration

        with tempfile.TemporaryDirectory() as directory:
            _, motion = self._ffmpeg_assets(directory)
            for target, mode in ((2.0, "static-short"), (4.0, "accelerated-motion"), (6.0, "natural-motion-static-tail")):
                output = os.path.join(directory, f"{target}.mp4")
                result = align_clip(motion, motion, target, output)
                self.assertEqual(result.mode, mode)
                self.assertAlmostEqual(probe_duration(output), target, delta=0.15)

    def test_stage2_pipeline_composes_and_writes_report(self):
        from avatar_agent.media import probe_duration

        with tempfile.TemporaryDirectory() as directory:
            image, motion = self._ffmpeg_assets(directory)
            audio = os.path.join(directory, "audio.wav")
            manifest = os.path.join(directory, "transcript.json")
            import subprocess

            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=6", audio], check=True)
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"words": [
                    {"text": "Now", "start": 0.0, "end": 0.4},
                    {"text": "we", "start": 0.4, "end": 0.7},
                    {"text": "begin", "start": 0.7, "end": 1.2},
                    {"text": "the", "start": 3.5, "end": 3.8},
                    {"text": "main", "start": 3.8, "end": 4.2},
                    {"text": "lesson", "start": 4.2, "end": 4.8}
                ]}, handle)
            clips = [GestureClip("motion", "iconic", "begin main lesson", motion, keywords=["lesson"], embedding=[1, 0])]
            output_dir = os.path.join(directory, "run")
            report = Stage2Pipeline(clips).run(audio, motion, output_dir, transcript_path=manifest)
            self.assertTrue(os.path.isfile(report["video"]))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "run.json")))
            self.assertAlmostEqual(probe_duration(report["video"]), 6.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import List, Optional

from .agents import GestureNarrator
from .audio import attach_energy_peaks
from .asr import WhisperTimestampedASR, load_transcript
from .database import load_database
from .duration import align_clip, align_static_gap
from .media import concat_videos, mux_audio, probe_duration
from .models import AudioSegment, GestureClip
from .segmentation import segment_words


class Stage2Pipeline:
    def __init__(self, clips: List[GestureClip], fps: int = 24, llm_client: object | None = None):
        self.clips = [clip for clip in clips if clip.verified]
        if not self.clips:
            raise ValueError("Stage 2 requires at least one verified clip")
        self.fps = fps
        self.narrator = GestureNarrator.from_llm(llm_client) if llm_client else GestureNarrator()

    @classmethod
    def from_manifest(cls, manifest: str, fps: int = 24, llm_client: object | None = None) -> "Stage2Pipeline":
        return cls(load_database(manifest), fps=fps, llm_client=llm_client)

    def run(
        self,
        audio_path: str,
        rest_clip: str,
        output_dir: str,
        transcript_path: str | None = None,
        mux: bool = False,
        lipsync_command: str | None = None,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        words = load_transcript(transcript_path) if transcript_path else WhisperTimestampedASR().transcribe(audio_path)
        segments = segment_words(words)
        if not segments:
            raise ValueError("audio produced no segments")
        segments = attach_energy_peaks(audio_path, segments)
        aligned: List[str] = []
        decisions = []
        timeline = []
        cursor = 0.0
        for segment in segments:
            if segment.start - cursor > 0.02:
                timeline.append(("gap", cursor, segment.start, None))
            timeline.append(("speech", segment.start, segment.end, segment))
            cursor = segment.end
        audio_duration = probe_duration(audio_path)
        if audio_duration - cursor > 0.02:
            timeline.append(("gap", cursor, audio_duration, None))
        for item_index, (kind, start, end, segment) in enumerate(timeline):
            target = end - start
            path = os.path.join(output_dir, f"segment_{item_index:04d}.mp4")
            if kind == "gap":
                result = align_static_gap(rest_clip, target, path, fps=self.fps)
                aligned.append(result.output)
                decisions.append({"kind": kind, "start": start, "end": end, "duration": target, "mode": result.mode, "output": result.output})
                continue
            selected = self.narrator.choose(segment, self.clips)
            result = align_clip(selected.path, rest_clip, segment.duration, path, fps=self.fps)
            aligned.append(result.output)
            decisions.append({"segment": segment.to_dict(), "clip_id": selected.clip_id, "mode": result.mode, "output": result.output})
        silent_video = os.path.join(output_dir, "gesture_video.mp4")
        concat_videos(aligned, silent_video)
        final_video = silent_video
        if mux:
            final_video = os.path.join(output_dir, "gesture_video_with_audio.mp4")
            mux_audio(silent_video, audio_path, final_video)
        if lipsync_command:
            lipsync_output = os.path.join(output_dir, "avatar_agent_lipsync.mp4")
            command = lipsync_command.format(video=final_video, audio=audio_path, output=lipsync_output)
            subprocess.run(shlex.split(command), check=True)
            final_video = lipsync_output
        report = {"audio": os.path.abspath(audio_path), "video": os.path.abspath(final_video), "segments": decisions}
        with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        return report

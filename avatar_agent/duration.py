from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from .media import concat_videos, extract_frame, probe_duration, render_static, speed_video, trim_video


@dataclass(frozen=True)
class AlignmentResult:
    output: str
    target_duration: float
    mode: str


def align_static_gap(rest_clip: str, target_duration: float, output: str, fps: int = 24) -> AlignmentResult:
    """Render a complete timeline gap from one static rest frame."""
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avatar-agent-gap-") as temp:
        frame = os.path.join(temp, "rest.jpg")
        extract_frame(rest_clip, frame)
        render_static(frame, target_duration, output, fps)
    return AlignmentResult(output, target_duration, "static-gap")


def align_clip(
    motion_clip: str,
    rest_clip: str,
    target_duration: float,
    output: str,
    fps: int = 24,
    static_threshold: float = 3.0,
    motion_duration: float = 5.0,
) -> AlignmentResult:
    """Align a selected clip without slowing or time-stretching the audio.

    The policy follows the paper implementation: <3 seconds uses a static
    portrait frame; 3--<5 seconds speeds the 5-second motion clip; >=5 seconds
    keeps natural motion and pads any remainder with the rest frame.
    """
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avatar-agent-align-") as temp:
        frame = os.path.join(temp, "rest.jpg")
        extract_frame(rest_clip, frame)
        if target_duration < static_threshold:
            render_static(frame, target_duration, output, fps)
            return AlignmentResult(output, target_duration, "static-short")
        if target_duration < motion_duration:
            speed_video(motion_clip, target_duration, output, fps)
            return AlignmentResult(output, target_duration, "accelerated-motion")
        actual = probe_duration(motion_clip)
        motion_out = os.path.join(temp, "motion.mp4")
        if actual > target_duration:
            trim_video(motion_clip, target_duration, motion_out, fps)
            concat_videos([motion_out], output)
            return AlignmentResult(output, target_duration, "natural-motion-trimmed")
        if actual >= target_duration - 0.02:
            trim_video(motion_clip, target_duration, motion_out, fps)
            concat_videos([motion_out], output)
            return AlignmentResult(output, target_duration, "natural-motion")
        tail = os.path.join(temp, "tail.mp4")
        render_static(frame, target_duration - actual, tail, fps)
        concat_videos([motion_clip, tail], output)
        return AlignmentResult(output, target_duration, "natural-motion-static-tail")

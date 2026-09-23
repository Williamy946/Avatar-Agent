from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Iterable, List


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required executable not found: {name}")
    return path


def run(command: List[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_duration(path: str) -> float:
    require_binary("ffprobe")
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


def extract_frame(video: str, image: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(image)), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-frames:v", "1", "-q:v", "2", image])


def render_static(image: str, duration: float, output: str, fps: int = 24) -> None:
    if duration <= 0:
        raise ValueError("duration must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", image, "-t", f"{duration:.6f}", "-r", str(fps), "-an", "-pix_fmt", "yuv420p", output])


def speed_video(video: str, target_duration: float, output: str, fps: int = 24) -> None:
    source_duration = probe_duration(video)
    if target_duration <= 0 or source_duration <= 0:
        raise ValueError("video and target duration must be positive")
    factor = source_duration / target_duration
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-an", "-vf", f"setpts={factor:.10f}*PTS,fps={fps}", "-t", f"{target_duration:.6f}", "-pix_fmt", "yuv420p", output])


def trim_video(video: str, target_duration: float, output: str, fps: int = 24) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-an", "-vf", f"fps={fps}", "-t", f"{target_duration:.6f}", "-pix_fmt", "yuv420p", output])


def concat_videos(videos: Iterable[str], output: str) -> None:
    videos = list(videos)
    if not videos:
        raise ValueError("no videos to concatenate")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fd, list_path = tempfile.mkstemp(prefix="avatar-agent-concat-", suffix=".txt")
    os.close(fd)
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for video in videos:
                # The concat demuxer accepts single-quoted paths. JSON-style
                # double quotes are parsed as part of the filename by ffmpeg.
                safe_path = os.path.abspath(video).replace("'", "'\\''")
                handle.write("file '" + safe_path + "'\n")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output])
    finally:
        os.unlink(list_path)


def mux_audio(video: str, audio: str, output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", output])

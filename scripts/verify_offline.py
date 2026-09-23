from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from avatar_agent.media import probe_duration
from avatar_agent.duration import align_clip


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="avatar-agent-verify-") as directory:
        image = os.path.join(directory, "rest.png")
        motion = os.path.join(directory, "motion.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=64x64", "-frames:v", "1", image], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", image, "-t", "5", "-r", "24", "-pix_fmt", "yuv420p", motion], check=True)
        for target, expected in ((2.0, "static-short"), (4.0, "accelerated-motion"), (6.0, "natural-motion-static-tail")):
            output = os.path.join(directory, f"aligned_{target}.mp4")
            result = align_clip(motion, motion, target, output)
            actual = probe_duration(output)
            if result.mode != expected or abs(actual - target) > 0.15:
                raise AssertionError(f"target={target}: mode={result.mode}, duration={actual}")
            print(f"PASS target={target:.1f}s mode={result.mode} duration={actual:.3f}s")
    print("Offline Avatar-Agent verification passed")


if __name__ == "__main__":
    main()

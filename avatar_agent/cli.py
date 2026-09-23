from __future__ import annotations

import argparse
import json
import os

from .asr import load_transcript
from .database import load_database
from .pipeline import Stage2Pipeline
from .segmentation import segment_words
from .stage1 import CommandClipGenerator, CommandVerifier, FfprobeVerifier, LLMCaptionProvider, build_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only Avatar-Agent pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    segment = sub.add_parser("segment", help="segment a word-level transcript")
    segment.add_argument("--transcript", required=True)

    stage1 = sub.add_parser("stage1", help="build a gesture database")
    stage1.add_argument("--portrait", required=True)
    stage1.add_argument("--output-dir", required=True)
    stage1.add_argument("--manifest", required=True)
    stage1.add_argument("--generator-command", required=True)
    stage1.add_argument("--verifier-command", help="optional command with {clip} and {caption} placeholders")
    stage1.add_argument("--llm", action="store_true", help="use the configured LLM to propose captions")

    stage2 = sub.add_parser("stage2", help="compose a long tutor video")
    stage2.add_argument("--audio", required=True)
    stage2.add_argument("--transcript")
    stage2.add_argument("--manifest", required=True)
    stage2.add_argument("--rest-clip", required=True)
    stage2.add_argument("--output-dir", required=True)
    stage2.add_argument("--fps", type=int, default=24)
    stage2.add_argument("--mux-audio", action="store_true")
    stage2.add_argument("--lipsync-command", help="command with {video}, {audio}, {output} placeholders")
    stage2.add_argument("--llm", action="store_true", help="use the configured LLM agents")

    args = parser.parse_args()
    if args.command == "segment":
        segments = segment_words(load_transcript(args.transcript))
        print(json.dumps([segment.to_dict() for segment in segments], indent=2))
    elif args.command == "stage1":
        captions = None
        if args.llm:
            from .llm import OpenAICompatibleJSON
            captions = LLMCaptionProvider(OpenAICompatibleJSON()).generate()
        verifier = CommandVerifier(args.verifier_command) if args.verifier_command else FfprobeVerifier()
        build_database(args.portrait, args.output_dir, args.manifest, CommandClipGenerator(args.generator_command), verifier, captions=captions)
        print(args.manifest)
    else:
        client = None
        if args.llm:
            from .llm import OpenAICompatibleJSON
            client = OpenAICompatibleJSON()
        report = Stage2Pipeline.from_manifest(args.manifest, fps=args.fps, llm_client=client).run(args.audio, args.rest_clip, args.output_dir, args.transcript, args.mux_audio, args.lipsync_command)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

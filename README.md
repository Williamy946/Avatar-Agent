# Avatar-Agent source

This directory is the reproducible source package for the inference-only
Avatar-Agent pipeline described in the paper. It contains two stages:

1. Stage 1 proposes gesture captions, generates first-last-frame clips through
   a user-supplied generator command, verifies the clips, and writes a reusable
   gesture database manifest.
2. Stage 2 consumes word-level ASR timestamps, creates 4--6 second semantic
   segments (or preserves segments shorter than 3 seconds), ranks the same
   candidates with Semantic, Beat, and Diversity agents, resolves the rankings
   with a Director, aligns each selected clip, and optionally invokes MuseTalk.

The source package does not contain credentials. Configure an external model
provider with environment variables when using the optional LLM adapters:

```bash
export AVATAR_AGENT_API_KEY=...
export AVATAR_AGENT_BASE_URL=https://your-openai-compatible-endpoint/v1
export AVATAR_AGENT_MODEL=your-model
```

## Requirements

Python 3.10 or newer and the `ffmpeg` and `ffprobe` executables are required
for media operations. Install the optional Python integrations with:

```bash
pip install -r requirements.txt
```

`whisper-timestamped` and `openai` are only imported when the corresponding
adapter is used. The offline tests use no model, API, GPU, or network.

## Stage 2

The simplest reproducible path is to provide a transcript JSON with word-level
timestamps. Its shape is:

```json
{"language":"en", "words":[
  {"text":"Welcome", "start":0.0, "end":0.4},
  {"text":"to", "start":0.4, "end":0.6}
]}
```

Run:

```bash
python -m avatar_agent.cli stage2 \
  --audio lecture.wav \
  --transcript transcript.json \
  --manifest examples/gesture_database.json \
  --rest-clip clips/rest.mp4 \
  --output-dir outputs/lecture
```

The duration policy is explicit:

- segment shorter than 3 seconds: render a static portrait frame;
- segment from 3 to less than 5 seconds: speed the selected 5-second motion
  clip up to the segment duration;
- segment of at least 5 seconds: retain natural motion speed and pad the tail
  with the static rest frame when the segment is longer than the clip.

The generated silent video is written as `gesture_video.mp4`. Add
`--mux-audio` to mux the original audio. MuseTalk is intentionally exposed as
an external command (`--lipsync-command`) so the source package does not make
assumptions about a local MuseTalk checkout or model paths.

Word gaps before, between, and after speech segments are preserved as static
rest-frame intervals, so the composed video keeps the full input-audio
timeline.

Use `--llm` with `stage2` to replace the deterministic offline rankers with
the Semantic, Beat, Diversity, and Director adapters. The adapters pass the
same candidate list to all experts and apply the Director priority rule
semantic compatibility > beat alignment > motion diversity. They require the
three environment variables shown above.

## Stage 1

Stage 1 is provider-agnostic. A generator command receives these placeholders:
`{portrait}`, `{caption}`, `{output}`, and `{duration}`. For example:

```bash
python -m avatar_agent.cli stage1 \
  --portrait teacher.png \
  --output-dir outputs/database \
  --manifest outputs/database/gesture_database.json \
  --generator-command 'python tools/generate_clip.py --portrait {portrait} --caption {caption} --output {output} --duration {duration}'
```

The command is called once per generated caption. The source package does not
hard-code a commercial provider or API key; the provider adapter can be added
behind the same command interface. By default `FfprobeVerifier` checks that a
clip is readable and long enough. Pass `--verifier-command` to run a visual
multimodal verifier; a zero exit status accepts the clip.

## Verification

From this directory:

```bash
python -m unittest discover -s tests -v
python scripts/verify_offline.py
```

The verification script creates tiny media files with ffmpeg, exercises all
three duration branches, checks the output duration, and uses deterministic
mock agents. It does not run video generation, Whisper, MuseTalk, or an LLM.

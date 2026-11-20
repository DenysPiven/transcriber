# Transcriber

Local Whisper transcription of long recordings (meetings, screen captures, YouTube videos) into timestamped text and subtitles. On an Apple Silicon Mac it runs on the GPU and transcribes 1.5 hours of audio in under 3 minutes.

## Quick start

Requires Python 3.10+, `ffmpeg`, and `yt-dlp` for YouTube links.

```bash
brew install ffmpeg yt-dlp
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python transcribe.py ~/Movies/meeting.mp4
```

The first run downloads the model (~1.6 GB) into the Hugging Face cache.

Output goes to `out/<file_name>/`:

| File | Content |
|---|---|
| `transcript_<engine>_<model>.txt` | Paragraphs with `[hh:mm:ss]` timestamps |
| `transcript_<engine>_<model>.srt` | Subtitles |
| `audio_16k.wav` | Extracted 16 kHz mono audio |
| `youtube_subs.*` | YouTube's own subtitles, for comparison (URLs only) |

## Options

| Flag | Default | Notes |
|---|---|---|
| `--engine` | `mlx` | `mlx`: Apple Silicon GPU. `faster`: faster-whisper on the CPU, works anywhere |
| `--model` | `large-v3-turbo` | Also `large-v3`, `distil-large-v3`, `medium`, `small`, ... |
| `--language` | `en` | Language code or `auto` |
| `--recheck` | `large-v3` | `mlx` only: model for the second pass; `none` to skip |
| `--keep-fillers` | off | Keep "uh" / "um" in the output |
| `--beam` | `1` | `faster` only: beam size (`mlx` is always greedy) |
| `--threads` | all cores | `faster` only |
| `--out` | `out` | Output directory |

## Progress bar

Run the job in the background with a log, then watch it:

```bash
.venv/bin/python -u transcribe.py meeting.mp4 > run.log 2>&1 &
./progress.sh run.log
```

```
[############----------------------------]  31.0%  00:30:12 / 01:39:31  ETA 00:01:52
```

Without an argument `progress.sh` follows the newest `*.log` in the project folder. With the `mlx` engine the bar restarts once for the short second pass.

## How it works

1. `ffmpeg` extracts 16 kHz mono audio.
2. Silero VAD finds speech and packs it into pieces of up to 30 s, split at pauses. Silence never reaches the model, which cuts work and hallucinations.
3. Each piece is decoded on its own as a single Whisper window. Decoding one long stream makes Whisper skip the rest of a window whenever it ends early, and passages get lost.
4. Every piece gets the same short prompt with punctuation and fillers. Without a prompt the output has no punctuation. With a "clean" prompt the model drops disfluent passages ("uh, you know, uh, ...") entirely.
5. Pieces with fewer than 1 word per second are decoded again with `--recheck`, and the longer result is kept.
6. Timestamps are mapped back to the original recording, then fillers are removed and the `.txt` and `.srt` files are written.

## Benchmark

1 h 39 min English meeting recording, MacBook Pro M4 Pro (14-core CPU, 20-core GPU):

| Setup | Time |
|---|---|
| `--engine faster --model large-v3 --beam 5` (CPU) | ~70 min (estimated from a partial run) |
| `--engine mlx --model large-v3-turbo` (GPU, default) | 2 min 49 s |

On a 5-minute clip, `large-v3-turbo` on MLX and `large-v3` with beam 5 on the CPU produced almost identical text, 16x apart in speed.

#!/usr/bin/env python3
"""Full transcription of a YouTube video or a local file with Whisper.

Engines:
    mlx     - mlx-whisper on the Apple Silicon GPU (default, fastest on a Mac)
    faster  - faster-whisper on the CPU (works anywhere)

Usage:
    python3 transcribe.py "https://youtu.be/VIDEO_ID"
    python3 transcribe.py meeting.mp4
    python3 transcribe.py meeting.mp4 --engine faster --model large-v3 --beam 5

Output (in out/<name>/):
    transcript_<engine>_<model>.txt  - paragraphs with [hh:mm:ss] timestamps
    transcript_<engine>_<model>.srt  - subtitles
    youtube_subs.*                   - YouTube subtitles, if available (for comparison)
"""
import argparse
import os
import re
import subprocess
import sys
import time
import wave
from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000
MAX_PIECE = 30  # seconds: one Whisper window
# Pieces at least this long (s) with fewer words per second get a second pass.
RECHECK_MIN_DURATION = 5
RECHECK_MIN_WPS = 1.0
# Sets a punctuated style. The fillers matter: with a clean prompt the model
# skips disfluent passages ("uh, you know, uh, ...") entirely.
STYLE_PROMPT = "Uh, so, um, yeah. Okay, you know, it's, uh, fine."

# Short model names -> mlx-community repos on Hugging Face.
MLX_MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
}


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ts(sec, srt=False):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


class Progress:
    """Prints one line per update: [position / total] percent ETA.

    progress.sh parses this format, so keep it stable.
    """

    def __init__(self, total):
        self.total, self.t0 = total, time.time()

    def show(self, done_fraction, position):
        elapsed = time.time() - self.t0
        eta = elapsed / done_fraction * (1 - done_fraction) if done_fraction else 0
        print(f"\r[{ts(position)} / {ts(self.total)}] {done_fraction:6.1%}  ETA {ts(eta)}",
              end="", flush=True)


def fetch_youtube(url, out_dir):
    # YouTube subtitles (manual and auto) - as a fallback and for comparison.
    try:
        run(["yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
             "--sub-langs", "en.*,en", "--convert-subs", "srt",
             "-o", os.path.join(out_dir, "youtube_subs.%(ext)s"), url])
    except subprocess.CalledProcessError:
        print("! Could not fetch YouTube subtitles, continuing without them")
    run(["yt-dlp", "-f", "bestaudio/best", "-x", "--audio-format", "wav",
         "-o", os.path.join(out_dir, "audio_raw.%(ext)s"), url])
    return os.path.join(out_dir, "audio_raw.wav")


def to_wav16k(src, dst):
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", dst])


def read_wav(path):
    with wave.open(path) as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0


def transcribe_faster(wav, model_name, language, beam, threads):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=threads)
    segments, info = model.transcribe(
        wav,
        language=None if language == "auto" else language,
        beam_size=beam,
        vad_filter=True,  # strips silence, fewer hallucinations
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # keeps the model from looping on long recordings
    )
    print(f"Language: {info.language} ({info.language_probability:.0%})", flush=True)
    progress, result = Progress(info.duration), []
    for seg in segments:
        result.append(seg)
        progress.show(seg.end / info.duration if info.duration else 0, seg.end)
    print()
    return result


def transcribe_mlx(wav, model_name, language, recheck_model):
    import mlx_whisper
    from faster_whisper.vad import (SpeechTimestampsMap, VadOptions,
                                    collect_chunks, get_speech_timestamps)

    audio = read_wav(wav)
    total = len(audio) / SAMPLE_RATE

    # Find speech with Silero VAD and pack it into pieces of up to 30 s, split at
    # pauses. Each piece is one Whisper window, decoded on its own: on a long
    # stream Whisper skips the rest of a window whenever it stops early.
    chunks = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=500, max_speech_duration_s=MAX_PIECE))
    if not chunks:
        print("No speech found", flush=True)
        return []
    pieces, meta = collect_chunks(audio, chunks, SAMPLE_RATE, max_duration=MAX_PIECE)
    ts_map = SpeechTimestampsMap(chunks, SAMPLE_RATE)
    speech_total = sum(m["duration"] for m in meta)
    print(f"Speech: {ts(speech_total)} of {ts(total)} in {len(pieces)} pieces", flush=True)

    def decode(piece, m, repo):
        out = mlx_whisper.transcribe(
            piece,
            path_or_hf_repo=repo,
            language=None if language == "auto" else language,
            initial_prompt=STYLE_PROMPT,  # without it the output has no punctuation
            # VAD already removed silence; the no-speech check would drop quiet speech.
            no_speech_threshold=None,
        )
        segments = []
        for s in out["segments"]:
            # On unclear audio the model can emit empty text, end <= start, or
            # segments past the end of the piece; those are noise.
            if not s["text"].strip() or s["end"] <= s["start"] or s["start"] >= m["duration"]:
                continue
            start = s["start"] + m["offset"]
            end = min(s["end"], m["duration"]) + m["offset"]
            segments.append(SimpleNamespace(start=ts_map.get_original_time(start),
                                            end=ts_map.get_original_time(end, is_end=True),
                                            text=s["text"]))
        return segments

    def word_count(segments):
        return sum(len(s.text.split()) for s in segments)

    def run_pass(indices, repo, label):
        print(f"{label}: {len(indices)} pieces with {MLX_MODELS.get(repo, repo)}", flush=True)
        work = sum(meta[i]["duration"] for i in indices)
        progress, done, results = Progress(total), 0.0, {}
        for i in indices:
            results[i] = decode(pieces[i], meta[i], MLX_MODELS.get(repo, repo))
            done += meta[i]["duration"]
            end = meta[i]["offset"] + meta[i]["duration"]
            progress.show(done / work, ts_map.get_original_time(end, is_end=True))
        print()
        return results

    per_piece = run_pass(range(len(pieces)), model_name, "Pass 1")

    # A fast model sometimes ends a window early and drops the rest of the speech.
    # Such pieces stand out by a low word rate; decode them again with the
    # recheck model and keep whichever version has more words.
    if recheck_model not in (None, "none", model_name):
        suspicious = [i for i, m in enumerate(meta)
                      if m["duration"] >= RECHECK_MIN_DURATION
                      and word_count(per_piece[i]) / m["duration"] < RECHECK_MIN_WPS]
        if suspicious:
            redone = run_pass(suspicious, recheck_model, "Pass 2 (low word rate)")
            improved = [i for i in suspicious if word_count(redone[i]) > word_count(per_piece[i])]
            for i in improved:
                per_piece[i] = redone[i]
            print(f"Pass 2 improved {len(improved)} of {len(suspicious)} pieces "
                  f"(+{sum(word_count(redone[i]) for i in improved)} words in them)", flush=True)

    return [s for i in range(len(pieces)) for s in per_piece[i]]


F = r"\b(?:[Uu]h+|[Uu]m+|[Uu]hm|[Ee]rm)\b"


def strip_fillers(text):
    """'Uh, you know, we, uh, need it, um.' -> 'You know, we need it.'"""
    starts_with_filler = re.match(r"\s*" + F, text)
    text = re.sub(rf",\s*{F}\s*,", "", text)        # "we, uh, need" -> "we need"
    text = re.sub(rf",\s*{F}(?=[.?!])", "", text)   # "and, uh." -> "and."
    text = re.sub(rf"{F}[,.?!]?\s*", "", text)      # "Uh, you know" -> "you know"
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"^[\s,.?!]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([.?!]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)
    if starts_with_filler and text:
        text = text[0].upper() + text[1:]
    return text


def write_outputs(segments, out_dir, stem):
    srt_path = os.path.join(out_dir, f"{stem}.srt")
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    with open(srt_path, "w") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{ts(s.start, True)} --> {ts(s.end, True)}\n{s.text.strip()}\n\n")

    # Paragraphs: new paragraph after a pause > 2 s or ~90 s of text.
    paras, cur, cur_start, prev_end = [], [], None, None
    for s in segments:
        gap = s.start - prev_end if prev_end is not None else 0
        if cur and (gap > 2.0 or s.start - cur_start > 90):
            paras.append((cur_start, " ".join(cur)))
            cur = []
        if not cur:
            cur_start = s.start
        cur.append(s.text.strip())
        prev_end = s.end
    if cur:
        paras.append((cur_start, " ".join(cur)))
    with open(txt_path, "w") as f:
        for start, text in paras:
            text = re.sub(r"\s+", " ", text)
            f.write(f"[{ts(start)}] {text}\n\n")
    return txt_path, srt_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="YouTube URL or path to an audio/video file")
    p.add_argument("--engine", choices=["mlx", "faster"], default="mlx",
                   help="mlx (Apple Silicon GPU, fastest) or faster (CPU)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="large-v3-turbo (fast, near large-v3 quality), large-v3, "
                        "distil-large-v3, small, ...")
    p.add_argument("--language", default="en", help="language code (en, uk, ...) or auto")
    p.add_argument("--recheck", default="large-v3",
                   help="mlx: model for the second pass over pieces with a low word "
                        "rate; 'none' to skip")
    p.add_argument("--keep-fillers", action="store_true",
                   help="keep uh/um in the output (removed by default)")
    p.add_argument("--beam", type=int, default=1,
                   help="beam size for the faster engine (mlx is always greedy)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--out", default="out")
    a = p.parse_args()

    is_url = re.match(r"https?://", a.source)
    name = re.sub(r"\W+", "_", a.source.split("/")[-1].split("?")[0]) or "video"
    out_dir = os.path.join(a.out, name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    src = fetch_youtube(a.source, out_dir) if is_url else a.source
    wav = os.path.join(out_dir, "audio_16k.wav")
    to_wav16k(src, wav)

    if a.engine == "mlx":
        segments = transcribe_mlx(wav, a.model, a.language, a.recheck)
    else:
        segments = transcribe_faster(wav, a.model, a.language, a.beam, a.threads)
    if not a.keep_fillers:
        segments = [SimpleNamespace(start=s.start, end=s.end, text=t)
                    for s in segments if (t := strip_fillers(s.text))]
    stem = f"transcript_{a.engine}_{re.sub(r'[^A-Za-z0-9.-]+', '-', a.model)}"
    txt, srt = write_outputs(segments, out_dir, stem)
    print(f"Took {ts(time.time() - t0)}")
    print(f"Done: {txt}, {srt}")


if __name__ == "__main__":
    sys.exit(main())

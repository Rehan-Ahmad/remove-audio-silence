#!/usr/bin/env python3
"""
Trim leading/trailing silence using Silero VAD, but KEEP 0.25 s at both ends.
Writes 24 kHz WAVs for StyleTTS2.

Usage:
  python trim_with_silero_vad.py \
      --in_dir data/raw_wavs --out_dir data/wavs --keep_silence 0.25 \
      [--onnx] [--energy_threshold 0.5] [--min_speech_ms 120] [--min_silence_ms 200] \
      [--viz_dir data/viz]

Notes:
- Silero supports 16k and 8k; we run detection at 16k and map timestamps to the
  original sampling rate before saving at 24k for StyleTTS2.
- If no speech is detected, the file is resampled and copied through.
- If --viz_dir is given, a PNG is written per file with the waveform and VAD
  activity (square-wave overlay).
"""
import argparse, os, sys, math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# plotting (non-GUI backend)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silero imports
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio


def resample(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def clamp01(x):
    return max(0.0, min(1.0, x))


def save_vad_plot(audio, sr, speech_ts, out_png, time_step=0.01, kept_region=None):
    """
    Save a visualization: waveform + square-wave VAD mask (+ optional kept region).

    audio: 1D np.array (float32)
    sr: sample rate of audio
    speech_ts: list of {'start': s, 'end': e} in seconds
    out_png: path to save PNG
    time_step: resolution (sec) of the VAD mask
    kept_region: optional (start_s, end_s) in seconds of the FINAL kept region
                 after padding and clipping (in the SAME timebase as 'audio').
    """
    if audio is None or len(audio) == 0:
        return

    t = np.arange(len(audio)) / float(sr)
    dur = t[-1]
    max_amp = max(1e-6, float(np.max(np.abs(audio))))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, audio, color="0.6", linewidth=0.6, label="Waveform")

    # Build square-wave-like VAD mask (raw speech_ts)
    if dur > 0:
        tt = np.arange(0.0, dur + time_step, time_step)
        mask = np.zeros_like(tt)
        for seg in speech_ts or []:
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            mask[(tt >= s) & (tt <= e)] = 1.0

        scaled = mask * (0.9 * max_amp)
        ax.step(tt, scaled, where="post", color="orange", linewidth=1.0, label="VAD speech")
        ax.fill_between(tt, 0, scaled, step="post", color="orange", alpha=0.25)

    # Optional: show final kept region (after padding)
    if kept_region is not None:
        k0, k1 = kept_region
        k0 = max(0.0, k0)
        k1 = max(k0, k1)
        ax.axvspan(k0, k1, color="green", alpha=0.15, label="Kept (with padding)")

    ax.set_xlim(0, max(dur, 0.1))
    ax.set_ylim(-max_amp * 1.05, max_amp * 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)



def process_one(
    in_wav,
    out_wav,
    keep_silence,
    model,
    analyze_sr=16000,
    energy_threshold=0.5,
    min_speech_ms=120,
    min_silence_ms=200,
    viz_png=None,
    vad_json=None,
):
    # Read original (float32)
    audio, orig_sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # For VAD: cast to 16k (Silero supports 8k/16k; JIT will cast multiples internally,
    # but we explicitly analyze at 16k for clarity)
    audio_16k = resample(audio, orig_sr, analyze_sr)

    # Run Silero VAD to get speech timestamps (returned in seconds)
    speech_ts = get_speech_timestamps(
        audio_16k,
        model,
        sampling_rate=analyze_sr,
        threshold=clamp01(energy_threshold),
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=True,
    )

    if vad_json is not None:
        try:
            os.makedirs(os.path.dirname(vad_json), exist_ok=True)
            import json
            with open(vad_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "in_wav": in_wav,
                        "orig_sr": orig_sr,
                        "analyze_sr": analyze_sr,
                        "energy_threshold": energy_threshold,
                        "min_speech_ms": min_speech_ms,
                        "min_silence_ms": min_silence_ms,
                        "speech_ts": speech_ts,  # list of {start, end} in seconds
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"ERR vad_json {vad_json}: {e}", file=sys.stderr)

    # If nothing detected, just resample/copy to 24k
    if not speech_ts:
        audio_24k = resample(audio, orig_sr, 24000)
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        sf.write(out_wav, audio_24k, 24000, subtype="PCM_16")
        return "no-speech"

    # Determine first/last speech (in seconds at analyze_sr)
    start_s = speech_ts[0]["start"]
    end_s = speech_ts[-1]["end"]

    # Keep context on both sides
    pad = keep_silence
    t0 = max(0.0, start_s - pad)
    t1 = min(len(audio_16k) / analyze_sr, end_s + pad)

    # Save visualization if requested (now we know kept_region)
    if viz_png is not None:
        try:
            # audio_16k + analyze_sr, with both raw VAD and final kept region
            save_vad_plot(
                audio_16k,
                analyze_sr,
                speech_ts,
                viz_png,
                kept_region=(t0, t1),
            )
        except Exception as e:
            print(f"ERR viz {viz_png}: {e}", file=sys.stderr)

    # Map seconds to original sample indices
    start_orig = int(round(t0 * orig_sr))
    end_orig = int(round(t1 * orig_sr))
    start_orig = max(0, start_orig)
    end_orig = min(len(audio), end_orig)
    if end_orig <= start_orig:
        start_orig, end_orig = 0, len(audio)

    trimmed = audio[start_orig:end_orig]

    # Final write at 24 kHz for StyleTTS2
    trimmed_24k = resample(trimmed, orig_sr, 24000)
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    sf.write(out_wav, trimmed_24k, 24000, subtype="PCM_16")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_silence", type=float, default=0.25)
    ap.add_argument("--onnx", action="store_true", help="Use ONNX backend")
    ap.add_argument("--energy_threshold", type=float, default=0.5)
    ap.add_argument("--min_speech_ms", type=int, default=120)
    ap.add_argument("--min_silence_ms", type=int, default=200)
    ap.add_argument(
        "--viz_dir",
        default=None,
        help="If set, save waveform+VAD PNGs mirroring input tree",
    )
    ap.add_argument(
        "--vad_dir",
        default=None,
        help="If set, save raw speech_ts JSONs mirroring input tree",
    )
    args = ap.parse_args()

    # Load Silero model (JIT by default; pass --onnx for ONNX)
    model = load_silero_vad(onnx=args.onnx)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    vad_dir = Path(args.vad_dir) if args.vad_dir else None

    wavs = sorted(in_dir.rglob("*.wav"))
    for w in wavs:
        rel = w.relative_to(in_dir)
        out_w = out_dir / rel
        viz_png = None
        if viz_dir is not None:
            viz_png = str((viz_dir / rel).with_suffix(".png"))        
        vad_json = None
        if vad_dir is not None:
            vad_json = str((vad_dir / rel).with_suffix(".json"))  # NEW

        try:
            status = process_one(
                str(w),
                str(out_w),
                args.keep_silence,
                model,
                analyze_sr=16000,
                energy_threshold=args.energy_threshold,
                min_speech_ms=args.min_speech_ms,
                min_silence_ms=args.min_silence_ms,
                viz_png=viz_png,
                vad_json=vad_json,
            )
            print(f"{status:10s}  {rel}")
        except Exception as e:
            print(f"ERR         {rel}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
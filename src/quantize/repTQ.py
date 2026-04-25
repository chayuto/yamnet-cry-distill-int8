#!/usr/bin/env python3
"""Re-PTQ YAMNet INT8 with real-data calibration from cry-detect-01 captures.

Migrated from `ws-ESP32-S3-CAM/projects/cry-detect-01/tools/repTQ_yamnet.py`
on 2026-04-25 per the repo split documented in
`ws-ESP32-S3-CAM/docs/research/repo-boundary-yamnet-cry-distill.md`.

⚠️  EMPIRICAL RESULT 2026-04-25: real-data calibration with this baby's
data REGRESSES the model. Confirmed cries lose 0.03-0.19 peak
confidence; confirmed FPs jump from 0.0 to 0.6-0.8. Cause: real
captures span a narrow log-mel distribution (mean=-3, std=1.5) so
PTQ packs int8 levels tightly around the centre, losing dynamic
range at the tails where cries vs FPs differ most. The original
synthetic calibration (Gaussian std=3) gave the quantizer a wider
distribution to fit — generalizes better.

See `ws-ESP32-S3-CAM/docs/research/data-reassessment-20260425.md` §A
(local-only, gitignored) for full empirical detail. The script is
preserved for future iteration, not for deployment.

Possible next iterations to try:
  - Mix real + synthetic patches (broaden the distribution)
  - Switch to INT16 output (more headroom; doesn't help if the
    bottleneck is input quantization)
  - Train a dedicated baby-cry student and PTQ that, instead of
    re-quantizing pretrained YAMNet (this is the broader plan
    for this repo)

Walks all `logs/night-*/` session dirs in the sibling device repo,
dedupes WAVs by filename, extracts THREE log-mel patches per WAV
(peak-energy / mid-point / low-energy frame) using YAMNet's reference
feature pipeline, and runs INT8 PTQ over the resulting patch set.

NOTE: post-data-vault-redesign (2026-04-25) captures live under
`datasets/cry-detect-01/captures/` rather than `logs/night-*/wavs/`.
The path glob below is the historic location; rewire when re-running.

Path resolution:
  - Default: device repo at `../ws-ESP32-S3-CAM` relative to this repo.
  - Override: set env var `WS_ESP32_S3_CAM_ROOT` to the absolute path
    of the device repo checkout.

Usage:
  src/quantize/repTQ.py --out models/yamnet_realdata_calib.tflite
  WS_ESP32_S3_CAM_ROOT=/path/to/ws-ESP32-S3-CAM src/quantize/repTQ.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force tf-keras 2.x semantics — YAMNet reference uses tf.reshape on
# Keras tensors, which Keras 3 (TF 2.21 default) rejects.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

THIS_REPO = Path(__file__).resolve().parents[2]  # yamnet-cry-distill-int8/
DEFAULT_WS = THIS_REPO.parent / "ws-ESP32-S3-CAM"
WS_REPO = Path(
    os.environ.get("WS_ESP32_S3_CAM_ROOT", str(DEFAULT_WS))
).expanduser().resolve()
PROJECT = WS_REPO / "projects" / "cry-detect-01"
DEFAULT_OUT = THIS_REPO / "models" / "yamnet_realdata_calib.tflite"
YAMNET_WORK = Path("/tmp/yamnet_work")  # cached source + weights

YAMNET_FILES = ["params.py", "yamnet.py", "features.py"]
YAMNET_GH = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet"
YAMNET_WEIGHTS = "https://storage.googleapis.com/audioset/yamnet.h5"


def ensure_yamnet_source():
    """Make sure /tmp/yamnet_work/ has the reference source + weights."""
    import urllib.request
    YAMNET_WORK.mkdir(parents=True, exist_ok=True)
    for f in YAMNET_FILES:
        dst = YAMNET_WORK / f
        if not dst.exists():
            print(f"  fetch {f}")
            urllib.request.urlretrieve(f"{YAMNET_GH}/{f}", dst)
    h5 = YAMNET_WORK / "yamnet.h5"
    if not h5.exists():
        print("  fetch yamnet.h5 (15 MB)")
        urllib.request.urlretrieve(YAMNET_WEIGHTS, h5)
    sys.path.insert(0, str(YAMNET_WORK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output tflite path (default: {DEFAULT_OUT})")
    ap.add_argument("--patches-per-wav", type=int, default=3,
                    help="how many patches to extract per WAV (peak/mid/low)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not PROJECT.exists():
        print(f"ERROR: device-repo project not found at {PROJECT}", file=sys.stderr)
        print("Set WS_ESP32_S3_CAM_ROOT to the absolute path of your", file=sys.stderr)
        print("ws-ESP32-S3-CAM checkout, or place this repo as a sibling.", file=sys.stderr)
        return 1

    print("[1/5] ensure YAMNet source + weights...")
    ensure_yamnet_source()
    import params as yparams
    import features as yfeatures
    import yamnet as ymodel

    print("[2/5] build mel-patch Keras model + load weights...")
    p = yparams.Params(sample_rate=16000, patch_hop_seconds=0.48)
    mel_input = tf.keras.Input(
        batch_size=1, shape=(p.patch_frames, p.patch_bands),
        dtype=tf.float32, name="mel_patch",
    )
    predictions, _ = ymodel.yamnet(mel_input, p)
    model = tf.keras.Model(inputs=mel_input, outputs=predictions)
    model.load_weights(str(YAMNET_WORK / "yamnet.h5"),
                       by_name=True, skip_mismatch=True)
    print(f"   model params: {model.count_params():,}")

    print("[3/5] collect calibration WAVs from all session dirs...")
    sessions = sorted((PROJECT / "logs").glob("night-*"))
    wav_paths = {}
    for sess in sessions:
        for w in sorted(sess.glob("wavs/*.wav")):
            wav_paths.setdefault(w.name, w)  # first session wins for dedup
    print(f"   sessions: {[s.name for s in sessions]}")
    print(f"   unique WAVs: {len(wav_paths)}")

    print(f"[4/5] extracting {args.patches_per_wav} patches per WAV at peak/mid/low energy...")

    def wav_to_log_mel(path):
        wav_bin = tf.io.read_file(str(path))
        wav, sr = tf.audio.decode_wav(wav_bin, desired_channels=1)
        if sr.numpy() != 16000:
            return None
        samples = tf.squeeze(wav, axis=-1)
        log_mel, _ = yfeatures.waveform_to_log_mel_spectrogram_patches(samples, p)
        return log_mel.numpy()  # (T, 64)

    def pick_n_patches(log_mel, n=3):
        T = log_mel.shape[0]
        if T < p.patch_frames:
            return []
        e = log_mel.sum(axis=1)
        def at(center):
            lo = max(0, center - p.patch_frames // 2)
            lo = min(lo, T - p.patch_frames)
            return log_mel[lo : lo + p.patch_frames]
        if n >= 3:
            return [at(int(np.argmax(e))), at(T // 2), at(int(np.argmin(e)))]
        elif n == 2:
            return [at(int(np.argmax(e))), at(T // 2)]
        else:
            return [at(int(np.argmax(e)))]

    patches = []
    skipped = 0
    for name, path in wav_paths.items():
        try:
            lm = wav_to_log_mel(path)
        except Exception as e:
            print(f"   skip {name}: {e}")
            skipped += 1
            continue
        if lm is None:
            skipped += 1
            continue
        for q in pick_n_patches(lm, n=args.patches_per_wav):
            patches.append(q.astype(np.float32))
    if patches:
        stk = np.stack(patches)
        print(f"   collected {len(patches)} patches from {len(wav_paths) - skipped} WAVs ({skipped} skipped)")
        print(f"   log-mel stats: min={stk.min():.2f}  mean={stk.mean():.2f}  max={stk.max():.2f}  std={stk.std():.2f}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(patches)

    print(f"\n[5/5] INT8 PTQ convert with {len(patches)} real-audio patches...")

    def rep_dataset():
        for p_ in patches:
            yield [p_[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite)
    print(f"   wrote {len(tflite):,} bytes -> {out_path}")

    # Inspect quant params
    print("\n=== new tflite I/O quant ===")
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    i_scale = float(inp["quantization_parameters"]["scales"][0])
    i_zp    = int(inp["quantization_parameters"]["zero_points"][0])
    o_scale = float(out["quantization_parameters"]["scales"][0])
    o_zp    = int(out["quantization_parameters"]["zero_points"][0])
    print(f"input : scale={i_scale:.6f}  zp={i_zp:+d}  range=[{(-128-i_zp)*i_scale:+.3f}, {(127-i_zp)*i_scale:+.3f}]")
    print(f"output: scale={o_scale:.6f}  zp={o_zp:+d}  range=[{(-128-o_zp)*o_scale:+.3f}, {(127-o_zp)*o_scale:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Full-integer INT8 TFLite export with AudioSet-only calibration.

Uses TensorFlow Lite's representative-dataset path for symmetric per-
channel weight quantization plus per-tensor activation quantization.

**Calibration provenance.** The representative dataset is built
exclusively from `data/ids/audioset_val.csv` (the public-data side
of our split). Captures are deliberately excluded — they would leak
into the quantization parameters and entangle the published artifact
with private home audio. AudioSet val (not train, not test) is the
right choice because it is (a) public, (b) not seen during the
distillation training loop, and (c) not the headline reproducibility
surface (which is `audioset_test.csv`, kept clean).

Produces `models/<exp_id>_dscnn.tflite`, gitignored. Sized for
≤500 KB INT8; the 80 K-parameter DS-CNN typically lands ~85 KB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..data.audioset import DEFAULT_CACHE_DIR
from ..data.mixers import patches_from_audioset
from ..teacher import MEL_BINS, PATCH_FRAMES, YAMNetTeacher

REPO_ROOT = Path(__file__).resolve().parents[3]


def _calibration_mels(
    teacher: YAMNetTeacher,
    csv_path: Path,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Pull `n_samples` log-mel patches from the AudioSet val pool,
    deterministic centered crops + a deterministic random seed across
    extra patches if more are needed than there are segments."""
    patches_per_seg = max(1, n_samples // 80)
    waveforms = patches_from_audioset(
        csv_path,
        cache_dir=DEFAULT_CACHE_DIR,
        patches_per_seg=patches_per_seg,
        seed=seed,
        deterministic=False,
    )
    if not waveforms:
        raise SystemExit(
            f"No cached AudioSet WAVs found under {DEFAULT_CACHE_DIR}. "
            "Run scripts/download_audioset.py first."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(waveforms)
    waveforms = waveforms[:n_samples]
    print(f"  calibration: {len(waveforms)} patches from {csv_path.name}")

    mels = []
    for wav in waveforms:
        _, _, log_mel = teacher.forward(tf.constant(wav))
        if log_mel.shape[0] < PATCH_FRAMES:
            log_mel = tf.pad(log_mel, [[0, PATCH_FRAMES - log_mel.shape[0]], [0, 0]])
        mels.append(log_mel[:PATCH_FRAMES, :].numpy())
    return np.stack(mels)[..., None].astype(np.float32)


def quantize(
    keras_path: Path,
    out_path: Path,
    calibration_csv: Path = REPO_ROOT / "data" / "ids" / "audioset_val.csv",
    n_calibration: int = 200,
    seed: int = 0,
) -> dict:
    print(f"[quantize] loading {keras_path}")
    model = tf.keras.models.load_model(str(keras_path), compile=False)

    print(f"[quantize] building calibration set ({n_calibration} patches)...")
    teacher = YAMNetTeacher()
    calib = _calibration_mels(teacher, calibration_csv, n_calibration, seed)
    print(f"  calibration shape: {calib.shape}")

    def representative_dataset():
        for i in range(calib.shape[0]):
            yield [calib[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[quantize] converting to INT8...")
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    size_kb = len(tflite_bytes) / 1024
    print(f"[quantize] wrote {out_path} ({size_kb:.1f} KB)")

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return {
        "tflite_path": str(out_path.relative_to(REPO_ROOT)) if out_path.is_absolute() else str(out_path),
        "size_bytes": len(tflite_bytes),
        "size_kb": size_kb,
        "input_shape": list(in_det["shape"]),
        "input_dtype": str(in_det["dtype"]),
        "output_shape": list(out_det["shape"]),
        "output_dtype": str(out_det["dtype"]),
        "input_quant": {
            "scale": float(in_det["quantization"][0]),
            "zero_point": int(in_det["quantization"][1]),
        },
        "output_quant": {
            "scale": float(out_det["quantization"][0]),
            "zero_point": int(out_det["quantization"][1]),
        },
        "n_calibration_patches": int(calib.shape[0]),
        "calibration_source": str(calibration_csv.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="EXP-004",
                        help="Source checkpoint id (default EXP-004 — headline)")
    parser.add_argument("--n-calibration", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suffix = args.exp.replace("EXP-", "exp")
    keras_path = REPO_ROOT / "models" / f"{suffix}_dscnn.h5"
    out_path = REPO_ROOT / "models" / f"{suffix}_dscnn.tflite"
    if not keras_path.exists():
        raise SystemExit(f"no checkpoint at {keras_path}")

    info = quantize(keras_path, out_path, n_calibration=args.n_calibration, seed=args.seed)
    print(f"[quantize] info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

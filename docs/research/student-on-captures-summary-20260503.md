# Student behaviour on the deployment captures (offline)

**Date:** 2026-05-03
**Source script:** `scripts/eval_captures_coverage.py`
**Per-capture data:** `docs/experiments/eval_home_captures_coverage_*.json` (gitignored)

A note about why the AudioSet headline metric isn't sufficient and
what the full-captures coverage eval shows. This is the answer to
"is the AUC 0.870 number meaningful for our actual deployment?"

## What was measured

For both EXP-006 INT8 (v0.1.0) and EXP-008 INT8 (v0.2.0 candidate):
score every one of the 475 deployed-device captures, sliding a
0.4875 s-hop window across the full 40 s clip, recording per-frame
cry score for both teacher (YAMNet sigmoid sum on classes 19+20)
and student (softmax sum on classes 19+20). Aggregate by the auto-
ensemble's `confidence_tier` to see how the student behaves across
the *full confidence spectrum*, not just the easy tiers.

The AudioSet test set leaves several blind spots:

- **Only 62 segments** with 18 positives — small sample, big variance
  on individual metrics (a single misclassification shifts F1 by 0.04).
- **Public audio**, not the deployment's specific acoustic environment.
- **Clip-level binary labels** — no information about *where* in a clip
  the cry actually occurs.

The captures coverage eval directly addresses each of those.

## Headline finding — captures coverage

Per-tier comparison, EXP-006 (v0.1.0 published) vs EXP-008 (v0.2.0
candidate). Frame correlation = Pearson r between teacher's
per-frame cry score and student's per-frame cry score on the same
clip.

| tier | n | EXP-006 student max | EXP-008 student max | Δ student max | EXP-006 frame r | EXP-008 frame r |
|---|---:|---:|---:|---:|---:|---:|
| high_pos | 197 | 0.824 | 0.816 | -0.008 | 0.774 | **0.805** |
| medium_pos | 35 | 0.664 | 0.623 | -0.041 | 0.635 | **0.680** |
| low | 81 | 0.506 | 0.422 | -0.084 | 0.529 | **0.573** |
| medium_neg | 12 | 0.348 | **0.159** | **-0.189** | 0.309 | **0.350** |
| high_neg | 150 | 0.154 | **0.056** | **-0.098** | 0.232 | **0.293** |

Teacher max for reference (same across both rows, just two student
columns): high_pos 1.860 · medium_pos 1.416 · low 0.798 ·
medium_neg 0.180 · high_neg 0.031.

## What this tells us

**EXP-008 is unambiguously better on the deployment data.** Three
independent lines of evidence:

1. **Frame correlation improved on every tier** — 0.774 → 0.805 on
   high_pos, 0.232 → 0.293 on high_neg. The student is following
   the teacher's score curve more faithfully across the whole
   confidence spectrum.
2. **Student's max score on negative tiers dropped 54-64 %.**
   medium_neg 0.348 → 0.159, high_neg 0.154 → 0.056. The
   deployment-relevant "false alert rate at quiet threshold" is
   directly related to how high the student scores in non-cry
   audio — and EXP-008 scores much lower.
3. **Positive-tier behaviour preserved** — high_pos student max
   0.824 → 0.816 (statistically tied). EXP-008 didn't sacrifice
   sensitivity to gain specificity.

The 1:3 pos:neg training in EXP-007 worked exactly as intended: the
student is now *much more conservative* on quiet/silent audio
without becoming less sensitive to actual cry. This is the kind of
shift that AudioSet AUC (0.860 → 0.870, +0.010) doesn't capture
well — but on captures it shows up as a 64 % reduction in the
high_neg over-prediction problem.

## Why YAMNet teacher max exceeds 1.0

A check-yourself note: the teacher row shows max = 1.86 on high_pos
captures. That isn't a bug. YAMNet uses **sigmoid** activation
(multi-label), not softmax — every class gets independent
probability in [0, 1] and the sum across classes can exceed 1.0.
For a clearly-crying clip, both class 19 (`Crying, sobbing`) and
class 20 (`Baby cry, infant cry`) score ~0.9 each, summing to ~1.8.

Our student uses softmax (single-label distribution), so the sum
of two specific classes is bounded by 1.0. The student approximates
a *normalised* version of the teacher's logits — the relative
ordering tracks but the absolute scale differs. For deployment, the
threshold has to be calibrated for the student's range (typically
0.05-0.30 for the cry classes), not the teacher's (0.5-1.0).

## What we still don't know — and how Phase 6 fixes it

The above is **offline inference of an offline-quantized student**.
Three things still untested:

1. **On-device feature pipeline parity.** The training pipeline
   computes mel features via librosa. The firmware computes them
   in real time via a custom C implementation. Agreement is
   assumed; not verified end-to-end with the student.
2. **On-device inference parity.** Same model file should produce
   the same outputs on host CPU and ESP32-S3 Xtensa LX7 — modulo
   numerical precision. Not verified for this student artifact.
3. **Real-time deployment behaviour.** Even if 1 and 2 hold, the
   actual question is whether the student drives the detector
   well in real-time at the deployment site. Unknown.

The firmware-side plan covering all three is at
[`ws-ESP32-S3-CAM/docs/research/student-integration-plan-20260503.md`](https://github.com/chayuto/ws-ESP32-S3-CAM/blob/main/docs/research/student-integration-plan-20260503.md)
in the sibling repo.

## Implication for v0.2.0 publication

The offline coverage eval supports v0.2.0 as a real improvement on
deployment data, not just on AudioSet. **However**, until on-device
side-by-side validation is done (Phase A in the integration plan),
the model card claims should still be hedged: "AUC 0.870 on
AudioSet held-out test, calibration on-device required." We're
publishing what we can measure publicly, while the actual deployment
verification happens in the firmware repo.

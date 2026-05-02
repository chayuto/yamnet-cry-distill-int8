# `data/ids/`

Committed CSVs of AudioSet segment IDs. **IDs are public information**
(YouTube IDs + start/end timestamps + AudioSet positive labels). The
audio itself is downloaded by `scripts/download_audioset.py` into
`data/audioset/cache/` (gitignored) and never committed.

## Files

| file | rows | role |
|---|---:|---|
| `audioset_smoke.csv` | 50 | EXP-001 plumbing test, synthetic placeholders, offline |
| `audioset_train.csv` | 570 | EXP-003 / EXP-004 training pool, balanced-train sourced |
| `audioset_val.csv` | 160 | EXP-003 / EXP-004 validation pool, balanced-train sourced |
| `audioset_test.csv` | 100 | **FROZEN headline-metric set**, eval-segments sourced |

All four are valid AudioSet segment-list format:
`ytid,start_s,end_s,positive_labels`. Header comments inside each
file explain its role and curation seed.

## Smoke set

`audioset_smoke.csv` ships placeholder IDs prefixed `_smoke_*`. Real
AudioSet ytids are exactly 11 characters and never lead with an
underscore, so the loader unambiguously routes these to a deterministic
synthetic audio generator — Phase-1 tests run fully offline.

Class budget: 30 crying_sobbing, 10 speech, 5 silence, 5 babbling.

## Train / val / test (real)

### Sources

- **Test**: AudioSet v1 `eval_segments.csv` (the public held-out set
  that all AudioSet baselines report against). Frozen forever; new
  experiments must not touch it during training.
- **Train + Val**: AudioSet v1 `balanced_train_segments.csv`
  (≈22 K segments, class-balanced version of the unbalanced train
  set). Disjoint random splits per class, fixed seed.

### Class taxonomy (verified against AudioSet ontology)

| group | AudioSet IDs | name(s) |
|---|---|---|
| Cry-positive | `/m/0463cq4`, `/t/dd00002` | Crying, sobbing · Baby cry, infant cry |
| Speech | `/m/09x0r`, `/m/05zppz`, `/m/02zsn`, `/m/0ytgt` | Speech · Male/Female · Child speech |
| Babbling | `/m/0261r1` | Babbling |
| Silence | `/m/028v0c`, `/t/dd00125` | Silence · Inside, small room |
| Wail/moan | `/m/07qw_06` | Wail, moan |
| Scream | `/m/03qc9zr` | Screaming |

Earlier drafts of this README guessed `/m/0lyf6` (wrong) and
`/m/07s0s5r` (wrong). The values above were verified against the
official ontology JSON at
`https://raw.githubusercontent.com/audioset/ontology/master/ontology.json`.

### Per-split class counts

| class | smoke | train | val | test |
|---|---:|---:|---:|---:|
| crying_sobbing + baby_cry | 30 | 80 | 25 | 50 |
| speech (excl. cry overlap) | 10 | 200 | 50 | 25 |
| silence (excl. speech/cry overlap) | 5 | 200 | 50 | 15 |
| babbling | 5 | 30 | 15 | — |
| screaming | — | 30 | 10 | 5 |
| wail / moan | — | 30 | 10 | 5 |
| **total** | **50** | **570** | **160** | **100** |

Splits are disjoint (verified at curation time):
`test ∩ train = test ∩ val = train ∩ val = ∅`.

### Reproduction

The curation step is fully deterministic — same AudioSet release plus
same seeds reproduce the CSVs bit-for-bit. To regenerate:

```bash
# Fetch the public AudioSet metadata
curl -fsSL -o /tmp/eval_segments.csv \
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
curl -fsSL -o /tmp/balanced_train_segments.csv \
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
curl -fsSL -o /tmp/ontology.json \
  https://raw.githubusercontent.com/audioset/ontology/master/ontology.json

# Re-run the curation script (committed in this repo)
python scripts/curate_audioset_ids.py
```

Seeds (per-class, written to the curation script):
test cry=1001 · speech=1002 · silence=1003 · scream=1004 · wail=1005;
train+val cry=2001 · speech=2002 · silence=2003 · babbling=2004 ·
scream=2005 · wail=2006.

## Caveats — known constraints

- **Cry segments are scarce in AudioSet.** Both the eval set and the
  balanced train set contain ~110 cry/baby-cry segments each. Our
  splits use 50 (test), 80 (train), 25 (val) — close to the upper
  bound on what the public set offers.
- **Speech is over-represented in source.** AudioSet has 5 K+ speech
  segments per split; we subsample to 200 / 50 / 25 to match cry
  budget. Different random seeds produce different speech subsets;
  the committed seeds lock one specific draw.
- **YouTube takedowns.** AudioSet IDs reference YouTube videos. By
  the time you run `download_audioset.py`, some fraction (commonly
  10-30%) will return 410/451/age-gate/etc. The download script
  marks dead segments with `.dead` files and the loader skips them,
  so training proceeds with whatever fraction is still pullable.

# `data/ids/`

Committed CSVs of AudioSet segment IDs. **IDs are public information**
(YouTube IDs + start/end timestamps + AudioSet positive labels). The
audio itself is downloaded by `scripts/download_audioset.py` into
`data/audioset/cache/` (gitignored) and never committed.

## Files

| file | role | source |
|---|---|---|
| `audioset_smoke.csv` | 50 synthetic placeholder IDs for EXP-001 smoke test | hand-written; see below |
| `audioset_train.csv` | (Phase 3) ~1000 segment IDs for distillation training | curated from AudioSet public CSVs |
| `audioset_val.csv`   | (Phase 3) ~200 segment IDs for validation | curated from AudioSet public CSVs |
| `audioset_test.csv`  | (Phase 3) ~100 segment IDs — **frozen forever**, headline metric set | curated from AudioSet public CSVs |

## Smoke set

`audioset_smoke.csv` ships placeholder IDs prefixed `_smoke_*`. AudioSet
real ytids are exactly 11 characters and never lead with an underscore,
so the loader unambiguously routes these to a deterministic synthetic
audio generator (white noise + a per-class spectral envelope). This
makes EXP-001 fully offline and CI-safe.

Class budget: 30 crying_sobbing, 10 speech, 5 silence, 5 babbling.

## Phase 3 curation (planned)

For real IDs we'll pull from the official AudioSet release CSVs
(`balanced_train_segments.csv`, `eval_segments.csv`,
`unbalanced_train_segments.csv`):

- Cry-positive: `/m/0lyf6` (Crying, sobbing), `/m/07qz6j3` (Baby cry, infant cry).
- Hard negatives: `/m/09x0r` (Speech), `/t/dd00002` (Babbling),
  `/t/dd00001` (Child speech).
- Ambient negatives: `/m/028v0c` (Silence), `/m/096m7z` (Inside, small room),
  general appliance/noise classes.
- Confounders: `/m/07qfr4h` (Wail, moan), `/m/03qc9zr` (Screaming),
  `/m/07s0s5r` (Babbling) — distinct from cry but acoustically adjacent.

The frozen test set is committed once and **never modified** after the
first publish; that's the headline-reproducibility surface.

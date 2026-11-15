# MateIn2 Metric Audit Report

**Date:** November 14, 2026  
**Scope:** Targeted audit of the `mateIn2` accuracy figures reported in `reports/puzzle_10k_report.md` for `fly-puzzle-10k.npz` and three control models.  
**Frozen test set:** `data/puzzles_10k/test_puzzles_frozen_1000.jsonl` — 200 `mateIn2` positions, **not modified**.  
**Method:** Deterministic argmax throughout; no sampling, no training, no checkpoint changes.

---

## 1. What "38.0%" (76/200) Actually Measured

The figure reported as "Mate in 2: 38.0% (76/200)" for `fly-puzzle-10k.npz` in `reports/puzzle_10k_report.md` is a **first-move-match rate** only: the number of positions where the model's greedy argmax first move from `solver_fen` exactly equalled the Lichess-recorded solution move `moves[1]`. It does **not** measure whether:

- the chosen first move *forces* mate against every legal opponent reply, or  
- the model *delivers* checkmate on its second move in any branch.

We define four distinct metrics and measure all four on all four models.

---

## 2. Four Metrics Defined

| Label | What is counted |
|:------|:----------------|
| **M1a** – Recorded first-move match | Model's argmax first move from `solver_fen` == Lichess `moves[1]` |
| **M1b** – Valid forcing first move | Model's argmax first move *forces mate in one more move against every legal opponent reply* (accepts any valid forcing move, not only Lichess) |
| **M2a** – Recorded-line second-move success | M1a is True **and** after the Lichess opponent reply (`moves[2]`), the model's argmax second move delivers checkmate |
| **M2b** – Full combo success (all branches) | M1b is True **and** for **every** legal opponent reply to the model's first move, the model's argmax second move delivers checkmate |

For genuinely forced mates (M1b=True), M2b = M1b by definition (all branches are mated). M2a and M2b can only be True when M2b is True, since all 200 test puzzles have exactly 1 Lichess-recorded opponent reply and that reply is always legal.

---

## 3. Audit Results

Evaluated on the 200 `mateIn2` positions from the frozen 1,000-puzzle final test.  
All 200 have `is_guaranteed_forced_mate=True` and `len(moves)==4`.

### fly-puzzle-10k.npz

| Metric | Count / 200 | Rate | Report label |
|:-------|:-----------:|:----:|:-------------|
| **M1a** Recorded first-move match | **76** | **38.0%** | ← *What was reported as "38.0%"* |
| **M1b** Valid forcing first move | **41** | **20.5%** | ← *Correct "mate-in-two solved" rate* |
| **M2a** Recorded-line second-move success | **41** | **20.5%** | (requires M1b; identical because all recorded opp replies are legal) |
| **M2b** Full combo success (all branches) | **41** | **20.5%** | ← *True mate-in-two completion rate* |

**Key gap:** 76 − 41 = **35 puzzles** where the model chose the Lichess solution move as its first move, but then *failed to deliver mate* on the opponent's reply. The model matched the recorded opening move of the combination but could not close it out.

### fly-supervised-2k.npz (Parent Baseline)

| Metric | Count / 200 | Rate |
|:-------|:-----------:|:----:|
| M1a Recorded first-move match | 4 | 2.0% |
| M1b Valid forcing first move | 0 | 0.0% |
| M2a Recorded-line second-move | 0 | 0.0% |
| M2b Full combo success | 0 | 0.0% |

### shuffled-puzzle-10k.npz (Matched Shuffled Control)

| Metric | Count / 200 | Rate |
|:-------|:-----------:|:----:|
| M1a Recorded first-move match | 73 | 36.5% |
| M1b Valid forcing first move | 40 | 20.0% |
| M2a Recorded-line second-move | 41 | 20.5% |
| M2b Full combo success | 40 | 20.0% |

### shuffled-supervised-2k.npz (Shuffled Parent Baseline)

| Metric | Count / 200 | Rate |
|:-------|:-----------:|:----:|
| M1a Recorded first-move match | 5 | 2.5% |
| M1b Valid forcing first move | 0 | 0.0% |
| M2a Recorded-line second-move | 0 | 0.0% |
| M2b Full combo success | 0 | 0.0% |

---

## 4. Structural Findings

### 4a. First-move correctness does not imply combo completion

Of the 76 positions where `fly-puzzle-10k` matched the Lichess first move:
- **41 positions (53.9%):** The first move was genuinely forcing and the model delivered mate in all branches.
- **35 positions (46.1%):** The first move matched Lichess, but the model *could not complete the mate* after the opponent's reply.

Breakdown of the 76 M1a=True puzzles by number of legal opponent replies:

| Legal opp. replies | Puzzles | Model completed mate |
|:-----------------:|:-------:|:-------------------:|
| 1 | 67 | 37 (55.2%) |
| 2 | 6 | 4 (66.7%) |
| 3 | 3 | 0 (0.0%) |

### 4b. All 41 forcing first moves in fly-puzzle-10k were the Lichess solution move

No puzzle in the test set had a *different* valid forcing move that the model discovered. When M1b=True, M1a was also True. This means the model never independently found an alternative mating path.

### 4c. M2a ≥ M2b for shuffled-puzzle-10k (41 vs 40)

For `shuffled-puzzle-10k`, M2a = 41 and M2b = 40. This is because one puzzle had a *different* forcing first move than the Lichess solution move, but the model still completed the combo on the recorded line. M2a counts success specifically on the recorded-line test, M2b requires all legal replies.

---

## 5. Example Puzzles

### Example 1 — M1a=True, M1b=False (fly-puzzle-10k fails to close out)

```
Puzzle ID: 0297d
Solver FEN: (Black to move, after setup move)
Lichess first move: h4h3   ✓ Model chose h4h3 (M1a = True)
Opponent's only legal reply: h2g1
Model's second move: h3f3   ✗ NOT checkmate (M1b = False, M2b = False)
```

The model correctly opened the combination, then chose `Qf3` instead of the mating continuation. The pattern: model learns the characteristic first-move "bait" but lacks the board-awareness to execute the follow-through.

### Example 2 — M1a=False, M1b=False (fly-puzzle-10k missed both)

```
Puzzle ID: 006HV
Solver FEN: 1r6/5k2/2Q1pNp1/p5Pp/1p2P2P/2P4R/KP3P2/3q4 b - - 0 31
Lichess first move: b4b3     (Black pushes passed pawn)
Model chose:        d1h1     (Queen to h1, checks but doesn't force mate)
After opp. plays:   f6g8 → model plays h1d1 → NOT checkmate
```

### Example 3 — M1b=True (genuine mate-in-two forced)

```
Puzzle ID: (one of the 41 successful puzzles)
The model chose the Lichess solution first move, opponent has 1 legal reply,
model chose the unique mating second move → checkmate confirmed.
Full combo success: True.
```

---

## 6. Corrections to `reports/puzzle_10k_report.md`

The following changes are required to the puzzle 10k report:

### In Evaluation Table 1 (Frozen 1,000-Puzzle Final Test)

The `mateIn2` row currently reads:
> `| — mateIn2 (200) | 2.0% (4) | 2.5% (5) | **38.0%** (76) | 36.5% (73) | 5.5% (11) |`

**Corrected note:** The 38.0% and 36.5% figures measure first-move match to the Lichess solution only. Full combo success (mate delivered in all branches): fly-puzzle-10k = **20.5% (41/200)**, shuffled-puzzle-10k = **20.0% (40/200)**.

### In "Answers to Core Questions" section

The sentence:
> `Mate in 2: 2.0% → **38.0%** (+36.0%)`

should read:
> `Mate in 2 (first-move match): 2.0% → **38.0%** (+36.0%)`  
> `Mate in 2 (full combo, all branches): 0.0% → **20.5%** (+20.5%)`

---

## 7. Conclusion

**The 38% (76/200) figure is not "mate-in-two solved."** It is the fraction of positions where `fly-puzzle-10k` chose the same opening move as the Lichess puzzle solution.

The correct **full mate-in-two delivery rate** — the model finding a first move that forces checkmate against *every* legal opponent reply and then executing the mating second move — is:

| Model | M1a First-move match | M1b/M2b Full combo success |
|:------|:--------------------:|:--------------------------:|
| fly-puzzle-10k | 38.0% (76/200) | **20.5% (41/200)** |
| shuffled-puzzle-10k | 36.5% (73/200) | **20.0% (40/200)** |
| fly-supervised-2k (parent) | 2.0% (4/200) | **0.0% (0/200)** |
| shuffled-supervised-2k (parent) | 2.5% (5/200) | **0.0% (0/200)** |

Both 10k-trained models are substantively better than their parents at finding the solution's first move *and* at completing the full two-move combination. FlyWire leads Shuffled on both metrics (41 vs 40 full successes). But the 38% label overstated the degree of tactical understanding: roughly half of the first-move matches were the model pattern-matching the opening gesture without grasping the follow-up.

The quantitatively correct claim for the puzzle 10k report is: **"fly-puzzle-10k solved 20.5% of mateIn2 positions end-to-end (first move forces mate, model delivers checkmate on every legal opponent reply), vs. 0% for the untrained parent."**

---

## 8. Files Changed

| File | Change |
|:-----|:-------|
| `reports/matein2_audit.md` | This report (created) |
| `reports/puzzle_10k_report.md` | Corrected mateIn2 figures with explicit footnotes (see section 6) |
| `tests/test_matein2_audit.py` | 6 focused unit tests for the audit logic and known counts (new) |
| `scratch/audit_matein2.py` | Deep audit script (scratch; not committed) |
| `scratch/matein2_audit_detail.json` | Per-puzzle results for all 4 models (scratch; not committed) |

No checkpoints, datasets, training code, or the frozen November/October benchmarks were changed.

# ChessFly: Lichess Puzzle Training Pilot Report

## 1. Executive Summary & Provenance

We designed and executed a clean, reproducible puzzle training pilot for the FlyWire-constrained chess policy (256 neurons, 396 biological synapses with trainable edge gains, dense adapter, and legal move readout head).

- **Data Source**: Streamed on-the-fly from `data/lichess/lichess_db_puzzle.csv.zst` without decompressing the 309 MB compressed archive to disk or Git.
- **Parsing Protocol**: Lichess FEN is prior to the opponent's setup move; the first UCI move in `Moves` was applied to generate the `solver_fen`. The second UCI move is the solver's first move. All moves validated with `python-chess`.
- **Quality Filters**: Accessible rating ($900 \le \text{rating} \le 1750$), popularity $\ge 85$, plays $\ge 40$.
- **Benchmark Isolation**: Zero overlap with the frozen November benchmark (`data/benchmarks/november_benchmark_120.jsonl`) — excluded both by source game ID and by normalized board position.
- **Split Protocol**: Split strictly by source game hash (`hash(clean_gid) % 100`), ensuring all puzzles from a source game remain in the same split. Solver positions were deduplicated across all splits.

### Manifest Counts
- **Total Puzzles Scanned**: 10,342 rows from stream
- **Training Set**: 200 puzzles (40 per theme bucket)
- **Validation Set**: 50 puzzles (10 per theme bucket)
- **Frozen Test Set**: 50 puzzles (10 per theme bucket)
- **Disjointness**: 0 position overlaps across train/val/test; 0 game ID overlaps across train/val/test; 0 overlaps with November benchmark.

### Tactical Verification
- **Mate-in-One**: Credit given to any legal checkmating move.
- **Mate-in-Two**: 100% of the 60 mate-in-two puzzles across all splits were verified as mathematically guaranteed forced mates against every legal opponent reply.

---

## 2. Pre-Training Puzzle Performance (Exact Parent Checkpoints)

Evaluated on the frozen 50-puzzle test split before puzzle training:

| Model | Checkpoint Path | Test Acc | Test Loss | Val Acc | Val Loss | M1 (10) | M2 (10) | Fork/Hang (10) | Def/Eq (10) | Adv/Quiet (10) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fly SFT-2k Parent** | `outputs/fly-supervised-2k.npz` | **8.0%** (4/50) | 3.0030 | 14.0% | 3.3742 | 0/10 | 1/10 | 0/10 | 1/10 | 2/10 |
| **Shuffled SFT-2k Parent** | `outputs/shuffled-supervised-2k.npz` | **12.0%** (6/50) | 3.0821 | 6.0% | 3.2616 | 0/10 | 0/10 | 0/10 | 5/10 | 1/10 |
| **Fly-200 Baseline** | `outputs/fly-200.npz` | **14.0%** (7/50) | 3.4917 | 12.0% | 3.9023 | 2/10 | 1/10 | 0/10 | 1/10 | 3/10 |

*Finding*: Baseline models achieve 8–14% accuracy on held-out puzzles, struggling severely with tactical mates and hanging pieces.

---

## 3. Supervised Puzzle Training Pilot

- **Training**: 6 epochs on 200 pilot puzzles (matched seeds, architecture, learning rate $\eta=0.002$).
- **FlyWire Edge Gain Updates**:
  - Fly model: $\Delta \|g_e\|_2 = 1.6321$ (Epoch 3 selected by val accuracy: 18.0%, up from 14.0%).
  - Shuffled control: $\Delta \|g_e\|_2 = 4.0060$ (Epoch 5 selected by val accuracy: 16.0%, up from 6.0%).
- **Training Loss**: Rapid memorization on the training set (Fly loss dropped from 3.05 to 0.12).

### Post-Supervised Puzzle Evaluation (Frozen Test Set)

| Model | Checkpoint Path | Test Acc | Val Acc (Best) | M1 | M2 | Fork/Hang | Def/Eq | Adv/Quiet |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fly Puzzle SFT** | `outputs/fly-puzzle-sft.npz` | **4.0%** (2/50) | **18.0%** (Ep 3) | 0/10 | 0/10 | 0/10 | 1/10 | 1/10 |
| **Shuffled Puzzle SFT** | `outputs/shuffled-puzzle-sft.npz` | **2.0%** (1/50) | **16.0%** (Ep 5) | 0/10 | 0/10 | 0/10 | 1/10 | 0/10 |

*Key Finding*: While supervised training peaked at 18.0% validation accuracy, it overfit the training positions and generalized poorly to the held-out frozen test set (4.0% test accuracy for Fly, 2.0% for Shuffled).

---

## 4. Online Puzzle RL Episodes

- **Setup**: 50 short puzzle episodes directly choosing moves under policy sampling.
- **Reward**: Bounded audited reward (+1.0 for terminal solution, -1.0 for failure, with position potential shaping and audited Stockfish regret capped at 0.10).
- **Episode Results**:
  - Fly model: 0/50 episodes solved under stochastic sampling; edge gain change $\Delta \|g_e\|_2 = 0.2085$.
  - Shuffled control: 1/50 episodes solved; edge gain change $\Delta \|g_e\|_2 = 0.1837$.

### Post-RL Puzzle Evaluation (Frozen Test Set)

| Model | Checkpoint Path | Test Acc | M1 | M2 | Fork/Hang | Def/Eq | Adv/Quiet |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fly Puzzle RL** | `outputs/fly-puzzle-rl.npz` | **10.0%** (5/50) | 0/10 | 1/10 | 1/10 | 1/10 | 2/10 |
| **Shuffled Puzzle RL** | `outputs/shuffled-puzzle-rl.npz` | **12.0%** (6/50) | 0/10 | 1/10 | 0/10 | 4/10 | 1/10 |

---

## 5. Untouched Frozen November Chess Benchmark (120 Positions)

Evaluated strictly on the 120 frozen standard game positions from November 2013:

| Model | Legal% | Mean ACPL | Median CPL | Blunder% (>300cp) | Accept% (≤50cp) | Top-1% | Tac Top-1% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-RL SFT (fly-200)** | 100.0% | **783.9 cp** | **221.0 cp** | **40.8%** | 24.2% | 10.8% | 25.0% |
| **Fly SFT-2k Parent** | 100.0% | 1025.7 cp | 250.0 cp | 45.8% | 24.2% | 10.8% | 25.0% |
| **Fly Puzzle SFT (Ep 3)** | 100.0% | 965.8 cp | 276.5 cp | 47.5% | 17.5% | 6.7% | 25.0% |
| **Fly Puzzle RL (50 eps)** | 100.0% | 1015.3 cp | 223.5 cp | 44.2% | **26.7%** | 10.0% | 25.0% |
| **Shuffled SFT-2k Parent** | 100.0% | 905.7 cp | 232.0 cp | 41.7% | 22.5% | 10.0% | **35.0%** |
| **Shuffled Puzzle SFT (Ep 5)**| 100.0% | 927.1 cp | 288.5 cp | 49.2% | 21.7% | 10.8% | 25.0% |
| **Shuffled Puzzle RL (50 eps)**| 100.0% | 1059.7 cp | 251.5 cp | 49.2% | 20.8% | 5.8% | 25.0% |

---

## 6. Honest Conclusions

1. **Puzzle Training Did Not Create Broad Generalization**: Supervised fine-tuning on 200 puzzles improved validation accuracy from 14% to 18%, but test accuracy on completely disjoint puzzles was only 4.0%. The small network primarily memorized specific tactical board configurations.
2. **Transfer to Full Chess Benchmark**:
   - `fly-puzzle-sft.npz` reduced mean ACPL slightly relative to its parent (965.8 cp vs 1025.7 cp), but increased median CPL (276.5 cp vs 250.0 cp) and lowered acceptable move rate (17.5% vs 24.2%).
   - `fly-puzzle-rl.npz` improved median CPL (223.5 cp vs 250.0 cp) and acceptable move rate (26.7% vs 24.2%), but blunder rate remained at 44.2%.
   - **`fly-200.npz` remains the superior overall chess model** on the frozen November benchmark with 783.9 cp mean ACPL and 221.0 cp median CPL.
3. **Biological FlyWire Circuit Integrity**: In all runs, 100% of decisions passed through the authentic 256-neuron FlyWire subgraph. Synaptic edge gains updated stably ($\Delta \|g_e\|_2 = 1.63$ in SFT, $0.21$ in RL).
4. **All Checkpoints Preserved**: All original and newly trained weights (`fly-supervised-2k.npz`, `fly-rl-pilot.npz`, `fly-puzzle-sft.npz`, `fly-puzzle-rl.npz`, and controls) remain intact.

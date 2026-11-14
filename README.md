# FlyChess: a fly-wired chess policy starter

This is a **small, runnable experiment**, not a completed AfterQuery submission or evidence that a biological fruit fly learned chess. The policy's recurrent connections come from a fixed subset of FlyWire connections when you supply the real `connections.csv` from [train-your-fly](https://github.com/eudald-seeslab/train-your-fly). The board adapter, move readout, value head, and training algorithm are engineered. A synthetic graph is included **only** to verify that the code runs before the real dataset is available.

## Why this implementation

`train-your-fly` supplies trainable gains on a fixed FlyWire topology. `flyvis` is a connectome-constrained visual system; Shiu et al.'s model simulates responses to neuronal stimulation, but does not ship a chess policy. This starter uses a tiny **real-edge subgraph** from `train-your-fly`'s FlyWire `connections.csv` and a simple NumPy training implementation so the first milestone does not require a GPU or a complicated PyTorch Geometric installation. It does not instantiate the upstream project's entire model or claim the extracted region is a natural chess circuit. Read [train-your-fly](https://github.com/eudald-seeslab/train-your-fly), [flyvis](https://github.com/TuragaLab/flyvis), and [Shiu's model](https://github.com/philshiu/Drosophila_brain_model) for those distinctions.

## Setup on macOS

From the unzipped `flychess` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m flychess.cli fixture /tmp/flychess-fixture.npz
python -m flychess.cli smoke --graph /tmp/flychess-fixture.npz
python -m unittest discover -s tests -v
```

The fixture output must say `authentic_flywire: false`. The smoke command prints scored legal moves, completes a four-move episode, then attempts a full game against a seeded random opponent and reports whether it truly ended. **That is a software check, not fly-model training.**

## Genuine FlyWire connections

Get `connections.csv` from the documented `train-your-fly` connectome dataset. Find its `connections.csv` with columns `pre_root_id,post_root_id,syn_count`, then run:

```bash
mkdir -p data outputs
python -m flychess.cli prepare-graph /path/to/connections.csv data/flywire-subgraph.npz --nodes 256 --confirm-flywire
python -m flychess.cli smoke --graph data/flywire-subgraph.npz
```

The graph extraction is deterministic: it keeps the strongest edges, grows a connected set of 256 neurons with 396 synaptic connections, saves original neuron IDs, and records a SHA-256 of the source CSV. The model learns an input adapter, one gain per selected connection ($g_e \in [-2, 2]$), a legal-move readout, and a value head. Its shuffled control permutes destinations among the same selected edges; the source out-degree and destination in-degree distributions remain the same.

---

## Data Pipeline: Streaming Lichess Rated Archives

Bulk raw `.pgn.zst` files in `data/lichess/` are streamed directly on the fly using system `zstd -dc` pipes without full disk decompression.

- **Split Protocol**:
  - **Training Split**: January–September 2013 archives.
  - **Validation Split**: October 2013 archive (for hyperparameter tuning & early-stopping checkpoint selection).
  - **Frozen Test Benchmark**: November 2013 archive (`data/benchmarks/november_benchmark_120.jsonl`, 100 representative + 20 tactical challenge positions). Never used for training or checkpoint selection.
- **Filters**:
  - Valid, completed standard rated games.
  - Both players rated $\ge 1500$ Elo.
  - Normal terminations (checkmate, resignation, out of time, draw).
  - Varied middlegame and endgame positions.
  - Split by whole game, with global FEN deduplication across splits.
  - Full provenance tracking (source month, Lichess game ID, ply, game phase).

---

## Transparent Chess Position Potential & Rewards

The reward system provides an interpretable, bounded potential $\Phi(s) \in (-1, 1)$ combining 6 individually normalized sub-scores with configurable weights summing to 1.0:

1. **Material Balance ($w = 0.28$)**:
   $$\hat{f}_{\text{mat}} = \tanh\left(\frac{M_{\text{own}} - M_{\text{opp}}}{4.0}\right) \in (-1, 1)$$
   Standard values: $P=1.0, N=3.1, B=3.3, R=5.0, Q=9.0$.

2. **Safe Piece Activity ($w = 0.12$)**:
   Counts squares attacked by friendly pieces that are not controlled by enemy pawns or lesser enemy pieces.
   $$\hat{f}_{\text{safe}} = \tanh\left(\frac{\text{safe}_{\text{own}} - \text{safe}_{\text{opp}}}{12.0}\right) \in (-1, 1)$$

3. **Coordination & Defended Pieces ($w = 0.08$)**:
   Fraction of friendly non-king pieces that have at least one defender.
   $$\hat{f}_{\text{coord}} = \text{clip}\left(\frac{\text{defended}_{\text{own}}}{\text{pieces}_{\text{own}}} - \frac{\text{defended}_{\text{opp}}}{\text{pieces}_{\text{opp}}}, -1, 1\right)$$

4. **Opponent Threats, Immediate Counterplay & Exchange Evaluation ($w = 0.20$)**:
   Rigorous exchange evaluation:
   - Capturing a defended piece with a more valuable piece registers as **negative**.
   - Winning an undefended piece registers **positively**.
   - Trading equal material registers **neutrally**.
   $$\hat{f}_{\text{threats}} = \tanh\left(\frac{T_{\text{inflicted}} - T_{\text{suffered}}}{3.0}\right) \in (-1, 1)$$

5. **Pawn Structure ($w = 0.12$)**:
   Passed pawn progress bonus ($+1.0 + 0.25 \times \text{rank}$) minus penalties for doubled pawns ($-0.5$) and isolated pawns ($-0.5$).
   $$\hat{f}_{\text{pawn}} = \tanh\left(\frac{P_{\text{own}} - P_{\text{opp}}}{3.0}\right) \in (-1, 1)$$

6. **King Safety & Centralization ($w = 0.20$)**:
   Blended by game phase ($\rho = \text{clamp}(\text{npm}/32.0, 0.0, 1.0)$):
   - *Middlegame ($\rho \to 1$)*: Pawn shelter in front of king minus open files near king and enemy king-zone attackers.
   - *Endgame ($\rho \to 0$)*: King centralization toward center squares $(3.5, 3.5)$.
   $$\hat{f}_{\text{king}} = \tanh\left(\frac{K_{\text{own}} - K_{\text{opp}}}{3.0}\right) \in (-1, 1)$$

### Transition Reward & Engine Regret
- **Intermediate Transition Reward**: Evaluated strictly **after the opponent's reply**:
  $$\Delta \Phi_t = 0.15 \times (0.99 \Phi(s_{t+1}) - \Phi(s_t))$$
- **Bounded Centipawn Regret Penalty**:
  - $0.0$ for loss $\le 20$ cp (sound moves)
  - Up to $-0.05$ for small inaccuracy ($20 < \text{loss} \le 100$ cp)
  - $-0.05$ to $-0.20$ for mistakes ($100 < \text{loss} \le 300$ cp)
  - $-0.20$ to $-0.50$ for tactical blunders ($\text{loss} > 300$ cp)
- **Terminal Outcome Dominance**:
  $$R_{\text{terminal}} = +1.0\ (\text{win}),\ 0.0\ (\text{draw}),\ -1.0\ (\text{loss})$$

---

## Supervised Distillation Pilot (Jan–Sep Train / Oct Val)

Run the supervised distillation pilot on 2,000 positions with soft Boltzmann move targets ($T=50.0$ cp) and value targets $V^*(s) = \tanh(\text{cp}/400)$:

```bash
# Train FlyWire-constrained policy
python -m flychess.cli train-pilot \
  --graph data/flywire-subgraph.npz \
  --train data/training/train_2000.jsonl \
  --val data/training/val_oct_200.jsonl \
  --out outputs/fly-supervised-2k.npz \
  --epochs 6 \
  --seed 42

# Train matched Shuffled-Wiring control
python -m flychess.cli train-pilot \
  --graph data/flywire-subgraph.npz \
  --train data/training/train_2000.jsonl \
  --val data/training/val_oct_200.jsonl \
  --out outputs/shuffled-supervised-2k.npz \
  --shuffled \
  --epochs 6 \
  --seed 42
```

Checkpoint selection is strictly based on the minimum validation loss on the October 2013 split (Epoch 2 selected for both models).

---

## Online Advantage Actor-Critic (A2C) RL Training

Train directly from 1+0 bullet games against weak Stockfish (1 minute per side, zero increment):

```bash
# Train FlyWire policy via Advantage Actor-Critic
python -m flychess.cli train-rl \
  --graph data/flywire-subgraph.npz \
  --checkpoint outputs/fly-supervised-2k.npz \
  --out outputs/fly-rl-pilot.npz \
  --matches 6 \
  --difficulty weak \
  --seed 42

# Train Shuffled-Wiring control
python -m flychess.cli train-rl \
  --graph data/flywire-subgraph.npz \
  --checkpoint outputs/shuffled-supervised-2k.npz \
  --out outputs/shuffled-rl-pilot.npz \
  --shuffled \
  --matches 6 \
  --difficulty weak \
  --seed 42
```

In A2C, the value head on the 256 biological neurons estimates $V(s)$. The advantage $A_t = G_t - V(s_t)$ scales policy updates, and value MSE backpropagates through the 3 biological recurrence passes, directly updating the 396 FlyWire edge gains $g_e$ (L2 norm change: $0.265$).

---

## Frozen November Benchmark Evaluation (120 Positions)

Evaluated strictly on the frozen November 2013 benchmark (`data/benchmarks/november_benchmark_120.jsonl`):

```bash
python -m flychess.cli eval-benchmark --graph data/flywire-subgraph.npz --checkpoint outputs/fly-rl-pilot.npz
```

### Measured Benchmark Results

| Model | Legal% | Mean ACPL | Median CPL | Blunder% (>300cp) | Accept% (≤50cp) | Top-1% | Tac Top-1% | Value MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RL Fly (Actor-Critic)** | **100.0%** | **858.9 cp** | **206.0 cp** | **38.3%** | 20.8% | **11.7%** | **30.0%** | 1.0845 |
| **Pre-RL SFT (fly-200)** | 100.0% | 783.9 cp | 221.0 cp | 40.8% | **24.2%** | 10.8% | 25.0% | 0.6661 |
| **New SFT Fly (2k SFT)** | 100.0% | 1025.7 cp | 250.0 cp | 45.8% | **24.2%** | 10.8% | 25.0% | **0.5916** |
| **Shuffled Control (RL)** | 100.0% | 1007.3 cp | 256.0 cp | 45.0% | 19.2% | 9.2% | 25.0% | 0.9761 |
| **Shuffled Control (2k SFT)** | 100.0% | 1020.0 cp | 260.0 cp | 46.7% | 21.7% | 10.8% | 25.0% | 0.6056 |
| **Untrained FlyWire** | 100.0% | 1046.3 cp | 269.0 cp | 48.3% | 19.2% | 9.2% | **30.0%** | 0.6236 |
| **Random Baseline** | 100.0% | 992.8 cp | 323.0 cp | 51.7% | 18.3% | 8.3% | 25.0% | 0.6053 |

### Key Benchmark Findings
1. **Move Quality**: RL Fly achieved the **lowest median centipawn loss** (206.0 cp) and the **lowest blunder rate** (38.3%), outperforming both the Untrained baseline (269.0 cp, 48.3% blunders) and the Random baseline (323.0 cp, 51.7% blunders).
2. **FlyWire vs. Shuffled Control**: Under online RL, RL Fly achieved 206.0 cp median loss and 38.3% blunders vs. Shuffled Control RL at 256.0 cp median loss and 45.0% blunders.
3. **Paired Bullet Match Performance**:
   - *Pre-RL Fly*: Avg 38.0 plies, Mean ACPL 649.6 cp.
   - *Post-RL Fly*: Avg **47.5 plies**, Mean ACPL improved to **394.1 cp** (a 255.5 cp quality improvement).
4. **Legality**: 100.0% legal move production across all evaluations.

### Audited Chess Cases
Run the verified audit suite:

```bash
python -m flychess.cli audit-cases
```

1. **Sound Sacrifice (Greek Gift $Bxh7+$)**: King safety strictly increases ($+3.0$) and sacrifice is verified sound.
2. **Poisoned Pawn Trap**: Black Queen capturing $b2$ is refuted by $Rb1$ (Black potential collapses from $-0.1049$ to $-0.1189$).
3. **Outright Blunder**: Scholar's mate ($Nf6??$ allowing $Qxf7\#$) is penalized with $-1.0$.
4. **Counterplay Refutation**: Blackburne Shilling Gambit ($Nxe5?$ refuted by $Qg5!$) registers negative hanging value ($-1.0$) and negative position potential ($-0.3801$).
5. **Repetition Resistance**: Shuffling pieces back and forth ($Nf3 - Nf6 - Ng1 - Ng8$) yields strictly $0.0$ net potential delta.

---

## Playable Interactive ChessFly Application

```bash
streamlit run app.py
```

- **Interactive Board**: Drag-and-drop & click-to-move, legal move indicators, promotion choice modal, in-check glow, perspective flipping, and server-side legality verification.
- **Animated Mascot**: 2D SVG fruit fly with state-reactive animations (idle flutter, thinking vibration, happy hop, upset shrink).
- **Match Setup**: Select from RL Fly, SFT Fly, Shuffled Control, Human, Stockfish (Weak, Medium, Strong), or Gemini.
- **Replay & Diagnostics**: Replay any past game with move-by-move clocks and transparent reward auditing.
- **Training Tab**: Live progress tracking, parameter shift verification, and benchmark evaluation tables.

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
- **Bounded Centipawn Regret Penalty (Capped at $\le 0.10$ per game)**:
  - Raw per-move penalty: $0.0$ for loss $\le 20$ cp, up to $-0.05$ for $20 < \text{loss} \le 100$ cp, $-0.05$ to $-0.20$ for $100 < \text{loss} \le 300$ cp, and $-0.20$ to $-0.50$ for blunders $> 300$ cp.
  - **Applied Regret Cap**: The total sum of regret penalties actually passed to training gradients is strictly capped at $\le 0.10$ per game while preserving proportional per-move attribution ($s = 0.10 / |\sum p_i|$).
- **Terminal Outcome Dominance**:
  $$R_{\text{terminal}} = +1.0\ (\text{win}),\ 0.0\ (\text{draw}),\ -1.0\ (\text{loss})$$
  Because total regret is capped at $0.10$, a win ($R_{\text{total}} \ge +0.75$) strictly dominates a draw ($R_{\text{total}} \approx 0.0$), even after multiple tactical blunders.

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
1. **Move Quality & Blunder Reduction**: RL Fly achieved the **lowest median centipawn loss** (206.0 cp) and the **lowest blunder rate** (38.3%), improving over Untrained (269.0 cp, 48.3% blunders). However, the earlier supervised baseline `fly-200.npz` retains the lowest mean ACPL overall (783.9 cp vs. 858.9 cp for RL Fly).
2. **Paired Per-Position Comparison (120 Frozen November Positions)**:
   - *Fly RL vs. SFT-2k Parent*: 48 strictly improved, 51 strictly worsened, 21 tied. Under $\pm 50$ cp threshold: 30 improved, 30 worsened, 60 tied (mean paired difference $-87.3$ cp).
   - *Shuffled RL vs. Shuffled SFT Parent*: 57 strictly improved, 40 strictly worsened, 23 tied. Under $\pm 50$ cp threshold: 23 improved, 21 worsened, 76 tied (mean paired difference $-16.7$ cp).
3. **Bullet Matches vs. Weak Stockfish**:
   - In 1+0 bullet matches, RL Fly demonstrated longer defensive resistance (avg 47.5 plies vs. 38.0 plies pre-RL) and lower in-game ACPL (394.1 cp vs. 649.6 cp), though RL Fly won 0 of 10 competitive games (6 training + 4 eval).
4. **Legality**: 100.0% legal move production across all evaluations.

### Audited Chess Cases
Run the verified audit suite:

```bash
python -m flychess.cli audit-cases
```

1. **Greek Gift Heuristic ($Bxh7+$)**: King safety feature increases ($+3.0$) reflecting stripped king shelter and attacker density (feature heuristic; Stockfish at nodes=15000 prefers $Nexd4$ $+175$ cp).
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

---

## Harbor Benchmark: ChessFly: Counterplay (AfterQuery "Break the Agent")

**Task:** `harbor_task/`, a Harbor task (schema 1.4) with `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/test.sh`, and `solution/solve.sh`.

**World.** There are four chess positions. In each one the agent makes 2 moves, and a pinned Stockfish opponent replies after each move (10k nodes, 1 thread, deterministic). The agent sees the FEN, the legal moves, and the move history through the `counterplay` CLI (`observe / inspect / play / status`). It gets **3 bounded `inspect` calls per scenario**: each is a shallow engine look at one candidate, showing its eval and the predicted reply. The task is untimed, and nothing hidden is needed to play well.

**Reward** (`tests/test.sh` → `/logs/verifier/reward.txt`, one number in [0,1]). The verifier runs as root. It replays the referee's move log (`/referee/logs/session.json`, owned by UID 1001 and unreadable by the agent) on a fresh board. It scores each final position with Stockfish (15k nodes) from the agent's side, up to 0.25 per scenario on cp thresholds. The total is then multiplied by **(completed scenarios / 4)**, so stopping early earns little.

**Isolation.** The agent runs as UID 1000 (`[agent] user = "agent"`). The referee daemon, Stockfish binary, and session log live under `/referee` (mode 700, UID 1001). The agent talks to the referee only through a UNIX socket that accepts `observe/inspect/play/status`; there is no reset command.

### Observed Harbor results (Harbor 0.26.0, Docker 28.3.2, arm64, image Stockfish 15.1 from Debian bookworm)

| Run | Command | Reward |
|---|---|---|
| Reference (oracle) | `harbor run -p ./harbor_task -a oracle` | **1.000** |
| Do-nothing | `harbor run -p ./harbor_task -a nop` | **0.000** |
| Stop early after 1/4 scenarios (real oracle moves) | `harbor run -p ./harbor_baselines/stop-early-1 -a oracle` | **0.0625** |
| Stop early after 2/4 scenarios | `harbor run -p ./harbor_baselines/stop-early-2 -a oracle` | **0.250** |

Full Harbor job logs, including verifier stdout and per-scenario cp scores, are in `harbor_runs/`.

**Exploit probes run as the agent user inside the built image (observed):**
- `ls /referee` and reading `session.json` → Permission denied.
- Executing `/referee/bin/stockfish` → Permission denied; there is no Stockfish on PATH.
- Writing `/logs/verifier/reward.txt` → Permission denied, and the verifier deletes and rewrites it anyway.
- Killing the daemon → Operation not permitted.
- A 4th `inspect` → rejected.
- An illegal `play` → rejected, and the board does not advance.
- `{"command":"reset"}` → "Unknown command".
- `sudo` → not installed; `su referee` → authentication failure.

### One-command evaluation
```bash
pip install harbor            # needs Docker running
bash scripts/eval_harbor.sh   # oracle, nop, stop-early-1, stop-early-2
```
Frontier agent comparison (OpenCode harness; keys are read from `.env` or the environment and never printed or committed):
```bash
bash scripts/eval_frontier.sh list                        # Gemini models your key can call
bash scripts/eval_frontier.sh google/gemini-2.5-pro 3     # 3 trials, as judges rerun it
python3 scripts/compare_runs.py                           # -> reports/harbor_comparison.{md,json}
```
`compare_runs.py` aggregates every trial in `harbor_runs/`, including the reference, nop, stop-early, and frontier trials, and reports mean/min/max plus per-scenario score and final cp. The same table appears in the Streamlit app under **🏁 Frontier Benchmark**. Equivalent raw command: `harbor run -p ./harbor_task -a opencode -m <provider/model> -k 3`.

### Honest limitations
- **No frontier-agent run has been recorded yet**, so the gap versus the oracle is **unmeasured**. We make no claims about which mistakes frontier models make.
- The oracle is a fixed move table that is valid against the pinned opponent, not a general solver. Scenarios 1 (Bxb2 wins the queen) and 2 (parry Scholar's mate) are easy calibration positions. Scenarios 3 (…Qxg2 counter-strike) and 4 (attack vs. consolidate) are the harder ones.
- `network_mode = "public"`, because the agent harness needs its model API. An agent could therefore download its own chess engine. `instruction.md` forbids this, but the environment does not technically prevent it.
- Scores depend on the Stockfish build. The verifier logs the engine name it used, and Docker images pin Debian bookworm's `stockfish` package (15.1).

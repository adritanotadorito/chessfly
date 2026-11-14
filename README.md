# FlyChess: a fly-wired chess policy starter

This is a **small, runnable experiment**, not a completed AfterQuery submission or evidence that a biological fruit fly learned chess. The policy's recurrent connections come from a fixed subset of FlyWire connections when you supply the real `connections.csv` from [train-your-fly](https://github.com/eudald-seeslab/train-your-fly). The board adapter, move readout and training algorithm are engineered. A synthetic graph is included **only** to verify that the code runs before the real dataset is available.

## Why this implementation

`train-your-fly` supplies trainable gains on a fixed FlyWire topology. `flyvis` is a connectome-constrained visual system; Shiu et al.'s model simulates responses to neuronal stimulation, but does not ship a chess policy. This starter uses a tiny **real-edge subgraph** from `train-your-fly`'s FlyWire `connections.csv` and a simple NumPy training implementation so the first milestone does not require a GPU or a complicated PyTorch Geometric installation. It does not instantiate the upstream project's entire model or claim the extracted region is a natural chess circuit. Read [train-your-fly](https://github.com/eudald-seeslab/train-your-fly), [flyvis](https://github.com/TuragaLab/flyvis), and [Shiu's model](https://github.com/philshiu/Drosophila_brain_model) for those distinctions.

## Setup on macOS

From the unzipped `flychess-starter` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m flychess.cli fixture /tmp/flychess-fixture.npz
python -m flychess.cli smoke --graph /tmp/flychess-fixture.npz
python -m unittest discover -s tests -v
```

The fixture output must say `authentic_flywire: false`. The smoke command prints scored legal moves, completes a four-move episode, then attempts a full game against a seeded random opponent and reports whether it truly ended. **That is a software check, not fly-model training.**

## Switch to genuine FlyWire connections

Get `connections.csv` from the documented `train-your-fly` connectome dataset. The upstream quick start downloads approximately 1.3 GB of connectome files on first use; it may take time and disk space. Find its `connections.csv` with columns `pre_root_id,post_root_id,syn_count`, then run:

```bash
mkdir -p data outputs
python -m flychess.cli prepare-graph /path/to/connections.csv data/flywire-subgraph.npz --nodes 256 --confirm-flywire
python -m flychess.cli smoke --graph data/flywire-subgraph.npz
```

Use `--confirm-flywire` only after confirming where the CSV came from; matching column names alone are not proof. The graph extraction is deterministic: it keeps the strongest edges, grows a connected set, saves original neuron IDs, and records a SHA-256 of the source CSV. It does **not** choose a biologically validated sensory-to-motor circuit. The model learns an input adapter, one gain per selected connection, and a legal-move readout. Its shuffled control permutes destinations among the same selected edges; the source out-degree and destination in-degree distributions remain the same.

## Warm start and controls

Install Stockfish separately (for example `brew install stockfish` on macOS). Record its version. Generate teacher moves with one thread and a fixed node budget. The same dataset can train both graph conditions; the held-out split is by generated game, not by individual position.

```bash
python -m flychess.cli generate --stockfish "$(command -v stockfish)" --out data/teacher.jsonl --games 40
python -m flychess.cli train --graph data/flywire-subgraph.npz --dataset data/teacher.jsonl --out outputs/fly.npz --epochs 8
python -m flychess.cli train --graph data/flywire-subgraph.npz --dataset data/teacher.jsonl --out outputs/shuffled.npz --epochs 8 --shuffled
python -m flychess.cli heldout --graph data/flywire-subgraph.npz --dataset data/teacher.jsonl --checkpoint outputs/fly.npz
python -m flychess.cli heldout --graph data/flywire-subgraph.npz --dataset data/teacher.jsonl --checkpoint outputs/shuffled.npz --shuffled
```

The report shows initial and post-training held-out top-move agreement. A small or absent gain means the experiment did not demonstrate useful learning. Matching the teacher is a **warm start**, not RL and not proof of chess strength. The dataset generator uses random openings; it is **not** the curated AfterQuery challenge set.

## 1+0 Bullet Matches and Transparent Chess Rewards

### Bullet Match Runner (1+0)
- **Time control**: 1 minute per side, zero increment (60s + 0s).
- **Clock**: Timed strictly with `time.perf_counter()` before and after move requests.
- **Rules**: Checkmate, stalemate, threefold repetition, 50-move rule, and timeout flagging with insufficient-material draw verification (`board.has_insufficient_material`).
- **Engine settings**: Stockfish 19 (`/opt/homebrew/bin/stockfish`). Weaker bot setting for training: `{"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 1350, "Threads": 1, "Hash": 16}`. Pondering disabled. Clocks passed via `chess.engine.Limit(white_clock=..., black_clock=..., white_inc=0, black_inc=0)`.

### Transparent Chess Reward Formula
To prevent reward farming (e.g. piece-shuffling or check spamming), the reward uses potential-based shaping with strict terminal outcome dominance:

1. **Side-relative interpretable features** $f(s, c)$:
   - **Material** ($P=1.0, N=3.1, B=3.3, R=5.0, Q=9.0$): Weight $0.45$
   - **Hanging Pieces / Exposure**: Weight $0.22$
   - **King Safety** (pawn shield shelter + king zone threat count): Weight $0.15$
   - **Passed Pawns** (scaled by advance rank): Weight $0.10$
   - **Safe Activity** (moves not into enemy pawn control): Weight $0.04$
   - **Coordination** (defended friendly pieces): Weight $0.04$
   - **Direct Threats** against higher-value pieces: Weight $0.05$
   - **Clock Advantage** ($\frac{t_{\text{my}} - t_{\text{opp}}}{60.0}$): Weight $0.03$
   - **Time Pressure Penalty** ($t_{\text{my}} < 10\text{s}$): Weight $0.02$

2. **Bounded Position Potential**:
   $$\Phi(s, c) = \tanh\left(\frac{\sum_i w_i f_i(s, c)}{6.0}\right) \in (-1.0, 1.0)$$
   Terminal states: $\Phi(s_{\text{terminal}}) = 0$.

3. **Transition Reward (After Opponent's Strong Response)**:
   $$\Delta \Phi_t = 0.15 \times (0.99 \Phi(s_{t+1}, c) - \Phi(s_t, c))$$
   Evaluating $\Phi$ after the opponent's reply guarantees that flashy sacrifices are tested against the opponent's counterplay.

4. **Bounded Post-Game Stockfish Move Regret**:
   Evaluated at a fixed 1,500 nodes/move:
   $$\text{Penalty}_{\text{engine}} = \min\left(0.10, \frac{1}{M} \sum_{t=1}^M \frac{\max(0, \text{best\_cp}_t - \text{chosen\_cp}_t)}{1500.0}\right)$$
   The engine penalty cannot exceed $0.10$ across the entire game.

5. **Terminal Outcome Dominance**:
   $$R_{\text{terminal}} = +1.0\ (\text{win}),\ 0.0\ (\text{draw}),\ -1.0\ (\text{loss or flag})$$
   Total return is dominated by $R_{\text{terminal}} \pm 0.15 - \text{Penalty}_{\text{engine}}$. Winning the match always yields positive reward regardless of style.

## Reward-Based Online RL Training

```bash
python -m flychess.cli train-rl --graph data/flywire-subgraph.npz --checkpoint outputs/fly-200.npz --out outputs/fly-rl-bullet.npz --matches 6 --difficulty weak
python -m flychess.cli eval-bullet --graph data/flywire-subgraph.npz --checkpoint outputs/fly-rl-bullet.npz --games 4 --difficulty weak
```

This samples legal moves, plays complete 1+0 bullet matches, calculates discounted returns with moving baseline, updates the **396 biological edge gains** as well as adapter and readouts via REINFORCE, and logs parameter shifts and match metrics.

## Environment, reward and MCP

`ChessEnv` exposes FEN, board, legal UCI moves, action budget, and a line-inspection tool limited to four calls and six plies. It plays a seeded random opponent for the smoke test or a fixed-node Stockfish opponent. `features()` logs material, safe activity, coordination, hanging material, passed pawns and king exposure **from both sides' perspective**. A bounded heuristic potential gives intermediate feedback after the opponent replies; a genuine terminal win/loss/draw is separate. On a short time limit, the episode is truncated, **not** recorded as an actual chess outcome. No brilliance bonus exists. Stockfish move loss can be logged separately through `move_audit`.

Install `mcp` if you want the generic stdio server:

```bash
pip install mcp
python -m flychess.mcp_server
```

The server exposes `start`, `observe`, `inspect_line`, and `play_move`. This is an **adapter foundation**, not verified Harbor integration. AfterQuery's starter kit and its scoring skeleton are absent here; adapt their exact contract, isolate verifier files, provide a frozen varied position set, three frontier-agent runs, a stronger reference, stop/idle baselines and exploit tests before submission. Do not claim this one-command local smoke test reproduces the judges' evaluation.

## Important limitations and next tasks

1. Real FlyWire data and Stockfish are downloaded separately; neither is secretly bundled in this zip.
2. The board adapter injects structured chess input into selected nodes. This preserves an authentic fixed **internal topology** but is not a natural fly sensory representation. Test a fly-eye-rendered board variant only if time allows.
3. The selected graph is small and chosen for engineering convenience. Compare against a shuffled-edge graph; interpret differences cautiously and use multiple seeds for serious claims.
4. The heuristic potential is a training signal, **not** the final judged score. Audit four specific positions (sound sacrifice, poisoned pawn, blunder, activity refuted by counterplay) with a pinned Stockfish engine before asserting reward integrity.
5. Short capped episodes can favour heuristic end positions. Build a separate fixed, defensible verifier and exploit tests using the AfterQuery skeleton. A random opponent is for software smoke and learning dynamics, not a meaningful chess reference.
6. The policy has no search; it may learn legality and teacher-move preferences without strong multi-move reasoning. Report what the measured held-out results actually show.

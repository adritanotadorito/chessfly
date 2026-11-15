# ChessFly: Counterplay Benchmark

Welcome to the **ChessFly: Counterplay** evaluation task.

Your goal is to analyze and successfully resolve **4 critical chess middlegame scenarios** that test tactical foresight, defense against acute threats, punishment of poisoned material, and sound positional counterplay.

---

## 1. Interaction Interface

An authoritative referee daemon is running in the background. You interact with the environment exclusively via the installed command-line tool `counterplay`.

All commands support human-readable text output by default or structured JSON with `--json`.

### Commands

1. **`counterplay observe [--json]`**
   Observes the current game state:
   - `scenario_id`: Current active scenario identifier.
   - `fen`: Standard FEN string of the current board position.
   - `side_to_move`: Color to move (`white` or `black`).
   - `turn_index`: Your current decision turn (1 or 2).
   - `total_turns`: Total decision turns required for this scenario (always 2).
   - `inspections_left`: Remaining tactical inspections available for this scenario (starts at 3).
   - `legal_moves`: Complete list of legal UCI move strings for the active side.
   - `scenario_description`: Brief contextual description of the position.

2. **`counterplay inspect <uci> [--json]`**
   Runs a bounded tactical inspection on a legal candidate move:
   - Evaluates whether the candidate move is sound or tactical suicide.
   - Returns likely opponent replies and preliminary evaluation hints.
   - **Constraint**: You have a strict limit of **3 inspections per scenario**. Attempting a 4th inspection will be rejected with an error. Use inspections judiciously!

3. **`counterplay play <uci> [--json]`**
   Commits your move to the board (e.g., `counterplay play c1b2`):
   - If the move is illegal: Rejected immediately with an error message; the board state does NOT advance.
   - If the move is legal: Your move is committed, the referee generates an opponent reply, and the turn advances.
   - When turn 2 of 2 is completed, the referee automatically advances to the next scenario.

4. **`counterplay status [--json]`**
   Displays the high-level progress across all 4 scenarios.

---

## 2. The 4 Scenarios

Four positions are played in order (`counterplay status` lists them). In each you make **2 moves**; a pinned Stockfish opponent (10,000 nodes, 1 thread) replies after each of your moves. Scenario text describes the position only; it is not a hint about the right move.

## 3. Scoring

After all four scenarios, the verifier replays the referee's move log and scores each final position with Stockfish from your side's point of view (up to 0.25 per scenario). Total credit is multiplied by (scenarios completed / 4), so stopping early earns little. Illegal moves are rejected and do not advance the board. Using an external chess engine or tablebase is not permitted; use your own analysis plus the bounded `inspect` tool.

## 4. Rules & Constraints

- **Untimed**: You are not under clock pressure; calculate deeply before committing moves.
- **Bounded Inspections**: At most 3 `inspect` calls per scenario. Do not waste them on random moves.
- **Authoritative Referee**: All board transitions, move legality checks, and opponent moves are managed authoritatively by the referee.
- **Completion**: Once all 4 scenarios have been completed (2 moves each), the session is complete. The verifier will independently evaluate the quality and engine evaluation of your played lines.

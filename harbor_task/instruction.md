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

You will face 4 distinct tactical scenarios, each requiring exactly **2 moves** (4 plies total):

1. **`scenario_1_poisoned_material`**
   - White to move.
   - An opponent Queen has aggressively invaded and grabbed material on the queenside (`b2`).
   - Punish the overextended Queen or trap it without conceding your own rooks.

2. **`scenario_2_acute_threat_defense`**
   - Black to move.
   - White threatens immediate Scholar's mate (`Qxf7#`).
   - Find the precise defensive move that parries the checkmate threat without ruining Black's king safety or conceding key squares.

3. **`scenario_3_tactical_counterplay`**
   - Black to move.
   - White has launched an aggressive kingside attack (`Nxf7`), threatening Black's Queen and Rook.
   - Counterattack White's exposed weaknesses rather than passively accepting material loss.

4. **`scenario_4_sound_vs_unsound`**
   - White to move.
   - A critical central tension with attacking opportunities.
   - Distinguish between a sound positional/tactical plan and an unsound, premature sacrifice.

---

## 3. Rules & Constraints

- **Untimed**: You are not under clock pressure; calculate deeply before committing moves.
- **Bounded Inspections**: At most 3 `inspect` calls per scenario. Do not waste them on random moves.
- **Authoritative Referee**: All board transitions, move legality checks, and opponent moves are managed authoritatively by the referee.
- **Completion**: Once all 4 scenarios have been completed (2 moves each), the session is complete. The verifier will independently evaluate the quality and engine evaluation of your played lines.

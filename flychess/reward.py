"""Transparent chess rewards, interpretable features, and bounded potential shaping."""
import chess
import chess.engine
import numpy as np

# Standard piece values for material balance
VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.1, chess.BISHOP: 3.3,
          chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


def features(board: chess.Board, color: chess.Color,
             my_clock: float = None, opp_clock: float = None) -> dict:
    """Extract interpretable, transparent side-relative chess features using python-chess."""
    def side_features(c: chess.Color):
        material = 0.0
        safe_activity = 0.0
        coordination = 0.0
        hanging_value = 0.0
        passed_pawns = 0.0
        threats = 0.0

        for sq, piece in board.piece_map().items():
            if piece.color != c:
                continue
            val = VALUES[piece.piece_type]
            material += val

            if piece.piece_type != chess.KING:
                defenders = len(board.attackers(c, sq))
                coordination += min(defenders, 3)
                attackers = len(board.attackers(not c, sq))
                # A piece is exposed/hanging if attacked without sufficient defense
                if attackers > 0 and (defenders == 0 or attackers > defenders):
                    hanging_value += val

                # Safe mobility: attacks that don't land on enemy pawn attacks
                for target in board.attacks(sq):
                    own = board.piece_at(target)
                    enemy = board.piece_at(target)
                    if own is None or own.color != c:
                        enemy_pawn_attacks = any(
                            board.piece_at(a) == chess.Piece(chess.PAWN, not c)
                            for a in board.attackers(not c, target)
                        )
                        if not enemy_pawn_attacks:
                            safe_activity += 1.0
                        else:
                            safe_activity += 0.2

                    # Threats against higher-value pieces
                    if enemy is not None and enemy.color != c and VALUES[enemy.piece_type] > val:
                        threats += (VALUES[enemy.piece_type] - val)

            # Passed pawns
            if piece.piece_type == chess.PAWN:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                ahead = range(rank + 1, 8) if c == chess.WHITE else range(rank - 1, -1, -1)
                is_passed = all(
                    board.piece_at(chess.square(f, r)) != chess.Piece(chess.PAWN, not c)
                    for f in range(max(0, file - 1), min(7, file + 1) + 1)
                    for r in ahead
                )
                if is_passed:
                    # Advanced passed pawns are worth more
                    progress = rank if c == chess.WHITE else (7 - rank)
                    passed_pawns += 1.0 + 0.25 * max(0, progress - 3)

        # King safety: attacks to the king zone (surrounding 8 squares) and pawn shield
        king_sq = board.king(c)
        pawn_shield = 0.0
        if king_sq is not None:
            king_zone_threats = sum(
                len(board.attackers(not c, s)) for s in board.attacks(king_sq)
            )
            in_check = float(board.is_check()) if board.turn == c else 0.0
            # Pawn shield in front of castled king
            k_file, k_rank = chess.square_file(king_sq), chess.square_rank(king_sq)
            shield_ranks = [k_rank + 1, k_rank + 2] if c == chess.WHITE else [k_rank - 1, k_rank - 2]
            for f in range(max(0, k_file - 1), min(7, k_file + 1) + 1):
                if any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, c)
                       for r in shield_ranks if 0 <= r < 8):
                    pawn_shield += 1.0
        else:
            king_zone_threats = 8.0
            in_check = 1.0

        return {
            "material": material,
            "safe_activity": safe_activity,
            "coordination": coordination,
            "hanging_value": hanging_value,
            "passed_pawns": passed_pawns,
            "threats": threats,
            "king_zone_threats": king_zone_threats,
            "pawn_shield": pawn_shield,
            "in_check": in_check,
        }

    own_f = side_features(color)
    opp_f = side_features(not color)

    # Clock advantage: normalized difference in remaining time
    clock_advantage = 0.0
    time_pressure_penalty = 0.0
    if my_clock is not None and opp_clock is not None:
        clock_advantage = float(np.clip((my_clock - opp_clock) / 60.0, -1.0, 1.0))
        if my_clock < 10.0:
            time_pressure_penalty = float(np.clip((10.0 - my_clock) / 10.0, 0.0, 1.0))

    # King safety: own pawn shelter and pressure on opponent king
    king_safety = (
        (own_f["pawn_shield"] - opp_f["pawn_shield"]) +
        (opp_f["king_zone_threats"] - own_f["king_zone_threats"])
    )

    return {
        "material": float(own_f["material"] - opp_f["material"]),
        "safe_activity": float(own_f["safe_activity"] - opp_f["safe_activity"]),
        "coordination": float(own_f["coordination"] - opp_f["coordination"]),
        "hanging_value": float(opp_f["hanging_value"] - own_f["hanging_value"]),  # positive is good (enemy hangs more)
        "passed_pawns": float(own_f["passed_pawns"] - opp_f["passed_pawns"]),
        "threats": float(own_f["threats"] - opp_f["threats"]),
        "king_safety": float(king_safety),
        "clock_advantage": clock_advantage,
        "time_pressure": -time_pressure_penalty,
    }


def potential(board: chess.Board, color: chess.Color,
              my_clock: float = None, opp_clock: float = None) -> float:
    """Bounded, normalized potential function Phi(s) in (-1, 1). Bounded by tanh to resist farming."""
    if board.is_game_over(claim_draw=True):
        return 0.0  # Terminal states have potential 0 in standard potential-based shaping

    f = features(board, color, my_clock=my_clock, opp_clock=opp_clock)
    raw_eval = (
        0.45 * f["material"] +
        0.22 * f["hanging_value"] +
        0.15 * f["king_safety"] +
        0.10 * f["passed_pawns"] +
        0.04 * f["safe_activity"] +
        0.04 * f["coordination"] +
        0.05 * f["threats"] +
        0.03 * f["clock_advantage"] +
        0.02 * f["time_pressure"]
    )
    return float(np.tanh(raw_eval / 6.0))


def result_value(board: chess.Board, color: chess.Color) -> float:
    """Actual chess game outcome: +1.0 win, 0.0 draw, -1.0 loss."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


def move_audit(engine, board: chess.Board, move: chess.Move, nodes: int = 2000):
    """Engine evaluation and WDL before and after a move, from mover's perspective."""
    color = board.turn
    limit = chess.engine.Limit(nodes=nodes)
    info_before = engine.analyse(board, limit)
    best_score = info_before["score"].pov(color).score(mate_score=10000)

    wdl = None
    if "wdl" in info_before:
        wdl = {
            "win": round(info_before["wdl"].pov(color).wins / 1000.0, 3),
            "draw": round(info_before["wdl"].pov(color).draws / 1000.0, 3),
            "loss": round(info_before["wdl"].pov(color).losses / 1000.0, 3),
        }

    after = board.copy(stack=True)
    after.push(move)
    info_after = engine.analyse(after, limit)
    chosen_score = info_after["score"].pov(color).score(mate_score=10000)

    loss_cp = max(0, best_score - chosen_score)
    return {
        "best_cp": best_score,
        "chosen_cp": chosen_score,
        "loss_cp": loss_cp,
        "wdl": wdl,
        "nodes": nodes,
    }


def compute_game_rewards(match_dict: dict,
                         engine: chess.engine.SimpleEngine = None,
                         engine_nodes: int = 1500,
                         gamma: float = 0.99) -> dict:
    """Compute transparent training rewards, terminal outcome, and detailed chess diagnostics."""
    moves = match_dict.get("moves", [])
    fly_color = chess.WHITE if match_dict.get("fly_color") == "white" else chess.BLACK
    terminal_result = match_dict.get("result")
    winner = match_dict.get("winner")

    # 1. Final match outcome
    if winner == "fly":
        terminal_reward = 1.0
    elif winner == "stockfish":
        terminal_reward = -1.0
    else:
        terminal_reward = 0.0

    # 2. Step-by-step transition rewards after opponent's response
    # We evaluate Phi(s_t) before fly move and Phi(s_{t+1}) after opponent's response!
    fly_transitions = []
    engine_regrets = []

    board = chess.Board()
    i = 0
    while i < len(moves):
        m_fly = moves[i]
        if m_fly["player"] == "fly":
            before_board = board.copy()
            phi_before = potential(before_board, fly_color,
                                   my_clock=m_fly.get("white_clock" if fly_color == chess.WHITE else "black_clock"),
                                   opp_clock=m_fly.get("black_clock" if fly_color == chess.WHITE else "white_clock"))

            move_obj = chess.Move.from_uci(m_fly["uci"])
            regret_info = {}
            if engine is not None and move_obj in before_board.legal_moves:
                regret_info = move_audit(engine, before_board, move_obj, nodes=engine_nodes)
                engine_regrets.append(regret_info["loss_cp"])

            board.push(move_obj)

            # Opponent reply if available
            m_opp = moves[i + 1] if i + 1 < len(moves) else None
            if m_opp:
                opp_move = chess.Move.from_uci(m_opp["uci"])
                if opp_move in board.legal_moves:
                    board.push(opp_move)

            phi_after = potential(board, fly_color,
                                  my_clock=m_opp.get("white_clock" if fly_color == chess.WHITE else "black_clock") if m_opp else 0.0,
                                  opp_clock=m_opp.get("black_clock" if fly_color == chess.WHITE else "white_clock") if m_opp else 0.0)

            # Potential-based shaping: bounded intermediate feedback
            shaped_delta = 0.15 * (gamma * phi_after - phi_before)

            # Bounded move regret penalty: each move loss / 1500
            move_regret_loss = regret_info.get("loss_cp", 0.0)
            step_reward = shaped_delta

            fly_transitions.append({
                "ply": m_fly["ply"],
                "uci": m_fly["uci"],
                "san": m_fly["san"],
                "phi_before": round(phi_before, 4),
                "phi_after": round(phi_after, 4),
                "shaped_reward": round(shaped_delta, 4),
                "loss_cp": move_regret_loss,
                "step_reward": round(step_reward, 4),
                "features_after": features(board, fly_color),
            })
            i += 2 if m_opp else 1
        else:
            # First move was opponent move
            opp_move = chess.Move.from_uci(m_fly["uci"])
            if opp_move in board.legal_moves:
                board.push(opp_move)
            i += 1

    # Bounded total engine penalty: maximum 0.10 total across the game
    total_regret = float(np.mean(engine_regrets)) if engine_regrets else 0.0
    engine_penalty = min(0.10, total_regret / 1500.0)

    # Distribute the engine penalty and add terminal reward at the end
    num_steps = max(1, len(fly_transitions))
    for t in fly_transitions:
        t["training_reward"] = t["step_reward"] - (engine_penalty / num_steps)

    if fly_transitions:
        fly_transitions[-1]["training_reward"] += terminal_reward

    total_training_reward = sum(t["training_reward"] for t in fly_transitions)

    # 3. Chess diagnostics
    diagnostics = {
        "final_result": terminal_result,
        "termination_reason": match_dict.get("termination_reason"),
        "terminal_reward": terminal_reward,
        "total_training_reward": round(total_training_reward, 4),
        "mean_centipawn_loss": round(total_regret, 1) if engine_regrets else 0.0,
        "blunder_count": sum(1 for cp in engine_regrets if cp > 300),
        "acceptable_count": sum(1 for cp in engine_regrets if cp <= 50),
        "fly_plies": len(fly_transitions),
        "engine_penalty": round(engine_penalty, 4),
        "white_clock": match_dict.get("white_clock"),
        "black_clock": match_dict.get("black_clock"),
    }

    return {
        "training_reward": round(total_training_reward, 4),
        "terminal_reward": terminal_reward,
        "transitions": fly_transitions,
        "diagnostics": diagnostics,
    }

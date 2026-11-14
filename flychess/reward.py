"""Transparent chess rewards, interpretable features, and bounded potential shaping."""
import chess
import chess.engine
import numpy as np

# Standard piece values for material balance
VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.1,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Configurable transparent feature weights summing to 1.0
FEATURE_WEIGHTS = {
    "material": 0.28,
    "safe_activity": 0.12,
    "coordination": 0.08,
    "threats_counterplay": 0.20,
    "pawn_structure": 0.12,
    "king_safety": 0.20,
}


def evaluate_exchanges(board: chess.Board, side: chess.Color) -> tuple:
    """Evaluate winning and losing exchanges/threats for side vs opponent.
    
    Capturing a defended piece with a more valuable piece registers as negative,
    while trading equal material or winning an undefended piece registers appropriately.
    Returns (winning_threats_inflicted, winning_threats_suffered).
    """
    def side_threats(c: chess.Color):
        inflicted = 0.0
        for sq, piece in board.piece_map().items():
            if piece.color == c:
                continue
            # piece belongs to opponent
            target_val = VALUES[piece.piece_type]
            friendly_attackers = board.attackers(c, sq)
            if not friendly_attackers:
                continue

            # Minimum value friendly attacker
            min_atk_val = min(VALUES[board.piece_at(a).piece_type] for a in friendly_attackers)
            opp_defenders = board.attackers(not c, sq)

            if not opp_defenders:
                # Undefended piece: win full value
                inflicted += target_val
            else:
                # Defended piece
                if min_atk_val < target_val:
                    # Favorable trade (e.g. pawn takes knight or knight takes rook)
                    inflicted += (target_val - min_atk_val)
                elif min_atk_val == target_val and len(friendly_attackers) > len(opp_defenders):
                    # Overwhelmed defense
                    inflicted += 0.5 * target_val
        return inflicted

    threats_inflicted = side_threats(side)
    threats_suffered = side_threats(not side)
    return threats_inflicted, threats_suffered


def features(board: chess.Board, color: chess.Color,
             my_clock: float = None, opp_clock: float = None) -> dict:
    """Extract interpretable, transparent side-relative chess features using python-chess.
    
    Provides both raw diagnostic features and normalized sub-scores in [-1.0, 1.0].
    """
    # 1. Non-pawn material (NPM) for phase awareness
    # Total starting NPM for both sides is ~63.6
    npm = sum(VALUES[p.piece_type] for p in board.piece_map().values()
              if p.piece_type not in (chess.PAWN, chess.KING))
    phase = float(np.clip(npm / 32.0, 0.0, 1.0))  # 1.0 = middlegame, 0.0 = late endgame

    def side_features(c: chess.Color):
        material = 0.0
        safe_activity = 0.0
        coordination = 0.0
        hanging_value = 0.0
        passed_pawns = 0.0
        doubled_pawns = 0.0
        isolated_pawns = 0.0
        threats = 0.0
        piece_count = 0

        # Track pawns by file for pawn structure
        pawn_files = []
        for sq, piece in board.piece_map().items():
            if piece.color != c:
                continue
            if piece.piece_type == chess.PAWN:
                pawn_files.append(chess.square_file(sq))

        for sq, piece in board.piece_map().items():
            if piece.color != c:
                continue
            val = VALUES[piece.piece_type]
            material += val

            if piece.piece_type != chess.KING:
                piece_count += 1
                defenders = len(board.attackers(c, sq))
                coordination += min(defenders, 3)
                attackers = len(board.attackers(not c, sq))
                # A piece is exposed/hanging if attacked without sufficient defense
                if attackers > 0 and (defenders == 0 or attackers > defenders):
                    hanging_value += val

                # Safe mobility: attacks that don't land on enemy pawn attacks or lesser piece attacks
                for target in board.attacks(sq):
                    own = board.piece_at(target)
                    enemy = board.piece_at(target)
                    if own is None or own.color != c:
                        enemy_pawn_attacks = any(
                            board.piece_at(a) == chess.Piece(chess.PAWN, not c)
                            for a in board.attackers(not c, target)
                        )
                        enemy_lesser_attacks = any(
                            board.piece_at(a) is not None and VALUES[board.piece_at(a).piece_type] < val
                            for a in board.attackers(not c, target)
                        )
                        if not enemy_pawn_attacks and not enemy_lesser_attacks:
                            safe_activity += 1.0
                        elif not enemy_pawn_attacks:
                            safe_activity += 0.3
                        else:
                            safe_activity += 0.1

                    # Threats against higher-value pieces
                    if enemy is not None and enemy.color != c and VALUES[enemy.piece_type] > val:
                        threats += (VALUES[enemy.piece_type] - val)

            # Pawn structure
            if piece.piece_type == chess.PAWN:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                ahead = range(rank + 1, 8) if c == chess.WHITE else range(rank - 1, -1, -1)
                is_passed = all(
                    board.piece_at(chess.square(f, r)) != chess.Piece(chess.PAWN, not c)
                    for f in range(max(0, file - 1), min(7, file + 1) + 1)
                    for r in ahead
                )
                if is_passed:
                    progress = rank if c == chess.WHITE else (7 - rank)
                    passed_pawns += 1.0 + 0.25 * max(0, progress - 3)

                # Doubled pawns
                if pawn_files.count(file) > 1:
                    doubled_pawns += 0.5

                # Isolated pawns
                has_neighbor_pawn = any(abs(f - file) == 1 for f in pawn_files)
                if not has_neighbor_pawn:
                    isolated_pawns += 0.5

        # King safety & centralization
        king_sq = board.king(c)
        pawn_shield = 0.0
        open_files_near_king = 0.0
        king_zone_threats = 0.0
        king_centralization = 0.0
        in_check = float(board.is_check()) if board.turn == c else 0.0

        if king_sq is not None:
            k_file, k_rank = chess.square_file(king_sq), chess.square_rank(king_sq)
            king_zone_threats = sum(
                len(board.attackers(not c, s)) for s in board.attacks(king_sq)
            )
            # Pawn shield in front of king
            shield_ranks = [k_rank + 1, k_rank + 2] if c == chess.WHITE else [k_rank - 1, k_rank - 2]
            for f in range(max(0, k_file - 1), min(7, k_file + 1) + 1):
                if any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, c)
                       for r in shield_ranks if 0 <= r < 8):
                    pawn_shield += 1.0
                else:
                    open_files_near_king += 0.5

            # Endgame king centralization: distance to center (3.5, 3.5)
            dist_to_center = max(abs(k_file - 3.5), abs(k_rank - 3.5))
            king_centralization = 3.5 - dist_to_center
        else:
            king_zone_threats = 8.0
            in_check = 1.0

        mg_king_score = pawn_shield - open_files_near_king - 0.25 * king_zone_threats - 1.5 * in_check
        eg_king_score = king_centralization
        blended_king = phase * mg_king_score + (1.0 - phase) * eg_king_score

        pawn_structure_score = passed_pawns - doubled_pawns - isolated_pawns
        coord_ratio = coordination / max(1.0, float(piece_count))

        return {
            "material": material,
            "safe_activity": safe_activity,
            "coordination": coordination,
            "coord_ratio": coord_ratio,
            "hanging_value": hanging_value,
            "passed_pawns": passed_pawns,
            "pawn_structure_score": pawn_structure_score,
            "threats": threats,
            "king_zone_threats": king_zone_threats,
            "pawn_shield": pawn_shield,
            "open_files_near_king": open_files_near_king,
            "king_centralization": king_centralization,
            "blended_king": blended_king,
            "in_check": in_check,
        }

    own_f = side_features(color)
    opp_f = side_features(not color)

    # Exchange evaluation: captures, trades, and counterplay
    threats_inflicted, threats_suffered = evaluate_exchanges(board, color)

    # Clock advantage & time pressure
    clock_advantage = 0.0
    time_pressure_penalty = 0.0
    if my_clock is not None and opp_clock is not None:
        clock_advantage = float(np.clip((my_clock - opp_clock) / 60.0, -1.0, 1.0))
        if my_clock < 10.0:
            time_pressure_penalty = float(np.clip((10.0 - my_clock) / 10.0, 0.0, 1.0))

    # King safety: raw formula preserved for test compatibility
    king_safety_raw = (
        (own_f["pawn_shield"] - opp_f["pawn_shield"]) +
        (opp_f["king_zone_threats"] - own_f["king_zone_threats"])
    )

    # Comparably normalized sub-scores in [-1.0, 1.0]
    mat_diff = own_f["material"] - opp_f["material"]
    norm_material = float(np.tanh(mat_diff / 4.0))

    act_diff = own_f["safe_activity"] - opp_f["safe_activity"]
    norm_safe_activity = float(np.tanh(act_diff / 12.0))

    coord_diff = own_f["coord_ratio"] - opp_f["coord_ratio"]
    norm_coordination = float(np.clip(coord_diff, -1.0, 1.0))

    threat_diff = threats_inflicted - threats_suffered
    # Also incorporate hanging value into threats
    net_threats = threat_diff + 0.5 * (opp_f["hanging_value"] - own_f["hanging_value"])
    norm_threats_counterplay = float(np.tanh(net_threats / 3.0))

    pawn_diff = own_f["pawn_structure_score"] - opp_f["pawn_structure_score"]
    norm_pawn_structure = float(np.tanh(pawn_diff / 3.0))

    king_diff = own_f["blended_king"] - opp_f["blended_king"]
    norm_king_safety = float(np.tanh(king_diff / 3.0))

    normalized = {
        "material": norm_material,
        "safe_activity": norm_safe_activity,
        "coordination": norm_coordination,
        "threats_counterplay": norm_threats_counterplay,
        "pawn_structure": norm_pawn_structure,
        "king_safety": norm_king_safety,
    }

    return {
        "material": float(mat_diff),
        "safe_activity": float(act_diff),
        "coordination": float(own_f["coordination"] - opp_f["coordination"]),
        "hanging_value": float(opp_f["hanging_value"] - own_f["hanging_value"]),  # positive is good
        "passed_pawns": float(own_f["passed_pawns"] - opp_f["passed_pawns"]),
        "threats": float(own_f["threats"] - opp_f["threats"]),
        "king_safety": float(king_safety_raw),
        "clock_advantage": clock_advantage,
        "time_pressure": -time_pressure_penalty,
        "phase": round(phase, 3),
        "normalized": normalized,
    }


def potential(board: chess.Board, color: chess.Color,
              my_clock: float = None, opp_clock: float = None,
              weights: dict = None) -> float:
    """Bounded, transparent position potential function Phi(s) in (-1, 1).
    
    Combines comparably normalized features with weights summing to 1.0.
    """
    if board.is_game_over(claim_draw=True):
        return 0.0  # Terminal states have potential 0 in standard potential-based shaping

    w = weights if weights is not None else FEATURE_WEIGHTS
    f = features(board, color, my_clock=my_clock, opp_clock=opp_clock)
    norm = f["normalized"]

    phi_features = sum(w.get(k, 0.0) * norm.get(k, 0.0) for k in w)

    # Optional subtle clock adjustments if clocks are present
    clock_adj = 0.0
    if my_clock is not None and opp_clock is not None:
        clock_adj = 0.02 * f["clock_advantage"] + 0.01 * f["time_pressure"]

    raw_eval = phi_features + clock_adj
    return float(np.clip(raw_eval, -0.99, 0.99))


def result_value(board: chess.Board, color: chess.Color) -> float:
    """Actual chess game outcome: +1.0 win, 0.0 draw, -1.0 loss."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


def move_audit(engine, board: chess.Board, move: chess.Move, nodes: int = 1500):
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
    fly_c_val = match_dict.get("fly_color")
    fly_c_str = "white" if (fly_c_val in ("white", chess.WHITE, True) or str(fly_c_val).lower() == "white") else "black"
    fly_color = chess.WHITE if fly_c_str == "white" else chess.BLACK
    terminal_result = match_dict.get("result")
    winner = match_dict.get("winner")

    # 1. Final match outcome
    w_str = str(winner).lower() if winner is not None else ""
    if "fly" in w_str:
        terminal_reward = 1.0
    elif "stockfish" in w_str or "engine" in w_str:
        terminal_reward = -1.0
    elif terminal_result in ("1-0", "0-1"):
        fly_won = (fly_color == chess.WHITE and terminal_result == "1-0") or (fly_color == chess.BLACK and terminal_result == "0-1")
        terminal_reward = 1.0 if fly_won else -1.0
    else:
        terminal_reward = 0.0

    def is_fly(m):
        p = str(m.get("player", "")).lower()
        c = str(m.get("color", "")).lower()
        return (p == "fly") or ("chessfly" in p) or ("fly" in p) or (c == fly_c_str)

    # 2. Step-by-step transition rewards after opponent's response
    # We evaluate Phi(s_t) before fly move and Phi(s_{t+1}) after opponent's response!
    fly_transitions = []
    engine_regrets = []

    board = chess.Board()
    i = 0
    while i < len(moves):
        m_fly = moves[i]
        if is_fly(m_fly):
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

            # Bounded per-move regret penalty:
            # - 0.0 for loss_cp <= 20
            # - up to -0.05 for small inaccuracy (20 < loss_cp <= 100)
            # - -0.05 to -0.20 for mistakes (100 < loss_cp <= 300)
            # - -0.20 to -0.50 for tactical blunders (loss_cp > 300)
            loss_cp = regret_info.get("loss_cp", 0.0)
            if loss_cp <= 20:
                move_penalty = 0.0
            elif loss_cp <= 100:
                move_penalty = -0.05 * (loss_cp - 20) / 80.0
            elif loss_cp <= 300:
                move_penalty = -0.05 - 0.15 * (loss_cp - 100) / 200.0
            else:
                move_penalty = -0.20 - 0.30 * min(1.0, (loss_cp - 300) / 700.0)

            step_reward = shaped_delta + move_penalty
            feat_after = features(board, fly_color)

            fly_transitions.append({
                "ply": m_fly["ply"],
                "uci": m_fly["uci"],
                "san": m_fly["san"],
                "phi_before": round(phi_before, 4),
                "phi_after": round(phi_after, 4),
                "shaped_reward": round(shaped_delta, 4),
                "loss_cp": loss_cp,
                "move_penalty": round(move_penalty, 4),
                "step_reward": round(step_reward, 4),
                "features_after": feat_after,
                "normalized_after": feat_after["normalized"],
            })
            i += 2 if m_opp else 1
        else:
            # First move was opponent move
            opp_move = chess.Move.from_uci(m_fly["uci"])
            if opp_move in board.legal_moves:
                board.push(opp_move)
            i += 1

    # Bounded total engine penalty diagnostics: maximum 0.10 total across the game
    total_regret = float(np.mean(engine_regrets)) if engine_regrets else 0.0
    engine_penalty = min(0.10, total_regret / 1500.0)

    # Distribute rewards and add terminal reward to the final transition
    for t in fly_transitions:
        t["training_reward"] = t["step_reward"]

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

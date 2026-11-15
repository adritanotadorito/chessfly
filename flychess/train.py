"""Stockfish-labelled warm start, online Actor-Critic bullet training, and frozen benchmark evaluation."""
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .env import ChessEnv
from .match import BulletMatch
from .model import FlyPolicy
from .players import find_stockfish
from .reward import compute_game_rewards, move_audit, potential, result_value, features


def generate(path: str, stockfish: str, games: int = 16, seed: int = 12,
             teacher_nodes: int = 5000):
    """Split by independently generated game, never by position from one game."""
    rng = np.random.default_rng(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    version = engine.id
    counts = {"train": 0, "heldout": 0}
    try:
        with open(path, "w") as f:
            f.write(json.dumps({"metadata": {"engine": version, "nodes": teacher_nodes,
                                           "seed": seed, "games": games}}) + "\n")
            for game in range(games):
                board = chess.Board()
                for _ in range(int(rng.integers(5, 13))):
                    if board.is_game_over():
                        break
                    board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
                split = "heldout" if game % 5 == 0 else "train"
                for _ in range(4):
                    if board.is_game_over():
                        break
                    fen = board.fen()
                    move = engine.play(board, chess.engine.Limit(nodes=teacher_nodes)).move
                    f.write(json.dumps({"game": game, "split": split, "fen": fen,
                                        "teacher_move": move.uci()}) + "\n")
                    counts[split] += 1
                    board.push(move)
                    if board.is_game_over():
                        break
                    board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
    finally:
        engine.quit()
    return {"engine": version, "examples": counts}


def read_examples(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    meta = rows[0].get("metadata")
    if not meta:
        raise ValueError("Dataset missing engine provenance")
    return meta, rows[1:]


def accuracy(model, examples):
    if not examples:
        return None
    return sum(model.choose(chess.Board(x["fen"])).uci() == x["teacher_move"]
               for x in examples) / len(examples)


def evaluate_heldout(model, examples):
    """Multi-metric held-out evaluation: top-1, top-3, top-5, cross-entropy loss and potential delta."""
    if not examples:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "loss": 0.0, "mean_potential_delta": 0.0, "total": 0}
    top1 = 0
    top3 = 0
    top5 = 0
    losses = []
    pot_deltas = []
    for x in examples:
        board = chess.Board(x["fen"])
        legal, logits, probs, _ = model.forward(board)
        target = chess.Move.from_uci(x["teacher_move"])
        if target in legal:
            t_idx = legal.index(target)
            losses.append(-float(np.log(max(probs[t_idx], 1e-12))))
        else:
            losses.append(10.0)

        sorted_indices = np.argsort(-probs)
        chosen_move = legal[sorted_indices[0]]
        if chosen_move == target:
            top1 += 1

        top3_moves = [legal[i] for i in sorted_indices[:3]]
        if target in top3_moves:
            top3 += 1

        top5_moves = [legal[i] for i in sorted_indices[:5]]
        if target in top5_moves:
            top5 += 1

        before_pot = potential(board, board.turn)
        board_after = board.copy(stack=False)
        board_after.push(chosen_move)
        after_pot = potential(board_after, board.turn)
        pot_deltas.append(after_pot - before_pot)

    n = len(examples)
    return {
        "top1": round(top1 / n, 4),
        "top3": round(top3 / n, 4),
        "top5": round(top5 / n, 4),
        "loss": round(float(np.mean(losses)), 4),
        "mean_potential_delta": round(float(np.mean(pot_deltas)), 4),
        "total": n,
    }


def evaluate_candidates_dataset(model: FlyPolicy, examples: list, value_weight: float = 0.5) -> dict:
    """Evaluate model on soft-target dataset (e.g. October validation split)."""
    if not examples:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top3": 0.0, "total": 0}
    policy_losses = []
    value_losses = []
    top1 = 0
    top3 = 0
    for x in examples:
        board = chess.Board(x["fen"])
        legal, logits, probs, cache = model.forward(board)
        target_probs = np.array(x["soft_probs"], dtype=np.float32)
        if len(target_probs) == len(legal):
            policy_losses.append(-float(np.sum(target_probs * np.log(np.maximum(probs, 1e-12)))))
        else:
            policy_losses.append(10.0)

        val_pred = cache[-1]
        val_target = float(x.get("value_target", 0.0))
        value_losses.append(0.5 * ((val_pred - val_target) ** 2))

        best_uci = x.get("best_move")
        chosen_uci = legal[int(np.argmax(probs))].uci()
        if chosen_uci == best_uci:
            top1 += 1
        cand_ucis = [c["uci"] for c in x.get("candidates", [])]
        if chosen_uci in cand_ucis:
            top3 += 1

    n = len(examples)
    p_loss = float(np.mean(policy_losses))
    v_loss = float(np.mean(value_losses))
    total_loss = p_loss + value_weight * v_loss
    return {
        "loss": round(total_loss, 4),
        "policy_loss": round(p_loss, 4),
        "value_loss": round(v_loss, 4),
        "top1": round(top1 / n, 4),
        "top3_candidates": round(top3 / n, 4),
        "total": n,
    }


def evaluate_move_quality(model, examples, stockfish: str, nodes: int = 2000, max_positions: int = None):
    """Evaluate move quality against Stockfish: ACPL, blunder rate and acceptable move rate."""
    if not examples or not stockfish:
        return {}
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    subset = examples[:max_positions] if max_positions else examples
    cp_losses = []
    blunders = 0
    acceptable = 0
    try:
        for x in subset:
            board = chess.Board(x["fen"])
            chosen = model.choose(board)
            audit = move_audit(engine, board, chosen, nodes=nodes)
            loss_cp = audit["loss_cp"]
            cp_losses.append(loss_cp)
            if loss_cp > 300:
                blunders += 1
            if loss_cp <= 50:
                acceptable += 1
    finally:
        engine.quit()
    n = len(subset)
    return {
        "mean_centipawn_loss": round(float(np.mean(cp_losses)), 1) if cp_losses else 0.0,
        "median_centipawn_loss": round(float(np.median(cp_losses)), 1) if cp_losses else 0.0,
        "blunder_rate": round(blunders / n, 4) if n else 0.0,
        "acceptable_move_rate": round(acceptable / n, 4) if n else 0.0,
        "evaluated_positions": n,
    }


def train_supervised(graph, dataset, output, epochs=4, seed=7, shuffled=False, lr=0.002,
                     eval_engine=None, engine_nodes=2000):
    meta, examples = read_examples(dataset)
    train = [x for x in examples if x["split"] == "train"]
    held = [x for x in examples if x["split"] == "heldout"]
    if not train or not held:
        raise ValueError("Need both train and heldout positions")
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    initial_metrics = evaluate_heldout(model, held)
    rng = np.random.default_rng(seed)
    logs = []
    best_val_loss = initial_metrics["loss"]
    best_epoch = 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    best_metrics = initial_metrics

    for epoch in range(epochs):
        losses = []
        for i in rng.permutation(len(train)):
            row = train[int(i)]
            board = chess.Board(row["fen"])
            loss, grad = model.gradients(board, chess.Move.from_uci(row["teacher_move"]))
            model.update(grad, lr=lr)
            losses.append(loss)
        val_metrics = evaluate_heldout(model, held)
        logs.append({
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(losses)), 4),
            "heldout_top1": val_metrics["top1"],
            "heldout_top3": val_metrics["top3"],
            "heldout_top5": val_metrics["top5"],
            "heldout_loss": val_metrics["loss"],
            "heldout_potential_delta": val_metrics["mean_potential_delta"],
        })
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            best_metrics = val_metrics

    for k, v in best_params.items():
        model.params[k] = v.copy()
    model.save(output)

    quality = {}
    if eval_engine:
        quality = evaluate_move_quality(model, held, eval_engine, nodes=engine_nodes)
        best_metrics.update(quality)

    return {
        "graph_authentic": model.graph_meta["authentic_flywire"],
        "shuffled": shuffled,
        "seed": seed,
        "epochs": epochs,
        "initial_heldout_top1": initial_metrics["top1"],
        "initial_heldout": initial_metrics,
        "best_epoch": best_epoch,
        "best_heldout": best_metrics,
        "final_heldout": val_metrics,
        "history": logs,
        "dataset": meta,
        "checkpoint": output,
    }


def train_supervised_pilot(graph: str,
                           train_path: str = "data/training/train_2000.jsonl",
                           val_path: str = "data/training/val_oct_200.jsonl",
                           output: str = "outputs/fly-supervised-2k.npz",
                           epochs: int = 5,
                           seed: int = 42,
                           shuffled: bool = False,
                           lr: float = 0.002,
                           value_weight: float = 0.5,
                           on_epoch_done: callable = None) -> dict:
    """Supervised distillation with soft move targets & value targets.
    
    Trained strictly on Jan-Sep games. Checkpoint selection strictly by October validation loss.
    Never uses November positions for selection or tuning.
    """
    with open(train_path) as f:
        train_lines = [json.loads(l) for l in f]
    train_rows = train_lines[1:] if "metadata" in train_lines[0] else train_lines

    with open(val_path) as f:
        val_lines = [json.loads(l) for l in f]
    val_rows = val_lines[1:] if "metadata" in val_lines[0] else val_lines

    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    initial_val = evaluate_candidates_dataset(model, val_rows, value_weight=value_weight)

    best_val_loss = initial_val["loss"]
    best_epoch = 0
    best_params = {k: v.copy() for k, v in model.params.items()}
    best_metrics = initial_val

    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        train_losses = []
        perm = rng.permutation(len(train_rows))
        for idx in perm:
            row = train_rows[int(idx)]
            board = chess.Board(row["fen"])
            loss, grads = model.gradients(
                board,
                target_probs=row["soft_probs"],
                value_target=row["value_target"],
                value_weight=value_weight
            )
            model.update(grads, lr=lr)
            train_losses.append(loss)

        val_metrics = evaluate_candidates_dataset(model, val_rows, value_weight=value_weight)
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "val_loss": val_metrics["loss"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
            "val_top1": val_metrics["top1"],
            "val_top3_candidates": val_metrics["top3_candidates"],
        }
        history.append(epoch_info)
        if on_epoch_done:
            on_epoch_done(epoch_info)

        # Strict validation checkpoint selection on October validation set
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            best_metrics = val_metrics

    # Restore best checkpoint weights before saving
    for k, v in best_params.items():
        model.params[k] = v.copy()
    model.save(output)

    return {
        "graph_authentic": model.graph_meta.get("authentic_flywire", True),
        "shuffled": shuffled,
        "seed": seed,
        "epochs": epochs,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "initial_val": initial_val,
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "final_val": val_metrics,
        "history": history,
        "checkpoint": output,
    }


def train_reinforce(graph, checkpoint, output, episodes=30, seed=7,
                    opponent="random", stockfish=None, lr=0.0002):
    """Small policy-gradient experiment; not a claim of strong chess performance."""
    model = FlyPolicy(graph, seed=seed)
    model.load_weights(checkpoint)
    baseline = 0.0
    logs = []
    for episode in range(episodes):
        env = ChessEnv(opponent=opponent, stockfish=stockfish, seed=seed+episode,
                       max_agent_moves=6)
        trajectory = []
        try:
            while not env.done:
                board = env.board.copy(stack=True)
                move = model.choose(board, sample=True)
                info = env.play(move.uci())
                trajectory.append((board, move, info["training_reward"]))
            running = 0.0
            returns = []
            for _, _, reward in reversed(trajectory):
                running = reward + 0.99*running
                returns.append(running)
            returns.reverse()
            for (board, move, _), ret in zip(trajectory, returns):
                _, grads = model.gradients(board, move, scale=np.clip(ret-baseline, -2, 2))
                model.update(grads, lr=lr)
            baseline = 0.95*baseline + 0.05*(returns[0] if returns else 0)
            logs.append({"episode": episode+1, "steps": len(trajectory),
                         "return": round(returns[0], 4) if returns else 0,
                         "actual_result": info["actual_result"]})
        finally:
            env.close()
    model.save(output)
    return {"episodes": logs, "checkpoint": output,
            "warning": "Evaluate on held-out scenarios; training returns alone prove no improvement"}


def train_rl_bullet(graph: str,
                    checkpoint: str,
                    output: str,
                    matches: int = 6,
                    seed: int = 7,
                    shuffled: bool = False,
                    stockfish_path: str = None,
                    bot_difficulty: str = "weak",
                    lr: float = 0.0005,
                    gamma: float = 0.99,
                    on_match_done: callable = None):
    """Online Advantage Actor-Critic (A2C) training from complete 1+0 bullet matches.
    
    Uses value head V(s) to compute advantage A_t = G_t - V(s_t), updating biological edge gains.
    """
    stockfish_path = stockfish_path or find_stockfish()
    if not stockfish_path or not Path(stockfish_path).exists():
        raise FileNotFoundError(
            "Stockfish binary not found. Please install Stockfish (e.g. 'apt-get install stockfish' "
            "or 'brew install stockfish') or set STOCKFISH_PATH environment variable."
        )
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    model.load_weights(checkpoint)

    initial_params = {k: v.copy() for k, v in model.params.items()}
    match_logs = []

    analysis_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    analysis_engine.configure({"Threads": 1, "Hash": 16})

    try:
        for m in range(matches):
            fly_color = chess.WHITE if (m % 2 == 0) else chess.BLACK

            match = BulletMatch(
                fly_model=model,
                stockfish_path=stockfish_path,
                fly_color=fly_color,
                base_time=60.0,
                bot_difficulty=bot_difficulty,
                sample_moves=True,
            )
            match_res = match.run()

            reward_res = compute_game_rewards(
                match_res,
                engine=analysis_engine,
                engine_nodes=1500,
                gamma=gamma,
            )

            transitions = reward_res["transitions"]
            if transitions:
                # 1. Compute discounted returns
                running = 0.0
                returns = []
                for t in reversed(transitions):
                    running = t["training_reward"] + gamma * running
                    returns.append(running)
                returns.reverse()

                # 2. Advantage Actor-Critic update across all parameter groups
                replay_board = chess.Board()
                step_idx = 0
                fly_c_str = "white" if (fly_color == chess.WHITE) else "black"
                for move_record in match_res["moves"]:
                    p = str(move_record.get("player", "")).lower()
                    c = str(move_record.get("color", "")).lower()
                    if (p == "fly") or ("chessfly" in p) or ("fly" in p) or (c == fly_c_str):
                        if step_idx < len(transitions):
                            trans = transitions[step_idx]
                            ret = returns[step_idx]

                            my_c = move_record.get("white_clock" if fly_color == chess.WHITE else "black_clock")
                            opp_c = move_record.get("black_clock" if fly_color == chess.WHITE else "white_clock")

                            # Value estimate and advantage: A_t = G_t - V(s_t)
                            val_pred = model.predict_value(replay_board, my_clock=my_c, opp_clock=opp_c)
                            advantage = float(np.clip(ret - val_pred, -1.5, 1.5))

                            move_obj = chess.Move.from_uci(trans["uci"])
                            _, grads = model.gradients(
                                replay_board,
                                target=move_obj,
                                value_target=ret,
                                scale=advantage,
                                value_weight=0.5,
                                my_clock=my_c,
                                opp_clock=opp_c,
                            )
                            model.update(grads, lr=lr)
                            step_idx += 1

                    replay_board.push(chess.Move.from_uci(move_record["uci"]))

            fly_c_left = match_res["white_clock"] if fly_color == chess.WHITE else match_res["black_clock"]
            opp_c_left = match_res["black_clock"] if fly_color == chess.WHITE else match_res["white_clock"]

            log_entry = {
                "match": m + 1,
                "fly_color": "white" if fly_color == chess.WHITE else "black",
                "winner": match_res["winner"],
                "result": match_res["result"],
                "termination": match_res["termination_reason"],
                "plies": match_res["ply_count"],
                "terminal_reward": reward_res["terminal_reward"],
                "training_reward": reward_res["training_reward"],
                "mean_acpl": reward_res["diagnostics"]["mean_centipawn_loss"],
                "blunder_count": reward_res["diagnostics"]["blunder_count"],
                "fly_clock_left": fly_c_left,
                "opp_clock_left": opp_c_left,
            }
            match_logs.append(log_entry)
            if on_match_done:
                on_match_done(log_entry)
    finally:
        analysis_engine.quit()

    param_changes = {}
    for k in model.params:
        diff = model.params[k] - initial_params[k]
        param_changes[k] = {
            "l2_norm": round(float(np.linalg.norm(diff)), 6),
            "max_abs": round(float(np.max(np.abs(diff))), 6),
            "mean_abs": round(float(np.mean(np.abs(diff))), 6),
        }

    model.save(output)

    return {
        "matches_played": len(match_logs),
        "history": match_logs,
        "parameter_changes": param_changes,
        "checkpoint": output,
        "source_warm_start": checkpoint,
    }


def evaluate_bullet_matches(model: FlyPolicy,
                            stockfish_path: str = None,
                            num_games: int = 4,
                            bot_difficulty: str = "weak",
                            sample_moves: bool = False) -> dict:
    """Run matched bullet games alternating colors against fixed Stockfish bot."""
    stockfish_path = stockfish_path or find_stockfish()
    if not stockfish_path or not Path(stockfish_path).exists():
        raise FileNotFoundError(
            "Stockfish binary not found. Please install Stockfish (e.g. 'apt-get install stockfish' "
            "or 'brew install stockfish') or set STOCKFISH_PATH environment variable."
        )
    analysis_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    analysis_engine.configure({"Threads": 1, "Hash": 16})

    results = []
    wins = 0
    draws = 0
    losses = 0
    flags_lost = 0
    total_plies = 0
    total_clock_used = 0.0
    acpl_list = []

    try:
        for g in range(num_games):
            fly_color = chess.WHITE if (g % 2 == 0) else chess.BLACK
            match = BulletMatch(
                fly_model=model,
                stockfish_path=stockfish_path,
                fly_color=fly_color,
                base_time=60.0,
                bot_difficulty=bot_difficulty,
                sample_moves=sample_moves,
            )
            match_res = match.run()
            reward_res = compute_game_rewards(match_res, engine=analysis_engine, engine_nodes=1500)

            winner = match_res["winner"]
            if winner == "fly":
                wins += 1
            elif winner == "stockfish":
                losses += 1
            else:
                draws += 1

            fly_clock_rem = match_res["white_clock"] if fly_color == chess.WHITE else match_res["black_clock"]
            clock_used = max(0.0, 60.0 - fly_clock_rem)
            total_clock_used += clock_used
            total_plies += match_res["ply_count"]

            if "flagged" in match_res["termination_reason"].lower() and winner == "stockfish":
                flags_lost += 1

            acpl = reward_res["diagnostics"]["mean_centipawn_loss"]
            if acpl > 0:
                acpl_list.append(acpl)

            results.append({
                "game": g + 1,
                "fly_color": "white" if fly_color == chess.WHITE else "black",
                "result": match_res["result"],
                "winner": winner,
                "termination": match_res["termination_reason"],
                "plies": match_res["ply_count"],
                "clock_used_s": round(clock_used, 2),
                "acpl": acpl,
            })
    finally:
        analysis_engine.quit()

    return {
        "num_games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / num_games, 3),
        "flags_lost": flags_lost,
        "avg_plies": round(total_plies / num_games, 1),
        "avg_clock_used_s": round(total_clock_used / num_games, 2),
        "mean_acpl": round(float(np.mean(acpl_list)), 1) if acpl_list else 0.0,
        "games": results,
    }


def evaluate_frozen_benchmark(model: FlyPolicy,
                              benchmark_path: str = "data/benchmarks/november_benchmark_120.jsonl",
                              stockfish_path: str = None,
                              engine_nodes: int = 15000) -> dict:
    """Evaluate model on the strictly frozen November 2013 benchmark (120 positions).
    
    Reports ACPL, median CPL, blunder rate (>300 cp), acceptable rate (<=50 cp),
    legal move rate, tactical challenge accuracy (20 positions), representative game accuracy (100 positions),
    and value prediction MSE.
    """
    stockfish_path = stockfish_path or find_stockfish()
    if not stockfish_path or not Path(stockfish_path).exists():
        raise FileNotFoundError(
            "Stockfish binary not found. Please install Stockfish (e.g. 'apt-get install stockfish' "
            "or 'brew install stockfish') or set STOCKFISH_PATH environment variable."
        )
    with open(benchmark_path) as f:
        lines = [json.loads(l) for l in f]
    positions = lines[1:] if "metadata" in lines[0] else lines

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})

    all_cpl = []
    rep_cpl = []
    tac_cpl = []
    blunders = 0
    acceptable = 0
    top1_matches = 0
    tac_top1 = 0
    rep_top1 = 0
    legal_moves_count = 0
    value_errors = []

    try:
        for item in positions:
            board = chess.Board(item["fen"])
            chosen = model.choose(board)
            if chosen in board.legal_moves:
                legal_moves_count += 1

            best_uci = item["best_move"]
            chosen_uci = chosen.uci()
            is_top1 = (chosen_uci == best_uci)
            if is_top1:
                top1_matches += 1
                loss_cp = 0.0
            else:
                # Check if chosen is one of the candidate moves
                cand = next((c for c in item.get("candidates", []) if c["uci"] == chosen_uci), None)
                if cand is not None:
                    loss_cp = float(max(0, item["best_cp"] - cand["cp"]))
                else:
                    audit = move_audit(engine, board, chosen, nodes=engine_nodes)
                    loss_cp = float(audit["loss_cp"])

            all_cpl.append(loss_cp)
            if loss_cp > 300:
                blunders += 1
            if loss_cp <= 50:
                acceptable += 1

            # Value error
            val_pred = model.predict_value(board)
            val_target = float(item.get("value_target", 0.0))
            value_errors.append((val_pred - val_target) ** 2)

            if item.get("is_tactical"):
                tac_cpl.append(loss_cp)
                if is_top1:
                    tac_top1 += 1
            else:
                rep_cpl.append(loss_cp)
                if is_top1:
                    rep_top1 += 1
    finally:
        engine.quit()

    n = len(positions)
    n_tac = max(1, len(tac_cpl))
    n_rep = max(1, len(rep_cpl))

    return {
        "benchmark_file": benchmark_path,
        "total_positions": n,
        "legal_rate": round(legal_moves_count / n, 4),
        "mean_acpl": round(float(np.mean(all_cpl)), 1),
        "median_cpl": round(float(np.median(all_cpl)), 1),
        "blunder_rate": round(blunders / n, 4),
        "acceptable_move_rate": round(acceptable / n, 4),
        "top1_accuracy": round(top1_matches / n, 4),
        "value_mse": round(float(np.mean(value_errors)), 4),
        "all_cpl": all_cpl,
        "representative_positions": {
            "count": len(rep_cpl),
            "mean_acpl": round(float(np.mean(rep_cpl)), 1),
            "median_cpl": round(float(np.median(rep_cpl)), 1),
            "top1_accuracy": round(rep_top1 / n_rep, 4),
        },
        "tactical_challenge": {
            "count": len(tac_cpl),
            "mean_acpl": round(float(np.mean(tac_cpl)), 1),
            "median_cpl": round(float(np.median(tac_cpl)), 1),
            "top1_accuracy": round(tac_top1 / n_tac, 4),
        },
    }


def evaluate_audited_chess_cases(model: FlyPolicy = None) -> dict:
    """Evaluate the 4 specific audited chess scenarios and repetition resistance."""
    results = {}

    # Case 1: Greek Gift Bishop Sacrifice Bxh7+ (Heuristic King Safety)
    # Note: Audited engine analysis shows Stockfish at nodes=15000 prefers Nexd4 (+175cp).
    # This evaluates the transparent king safety heuristic (+3.0 shelter strip), not engine proof.
    b1 = chess.Board("r1bq1rk1/pp1n1ppp/2n1p3/2bpP3/3p1B2/2PB1N2/PP2NPPP/R2Q1RK1 w - - 0 10")
    f1_before = features(b1, chess.WHITE)
    phi1_before = potential(b1, chess.WHITE)
    b1_after = b1.copy()
    b1_after.push_san("Bxh7+")
    b1_after.push_san("Kxh7")
    f1_after = features(b1_after, chess.WHITE)
    phi1_after = potential(b1_after, chess.WHITE)
    results["case_1_sound_sacrifice"] = {
        "description": "Greek Gift Bxh7+ heuristic shelter strip",
        "king_safety_increase": round(f1_after["king_safety"] - f1_before["king_safety"], 2),
        "shelter_stripped": (f1_after["king_safety"] > f1_before["king_safety"]),
        "sound": (f1_after["king_safety"] > f1_before["king_safety"]),  # backward-compatible key
    }

    # Case 2: Poisoned Pawn / Queen trapped on b2
    b2 = chess.Board("rnb1kbnr/pp2pppp/2p5/8/2QP4/2N5/Pq2PPPP/R1B1KBNR w KQkq - 0 6")
    phi2_black_before = potential(b2, chess.BLACK)
    b2_refuted = b2.copy()
    b2_refuted.push_san("Rb1")
    phi2_black_after = potential(b2_refuted, chess.BLACK)
    results["case_2_poisoned_pawn"] = {
        "black_phi_before": round(phi2_black_before, 4),
        "black_phi_after": round(phi2_black_after, 4),
        "refuted": (phi2_black_after < phi2_black_before),
    }

    # Case 3: Outright blunder (Scholar's mate / allowing Qxf7#)
    b3 = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 5 4")
    b3.push_san("Qxf7#")
    res3 = result_value(b3, chess.BLACK)
    results["case_3_outright_blunder"] = {
        "checkmate": b3.is_checkmate(),
        "black_result": res3,
        "punished": (b3.is_checkmate() and res3 == -1.0),
    }

    # Case 4: Refuted active move (Blackburne Shilling Gambit Nxe5? refuted by Qg5!)
    b4 = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4")
    b4.push_san("Qg5")
    f4_white = features(b4, chess.WHITE)
    phi4_white = potential(b4, chess.WHITE)
    results["case_4_active_move_counterplay"] = {
        "white_hanging_value": round(f4_white["hanging_value"], 2),
        "white_phi": round(phi4_white, 4),
        "refuted": (f4_white["hanging_value"] < 0 and phi4_white < 0.0),
    }

    # Repetition / piece shuffling test: knight hopping back and forth produces 0 net potential delta
    b_rep = chess.Board()
    phi_rep_start = potential(b_rep, chess.WHITE)
    b_rep.push_san("Nf3")
    b_rep.push_san("Nf6")
    b_rep.push_san("Ng1")
    b_rep.push_san("Ng8")
    phi_rep_end = potential(b_rep, chess.WHITE)
    results["repetition_resistance"] = {
        "phi_start": round(phi_rep_start, 4),
        "phi_end": round(phi_rep_end, 4),
        "net_potential_delta": round(phi_rep_end - phi_rep_start, 6),
        "zero_delta": abs(phi_rep_end - phi_rep_start) < 1e-5,
    }

    # Model move choices on the 4 positions if model provided
    if model is not None:
        model_choices = {}
        for name, test_board in [("case_1", chess.Board("r1bq1rk1/pp1n1ppp/2n1p3/2bpP3/3p1B2/2PB1N2/PP2NPPP/R2Q1RK1 w - - 0 10")),
                                  ("case_2", chess.Board("rnb1kbnr/pp2pppp/2p5/8/2QP4/2N5/Pq2PPPP/R1B1KBNR w KQkq - 0 6")),
                                  ("case_3_defend", chess.Board("r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4")),
                                  ("case_4_black", chess.Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4"))]:
            chosen = model.choose(test_board)
            val = model.predict_value(test_board)
            model_choices[name] = {"chosen": chosen.uci(), "chosen_san": test_board.san(chosen), "value_pred": round(val, 4)}
        results["model_choices"] = model_choices

    return results

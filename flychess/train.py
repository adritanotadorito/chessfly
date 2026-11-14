"""Stockfish-labelled warm start and a small REINFORCE continuation."""
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .env import ChessEnv
from .match import BulletMatch
from .model import FlyPolicy
from .reward import compute_game_rewards, move_audit, potential


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
        # Best-epoch tracking by held-out cross-entropy loss (validation loss)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            best_metrics = val_metrics

    # Restore best-epoch weights before saving checkpoint
    for k, v in best_params.items():
        model.params[k] = v.copy()
    model.save(output)

    # Optional engine evaluation on the best saved checkpoint
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
                    stockfish_path: str = "/opt/homebrew/bin/stockfish",
                    bot_difficulty: str = "weak",
                    lr: float = 0.0005,
                    gamma: float = 0.99,
                    on_match_done: callable = None):
    """Real online policy-gradient training from complete 1+0 bullet matches."""
    model = FlyPolicy(graph, seed=seed, shuffled=shuffled)
    model.load_weights(checkpoint)

    # Snapshot initial parameters for verification
    initial_params = {k: v.copy() for k, v in model.params.items()}

    baseline = 0.0
    match_logs = []

    # Pre-open analysis engine for post-game reward auditing
    analysis_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    analysis_engine.configure({"Threads": 1, "Hash": 16})

    try:
        for m in range(matches):
            # Alternate colors evenly across matches
            fly_color = chess.WHITE if (m % 2 == 0) else chess.BLACK

            # 1. Run complete 1+0 bullet game with sampled moves
            match = BulletMatch(
                fly_model=model,
                stockfish_path=stockfish_path,
                fly_color=fly_color,
                base_time=60.0,
                bot_difficulty=bot_difficulty,
                sample_moves=True,
            )
            match_res = match.run()

            # 2. Compute transparent chess rewards and diagnostics
            reward_res = compute_game_rewards(
                match_res,
                engine=analysis_engine,
                engine_nodes=1500,
                gamma=gamma,
            )

            transitions = reward_res["transitions"]
            if transitions:
                # 3. Compute discounted returns
                running = 0.0
                returns = []
                for t in reversed(transitions):
                    running = t["training_reward"] + gamma * running
                    returns.append(running)
                returns.reverse()

                # 4. Policy gradient update across all parameter groups
                # Replay moves to get exact board state for backprop
                replay_board = chess.Board()
                step_idx = 0
                for move_record in match_res["moves"]:
                    if move_record["player"] == "fly":
                        trans = transitions[step_idx]
                        ret = returns[step_idx]
                        advantage = float(np.clip(ret - baseline, -2.0, 2.0))

                        move_obj = chess.Move.from_uci(trans["uci"])
                        my_c = move_record.get("white_clock" if fly_color == chess.WHITE else "black_clock")
                        opp_c = move_record.get("black_clock" if fly_color == chess.WHITE else "white_clock")

                        _, grads = model.gradients(
                            replay_board,
                            move_obj,
                            scale=advantage,
                            my_clock=my_c,
                            opp_clock=opp_c,
                        )
                        model.update(grads, lr=lr)
                        step_idx += 1

                    replay_board.push(chess.Move.from_uci(move_record["uci"]))

                baseline = 0.85 * baseline + 0.15 * (returns[0] if returns else 0.0)

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

    # 5. Calculate and verify parameter changes for each group
    param_changes = {}
    for k in model.params:
        diff = model.params[k] - initial_params[k]
        param_changes[k] = {
            "l2_norm": round(float(np.linalg.norm(diff)), 6),
            "max_abs": round(float(np.max(np.abs(diff))), 6),
            "mean_abs": round(float(np.mean(np.abs(diff))), 6),
        }

    # Save new RL checkpoint (preserving warm-start on disk)
    model.save(output)

    return {
        "matches_played": len(match_logs),
        "history": match_logs,
        "parameter_changes": param_changes,
        "checkpoint": output,
        "source_warm_start": checkpoint,
    }


def evaluate_bullet_matches(model: FlyPolicy,
                            stockfish_path: str = "/opt/homebrew/bin/stockfish",
                            num_games: int = 4,
                            bot_difficulty: str = "weak",
                            sample_moves: bool = False) -> dict:
    """Run matched bullet games alternating colors against fixed Stockfish bot."""
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

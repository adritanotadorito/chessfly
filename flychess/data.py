"""Lichess PGN stream reader, dataset sampling, and Stockfish multi-candidate labeling."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import chess
import chess.engine
import chess.pgn
import numpy as np

STOCKFISH_PATH = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
ZSTD_PATH = shutil.which("zstd") or "/opt/homebrew/bin/zstd"

PINNED_ENGINE_CONFIG = {
    "threads": 1,
    "hash_mb": 64,
    "nodes_budget": 15000,
    "multipv": 3,
    "temperature_cp": 50.0,
}


def stream_pgn_zst(archive_path: str) -> Generator[chess.pgn.Game, None, None]:
    """Stream games from a .pgn.zst archive without decompressing to disk."""
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    proc = subprocess.Popen(
        [ZSTD_PATH, "-dc", archive_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        while True:
            game = chess.pgn.read_game(proc.stdout)
            if game is None:
                break
            yield game
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()


def is_usable_game(game: chess.pgn.Game, min_rating: int = 1500, min_plies: int = 20) -> bool:
    """Filter for valid, complete standard games with usable ratings."""
    headers = game.headers
    event = headers.get("Event", "")
    if "Rated" not in event or "Standard" not in event:
        # Standard rated games only
        if "Variant" in headers and headers["Variant"].lower() != "standard":
            return False

    termination = headers.get("Termination", "").lower()
    if "abandoned" in termination or "rules" in termination:
        return False

    result = headers.get("Result", "*")
    if result not in ["1-0", "0-1", "1/2-1/2"]:
        return False

    try:
        w_elo = int(headers.get("WhiteElo", "0"))
        b_elo = int(headers.get("BlackElo", "0"))
        if w_elo < min_rating or b_elo < min_rating:
            return False
    except ValueError:
        return False

    # Check ply count
    moves = list(game.mainline_moves())
    return len(moves) >= min_plies


def sample_game_positions(game: chess.pgn.Game, rng: np.random.Generator) -> List[Tuple[str, int, str]]:
    """Sample 1 to 3 varied positions from a game: middlegames and endgames."""
    moves = list(game.mainline_moves())
    if len(moves) < 20:
        return []

    sampled = []
    board = game.board()

    # Target plies: middlegame (ply 16-40), late middlegame/endgame (ply 41+)
    target_plies = set()
    mid_ply = int(rng.integers(16, min(len(moves) - 2, 40)))
    target_plies.add(mid_ply)

    if len(moves) >= 45 and rng.random() < 0.6:
        end_ply = int(rng.integers(42, len(moves) - 1))
        target_plies.add(end_ply)

    for i, m in enumerate(moves):
        board.push(m)
        ply = i + 1
        if ply in target_plies and not board.is_game_over():
            sampled.append((board.fen(), ply, "endgame" if ply >= 42 else "middlegame"))

    return sampled


def label_position_stockfish(
    engine: chess.engine.SimpleEngine,
    fen: str,
    nodes: int = 15000,
    multipv: int = 3,
    temperature_cp: float = 50.0,
) -> Dict[str, Any]:
    """Analyze position with Stockfish to get value target and multi-candidate soft probabilities."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    color = board.turn
    limit = chess.engine.Limit(nodes=nodes)
    analysis = engine.analyse(board, limit, multipv=min(multipv, len(legal_moves)))

    candidates = []
    best_score_cp = None

    for entry in analysis:
        move = entry.get("pv", [None])[0]
        if move is None:
            continue
        score_obj = entry.get("score")
        if score_obj is None:
            continue
        cp = score_obj.pov(color).score(mate_score=10000)
        if best_score_cp is None:
            best_score_cp = cp
        candidates.append({"uci": move.uci(), "cp": cp})

    if best_score_cp is None:
        best_score_cp = 0.0

    # Value target bounded in (-1.0, 1.0) via tanh(cp / 400.0)
    value_target = float(np.tanh(best_score_cp / 400.0))

    # Compute soft target probability distribution over all legal moves
    # Top moves receive probabilities based on relative loss; other legal moves receive floor loss (e.g. 300 cp)
    candidate_map = {c["uci"]: c["cp"] for c in candidates}
    logits = []
    move_list = []
    for m in legal_moves:
        u = m.uci()
        move_list.append(u)
        if u in candidate_map:
            loss = max(0.0, best_score_cp - candidate_map[u])
        else:
            loss = 350.0  # Assumed unconsidered/inferior move
        logits.append(-loss / temperature_cp)

    logits = np.array(logits, dtype=np.float32)
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)

    best_move_uci = candidates[0]["uci"] if candidates else legal_moves[0].uci()

    return {
        "fen": fen,
        "best_move": best_move_uci,
        "best_cp": best_score_cp,
        "value_target": round(value_target, 4),
        "candidates": candidates,
        "legal_moves": move_list,
        "soft_probs": [round(float(p), 5) for p in probs],
        "nodes": nodes,
    }


def build_pipeline_dataset(
    archives_dir: str,
    output_dir: str,
    target_train: int = 2000,
    target_val: int = 200,
    seed: int = 42,
    nodes: int = 15000,
) -> Dict[str, Any]:
    """Sample and label non-overlapping datasets for train (Jan-Sep), val (Oct), and frozen benchmark (Nov)."""
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": PINNED_ENGINE_CONFIG["threads"], "Hash": PINNED_ENGINE_CONFIG["hash_mb"]})
    version = str(engine.id.get("name", "Stockfish"))

    seen_fens: Set[str] = set()

    # 1. Training Set (Jan-Sep 2013)
    train_archives = [
        Path(archives_dir) / "lichess_db_standard_rated_2013-01.pgn.zst",
        Path(archives_dir) / "lichess_db_standard_rated_2013-02.pgn.zst",
        Path(archives_dir) / "lichess_db_standard_rated_2013-06.pgn.zst",
        Path(archives_dir) / "lichess_db_standard_rated_2013-07.pgn.zst",
    ]
    train_records = []
    print(f"Sampling {target_train} training positions from Jan-Sep 2013 archives...")
    for arc in train_archives:
        if not arc.is_file():
            continue
        for game in stream_pgn_zst(str(arc)):
            if len(train_records) >= target_train:
                break
            if not is_usable_game(game):
                continue
            game_id = game.headers.get("Site", arc.stem)
            samples = sample_game_positions(game, rng)
            for fen, ply, phase in samples:
                if fen in seen_fens:
                    continue
                seen_fens.add(fen)
                labeled = label_position_stockfish(engine, fen, nodes=nodes)
                if labeled:
                    labeled.update({"game_id": game_id, "source": arc.name, "ply": ply, "phase": phase, "split": "train"})
                    train_records.append(labeled)
                    if len(train_records) >= target_train:
                        break
        if len(train_records) >= target_train:
            break

    train_file = out_path / "train_2000.jsonl"
    with open(train_file, "w") as f:
        meta = {
            "metadata": {
                "engine": version,
                "engine_config": PINNED_ENGINE_CONFIG,
                "split": "train",
                "count": len(train_records),
                "seed": seed,
            }
        }
        f.write(json.dumps(meta) + "\n")
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    # 2. Validation Set (October 2013)
    val_archive = Path(archives_dir) / "lichess_db_standard_rated_2013-10.pgn.zst"
    val_records = []
    print(f"Sampling {target_val} validation positions from October 2013 archive...")
    if val_archive.is_file():
        for game in stream_pgn_zst(str(val_archive)):
            if len(val_records) >= target_val:
                break
            if not is_usable_game(game):
                continue
            game_id = game.headers.get("Site", val_archive.stem)
            samples = sample_game_positions(game, rng)
            for fen, ply, phase in samples:
                if fen in seen_fens:
                    continue
                seen_fens.add(fen)
                labeled = label_position_stockfish(engine, fen, nodes=nodes)
                if labeled:
                    labeled.update({"game_id": game_id, "source": val_archive.name, "ply": ply, "phase": phase, "split": "val"})
                    val_records.append(labeled)
                    if len(val_records) >= target_val:
                        break

    val_file = out_path / "val_oct_200.jsonl"
    with open(val_file, "w") as f:
        meta = {
            "metadata": {
                "engine": version,
                "engine_config": PINNED_ENGINE_CONFIG,
                "split": "val_october",
                "count": len(val_records),
                "seed": seed,
            }
        }
        f.write(json.dumps(meta) + "\n")
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    engine.quit()

    return {
        "engine": version,
        "train_file": str(train_file),
        "train_count": len(train_records),
        "val_file": str(val_file),
        "val_count": len(val_records),
    }


def build_frozen_november_benchmark(
    archives_dir: str,
    benchmark_file: str,
    target_representative: int = 100,
    target_tactical: int = 20,
    seed: int = 999,
    nodes: int = 15000,
) -> Dict[str, Any]:
    """Build frozen November 2013 benchmark: 100 representative + 20 tactical challenge positions."""
    nov_archive = Path(archives_dir) / "lichess_db_standard_rated_2013-11.pgn.zst"
    if not nov_archive.is_file():
        raise FileNotFoundError(f"November archive not found: {nov_archive}")

    rng = np.random.default_rng(seed)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": PINNED_ENGINE_CONFIG["threads"], "Hash": PINNED_ENGINE_CONFIG["hash_mb"]})
    version = str(engine.id.get("name", "Stockfish"))

    rep_positions = []
    tactical_positions = []
    seen_fens: Set[str] = set()

    print("Building frozen November benchmark (100 representative + 20 tactical)...")

    for game in stream_pgn_zst(str(nov_archive)):
        if len(rep_positions) >= target_representative and len(tactical_positions) >= target_tactical:
            break
        if not is_usable_game(game):
            continue

        game_id = game.headers.get("Site", "nov-2013")
        board = game.board()
        moves = list(game.mainline_moves())

        for ply_idx, m in enumerate(moves):
            board.push(m)
            ply = ply_idx + 1
            if ply < 12 or board.is_game_over():
                continue

            fen = board.fen()
            if fen in seen_fens:
                continue

            # Check if position is a tactical challenge (e.g. check, hanging queen/rook, or big swing)
            is_tactical = False
            tactical_type = None

            if board.is_check():
                is_tactical = True
                tactical_type = "check"
            else:
                # Check for direct capture threats against major pieces
                for sq, piece in board.piece_map().items():
                    if piece.color == board.turn and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        if board.is_attacked_by(not board.turn, sq) and not board.is_attacked_by(board.turn, sq):
                            is_tactical = True
                            tactical_type = "hanging_piece"
                            break

            if is_tactical and len(tactical_positions) < target_tactical and rng.random() < 0.4:
                labeled = label_position_stockfish(engine, fen, nodes=nodes)
                if labeled:
                    seen_fens.add(fen)
                    labeled.update({
                        "game_id": game_id,
                        "source": nov_archive.name,
                        "ply": ply,
                        "is_tactical": True,
                        "tactical_type": tactical_type,
                    })
                    tactical_positions.append(labeled)
            elif not is_tactical and len(rep_positions) < target_representative and ply >= 16 and rng.random() < 0.2:
                labeled = label_position_stockfish(engine, fen, nodes=nodes)
                if labeled:
                    seen_fens.add(fen)
                    labeled.update({
                        "game_id": game_id,
                        "source": nov_archive.name,
                        "ply": ply,
                        "is_tactical": False,
                        "phase": "endgame" if ply >= 42 else "middlegame",
                    })
                    rep_positions.append(labeled)

    engine.quit()

    total_records = rep_positions + tactical_positions
    out_file = Path(benchmark_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        meta = {
            "metadata": {
                "title": "Frozen November 2013 ChessFly Benchmark",
                "engine": version,
                "engine_config": PINNED_ENGINE_CONFIG,
                "representative_count": len(rep_positions),
                "tactical_count": len(tactical_positions),
                "total_count": len(total_records),
                "source_month": "2013-11",
                "frozen": True,
            }
        }
        f.write(json.dumps(meta) + "\n")
        for r in total_records:
            f.write(json.dumps(r) + "\n")

    return {
        "engine": version,
        "benchmark_file": str(out_file),
        "representative_count": len(rep_positions),
        "tactical_count": len(tactical_positions),
        "total_count": len(total_records),
    }

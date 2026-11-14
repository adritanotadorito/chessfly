"""Game persistence and history store for ChessFly matches."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

GAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "games"


def ensure_games_dir() -> Path:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    return GAMES_DIR


def save_game(game_data: Dict[str, Any]) -> str:
    """Save a completed game record to disk. Returns game_id."""
    ensure_games_dir()
    game_id = game_data.get("id")
    if not game_id:
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rand_slug = uuid.uuid4().hex[:6]
        game_id = f"game-{timestamp_slug}-{rand_slug}"
        game_data["id"] = game_id

    if "timestamp" not in game_data:
        game_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    file_path = GAMES_DIR / f"{game_id}.json"
    file_path.write_text(json.dumps(game_data, indent=2))
    return game_id


def list_games(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent saved games with metadata summary, newest first."""
    ensure_games_dir()
    game_files = sorted(GAMES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries = []
    for p in game_files[:limit]:
        try:
            data = json.loads(p.read_text())
            summaries.append({
                "id": data.get("id", p.stem),
                "timestamp": data.get("timestamp", ""),
                "white": data.get("white_player", "White"),
                "black": data.get("black_player", "Black"),
                "result": data.get("result", "*"),
                "winner": data.get("winner", ""),
                "termination": data.get("termination_reason", ""),
                "mode": data.get("mode", "bullet"),
                "ply_count": data.get("ply_count", len(data.get("moves", []))),
                "file_path": str(p),
            })
        except Exception:
            continue
    return summaries


def load_game(game_id: str) -> Optional[Dict[str, Any]]:
    """Load a full game record by ID."""
    ensure_games_dir()
    file_path = GAMES_DIR / f"{game_id}.json"
    if not file_path.is_file():
        # Try direct path or stem match
        matches = list(GAMES_DIR.glob(f"*{game_id}*.json"))
        if matches:
            file_path = matches[0]
        else:
            return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None

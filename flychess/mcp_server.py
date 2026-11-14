"""Generic local MCP adapter. Verify against AfterQuery's starter kit before judging."""
import os

from mcp.server.fastmcp import FastMCP

from .env import ChessEnv

mcp = FastMCP("flychess")
episode = None


@mcp.tool()
def start() -> dict:
    """Start a chess episode and get legal moves; verifier labels are never returned."""
    global episode
    if episode is not None:
        raise ValueError("Episode already started; reset requires a new server process")
    import chess
    episode = ChessEnv(fen=os.environ.get("FLYCHESS_FEN", chess.STARTING_FEN),
                       opponent=os.environ.get("FLYCHESS_OPPONENT", "random"),
                       stockfish=os.environ.get("FLYCHESS_STOCKFISH"),
                       seed=int(os.environ.get("FLYCHESS_SEED", "42")),
                       max_agent_moves=int(os.environ.get("FLYCHESS_MAX_AGENT_MOVES", "8")))
    return episode.observe()


@mcp.tool()
def observe() -> dict:
    """Get the current board and legal moves."""
    if episode is None:
        raise ValueError("Call start first")
    return episode.observe()


@mcp.tool()
def inspect_line(moves: list[str]) -> dict:
    """Explore up to six legal plies without an engine evaluation; four calls per episode."""
    if episode is None:
        raise ValueError("Call start first")
    return episode.inspect_line(moves)


@mcp.tool()
def play_move(uci: str) -> dict:
    """Commit one legal UCI move and receive the opponent's reply."""
    if episode is None:
        raise ValueError("Call start first")
    return episode.play(uci)


if __name__ == "__main__":
    mcp.run(transport="stdio")

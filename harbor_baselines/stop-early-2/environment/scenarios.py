"""Public scenario definitions for ChessFly: Counterplay.

Contains ONLY public FENs, side to move, number of turns, and scenario descriptions.
Does NOT contain reference solutions, engine evaluations, or secret labels.
"""

SCENARIOS = [
    {
        "id": "scenario_1_poisoned_material",
        "name": "Scenario 1: Queen Raid",
        "side": "white",
        "fen": "rnb1kbnr/pp2pppp/2p5/8/2QP4/2N5/Pq2PPPP/R1B1KBNR w KQkq - 0 6",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "Black's queen has just captured on b2. Choose the best continuation for White; you will play two moves against a strong engine.",
    },
    {
        "id": "scenario_2_acute_threat_defense",
        "name": "Scenario 2: Acute Threat Defense",
        "side": "black",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "White's queen on h5 and bishop on c4 both aim at f7. Choose the best continuation for Black; you will play two moves against a strong engine.",
    },
    {
        "id": "scenario_3_tactical_counterplay",
        "name": "Scenario 3: Knight Fork",
        "side": "black",
        "fen": "r1b1kbnr/pppp1Npp/8/6q1/2BnP3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "White's knight has captured on f7 and forks your queen on g5 and rook on h8. Choose the best continuation for Black; you will play two moves against a strong engine.",
    },
    {
        "id": "scenario_4_sound_vs_unsound",
        "name": "Scenario 4: Attack or Consolidate",
        "side": "white",
        "fen": "r1bq1rk1/pp1n1ppp/2n1p3/2bpP3/3p1B2/2PB1N2/PP2NPPP/R2Q1RK1 w - - 0 10",
        "agent_turns": 2,
        "max_inspections": 3,
        "description": "French-structure middlegame with central tension and attacking chances on the kingside. Choose the best continuation for White; you will play two moves against a strong engine.",
    },
]

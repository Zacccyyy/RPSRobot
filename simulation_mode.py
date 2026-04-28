"""
RPS Simulation Mode

Runs headless (no camera, no UI) games at high speed to generate
training data and compare AI strategies.

Usage:
    python simulation_mode.py

Simulated player strategies:
    - "random"          pure random (baseline)
    - "win_stay"        win-stay / lose-shift (common human bias)
    - "cycler"          Rock -> Paper -> Scissors -> repeat
    - "rock_heavy"      60% Rock, 20% Paper, 20% Scissors
    - "anti_pattern"    tries to counter what the AI played last
    - "mixed_human"     blend of all above (most realistic)

AI opponents:
    - "fair_play"       FairPlayAI (heuristic, beatable)
    - "challenge"       ChallengeAI (heuristic, escalating)
    - "random"          pure random (baseline)
    - "ml"              MLPredictionAI (if trained model exists)

Output:
    ~/Desktop/CapStone/simulation_results.xlsx
    (separate from real gameplay data)
"""

import random
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from fair_play_ai import FairPlayAI, VALID_GESTURES, COUNTER_MOVE, UPGRADE_MOVE, DOWNGRADE_MOVE


# ==================================================================
# CONFIG
# ==================================================================

OUTPUT_PATH = Path.home() / "Desktop" / "CapStone" / "simulation_results.xlsx"

# How many runs and rounds per combination.
RUNS_PER_COMBO = 10
ROUNDS_PER_RUN = 100

# Which player strategies and AI opponents to test.
PLAYER_STRATEGIES = [
    "random",
    "win_stay",
    "cycler",
    "rock_heavy",
    "anti_pattern",
    "mixed_human",
]

AI_OPPONENTS = [
    "random",
    "fair_play",
    "challenge",
]


# ==================================================================
# Simulated player strategies
# ==================================================================

class SimulatedPlayer:
    """
    Generates moves according to a named strategy.
    Tracks its own history so strategies can be stateful.
    """

    def __init__(self, strategy="random"):
        self.strategy = strategy
        self.history = []
        self.cycle_index = 0

    def reset(self):
        self.history = []
        self.cycle_index = 0

    def choose_move(self, last_outcome=None, last_own_move=None, last_opponent_move=None):
        """
        Pick the next move based on the strategy.

        last_outcome:      "win", "lose", "draw", or None (first round)
        last_own_move:     player's previous gesture, or None
        last_opponent_move: AI's previous gesture, or None
        """
        if self.strategy == "random":
            move = random.choice(VALID_GESTURES)

        elif self.strategy == "win_stay":
            move = self._win_stay(last_outcome, last_own_move)

        elif self.strategy == "cycler":
            move = self._cycler()

        elif self.strategy == "rock_heavy":
            move = self._rock_heavy()

        elif self.strategy == "anti_pattern":
            move = self._anti_pattern(last_opponent_move)

        elif self.strategy == "mixed_human":
            move = self._mixed_human(last_outcome, last_own_move, last_opponent_move)

        else:
            move = random.choice(VALID_GESTURES)

        self.history.append(move)
        return move

    def _win_stay(self, last_outcome, last_own_move):
        """
        Win-stay / lose-shift.
        After a win: repeat the same move.
        After a loss: switch to a random different move.
        After a draw: small random shift.
        """
        if last_outcome is None or last_own_move is None:
            return random.choice(VALID_GESTURES)

        if last_outcome == "win":
            # 80% stay, 20% random shift
            if random.random() < 0.80:
                return last_own_move
            return random.choice(VALID_GESTURES)

        if last_outcome == "lose":
            # 70% shift, 30% stay (people don't always shift)
            if random.random() < 0.70:
                options = [g for g in VALID_GESTURES if g != last_own_move]
                return random.choice(options)
            return last_own_move

        # Draw: 50% stay, 50% shift
        if random.random() < 0.50:
            return last_own_move
        return random.choice(VALID_GESTURES)

    def _cycler(self):
        """Rock -> Paper -> Scissors -> repeat."""
        cycle = ["Rock", "Paper", "Scissors"]
        move = cycle[self.cycle_index % 3]
        self.cycle_index += 1
        return move

    def _rock_heavy(self):
        """60% Rock, 20% Paper, 20% Scissors."""
        r = random.random()
        if r < 0.60:
            return "Rock"
        if r < 0.80:
            return "Paper"
        return "Scissors"

    def _anti_pattern(self, last_opponent_move):
        """
        Try to counter what the AI played last.
        If AI played Rock, play Paper.
        Adds noise so it's not perfectly predictable.
        """
        if last_opponent_move is None:
            return random.choice(VALID_GESTURES)

        # 65% counter, 35% random
        if random.random() < 0.65:
            return COUNTER_MOVE[last_opponent_move]
        return random.choice(VALID_GESTURES)

    def _mixed_human(self, last_outcome, last_own_move, last_opponent_move):
        """
        Realistic blend: randomly picks a sub-strategy each round.
        Weighted toward win-stay (most common human pattern).
        """
        r = random.random()

        if r < 0.40:
            return self._win_stay(last_outcome, last_own_move)
        if r < 0.60:
            return self._anti_pattern(last_opponent_move)
        if r < 0.75:
            return self._rock_heavy()
        if r < 0.85:
            # Upgrade after loss (common human tendency)
            if last_outcome == "lose" and last_own_move is not None:
                return UPGRADE_MOVE[last_own_move]
            return random.choice(VALID_GESTURES)
        return random.choice(VALID_GESTURES)


# ==================================================================
# AI opponent factory
# ==================================================================

def create_ai_opponent(ai_type):
    """
    Returns an AI instance and a function to get its move.

    All AIs use the same interface internally, but the wrapper
    normalises them for the simulation loop.
    """
    if ai_type == "random":
        return None, lambda history, streak, rn: random.choice(VALID_GESTURES)

    if ai_type == "fair_play":
        ai = FairPlayAI()
        def get_move(history, streak, round_number):
            return ai.choose_robot_move(history=history, round_number=round_number)
        return ai, get_move

    if ai_type == "challenge":
        from challenge_ai import ChallengeAI
        ai = ChallengeAI()
        def get_move(history, streak, round_number):
            return ai.choose_robot_move(history=history, streak=streak, round_number=round_number)
        return ai, get_move

    if ai_type == "ml":
        try:
            from ml_model import MLPredictionAI
            model_path = Path.home() / "Desktop" / "CapStone" / "rps_ml_model.pkl"
            if not model_path.exists():
                print(f"[Simulation] ML model not found at {model_path}, skipping.")
                return None, None
            ai = MLPredictionAI(model_path=str(model_path))
            def get_move(history, streak, round_number):
                return ai.choose_robot_move(history=history, streak=streak, round_number=round_number)
            return ai, get_move
        except ImportError:
            print("[Simulation] ML model imports failed, skipping.")
            return None, None

    return None, None


# ==================================================================
# RPS comparison
# ==================================================================

BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def compare_rps(player_move, robot_move):
    if player_move == robot_move:
        return "draw"
    if BEATS[player_move] == robot_move:
        return "win"
    return "lose"


# ==================================================================
# Single simulation run
# ==================================================================

def run_single_game(player_strategy, ai_type, num_rounds):
    """
    Simulates one complete run.

    Returns:
        {
            "rounds": [list of round dicts],
            "final_streak": int,
            "player_wins": int,
            "robot_wins": int,
            "draws": int,
        }
    """
    player = SimulatedPlayer(strategy=player_strategy)
    ai_instance, ai_get_move = create_ai_opponent(ai_type)

    if ai_get_move is None:
        return None

    if ai_instance is not None and hasattr(ai_instance, "reset"):
        ai_instance.reset()

    history = []
    streak = 0
    player_wins = 0
    robot_wins = 0
    draws = 0

    last_outcome = None
    last_player_move = None
    last_robot_move = None

    for round_num in range(1, num_rounds + 1):
        # Player chooses.
        player_move = player.choose_move(
            last_outcome=last_outcome,
            last_own_move=last_player_move,
            last_opponent_move=last_robot_move,
        )

        # AI chooses.
        robot_move = ai_get_move(history, streak, round_num)

        # Resolve.
        outcome = compare_rps(player_move, robot_move)

        if outcome == "win":
            player_wins += 1
            streak += 1
            player_outcome = "win"
        elif outcome == "lose":
            robot_wins += 1
            streak = 0
            player_outcome = "lose"
        else:
            draws += 1
            player_outcome = "draw"

        # Derive response type.
        response_type = None
        if last_player_move is not None:
            if player_move == last_player_move:
                response_type = "stay"
            elif UPGRADE_MOVE.get(last_player_move) == player_move:
                response_type = "upgrade"
            elif DOWNGRADE_MOVE.get(last_player_move) == player_move:
                response_type = "downgrade"

        round_record = {
            "round_number": round_num,
            "player_gesture": player_move,
            "robot_gesture": robot_move,
            "player_outcome": player_outcome,
            "previous_player_gesture": last_player_move,
            "player_response_type": response_type,
        }

        history.append(round_record)

        last_outcome = player_outcome
        last_player_move = player_move
        last_robot_move = robot_move

    # Calculate max streak.
    max_streak = 0
    current = 0
    for r in history:
        if r["player_outcome"] == "win":
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return {
        "rounds": history,
        "final_streak": max_streak,
        "player_wins": player_wins,
        "robot_wins": robot_wins,
        "draws": draws,
    }


# ==================================================================
# Excel output
# ==================================================================

def save_results_to_excel(all_results, output_path):
    """
    Saves simulation results to an Excel workbook with two sheets:
        - Sim_Summary   (one row per run)
        - Sim_Rounds    (every individual round)
    """
    wb = Workbook()

    # --- Summary sheet ---
    summary_ws = wb.active
    summary_ws.title = "Sim_Summary"
    summary_ws.append([
        "run_id",
        "player_strategy",
        "ai_opponent",
        "rounds_played",
        "player_wins",
        "robot_wins",
        "draws",
        "player_win_rate",
        "robot_win_rate",
        "max_streak",
        "timestamp",
    ])

    # --- Rounds sheet ---
    rounds_ws = wb.create_sheet("Sim_Rounds")
    rounds_ws.append([
        "run_id",
        "player_strategy",
        "ai_opponent",
        "round_number",
        "player_gesture",
        "robot_gesture",
        "round_result",
        "player_outcome",
        "previous_player_gesture",
        "player_response_type",
    ])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_counter = 0

    for combo_key, runs in all_results.items():
        strategy, ai_type = combo_key

        for run in runs:
            run_counter += 1
            run_id = f"SIM-{run_counter:04d}"

            total_decided = run["player_wins"] + run["robot_wins"]
            player_wr = run["player_wins"] / total_decided if total_decided > 0 else 0.0
            robot_wr = run["robot_wins"] / total_decided if total_decided > 0 else 0.0

            summary_ws.append([
                run_id,
                strategy,
                ai_type,
                len(run["rounds"]),
                run["player_wins"],
                run["robot_wins"],
                run["draws"],
                round(player_wr, 4),
                round(robot_wr, 4),
                run["final_streak"],
                timestamp,
            ])

            for r in run["rounds"]:
                result_label = {
                    "win": "player_win",
                    "lose": "robot_win",
                    "draw": "draw",
                }.get(r["player_outcome"], r["player_outcome"])

                rounds_ws.append([
                    run_id,
                    strategy,
                    ai_type,
                    r["round_number"],
                    r["player_gesture"],
                    r["robot_gesture"],
                    result_label,
                    r["player_outcome"],
                    r["previous_player_gesture"],
                    r["player_response_type"],
                ])

    # --- Formatting ---
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for ws in [summary_ws, rounds_ws]:
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"

    col_widths_summary = {
        "A": 14, "B": 18, "C": 14, "D": 14, "E": 14,
        "F": 14, "G": 10, "H": 16, "I": 16, "J": 14, "K": 22,
    }
    col_widths_rounds = {
        "A": 14, "B": 18, "C": 14, "D": 14, "E": 16,
        "F": 16, "G": 14, "H": 16, "I": 22, "J": 20,
    }

    for col, width in col_widths_summary.items():
        summary_ws.column_dimensions[col].width = width
    for col, width in col_widths_rounds.items():
        rounds_ws.column_dimensions[col].width = width

    # Save.
    try:
        wb.save(output_path)
        print(f"\nResults saved to: {output_path}")
    except PermissionError:
        print(f"\nCould not save — please close {output_path} in Excel and try again.")
    finally:
        wb.close()


# ==================================================================
# Reusable simulation runner (called from main.py or CLI)
# ==================================================================

def run_simulation(
    player_strategies=None,
    ai_opponents=None,
    runs_per_combo=RUNS_PER_COMBO,
    rounds_per_run=ROUNDS_PER_RUN,
    save_excel=True,
    output_path=None,
):
    """
    Run the full simulation and return structured results.

    Returns:
        {
            "elapsed_seconds": float,
            "total_rounds": int,
            "total_runs": int,
            "combo_results": [
                {
                    "strategy": str,
                    "ai": str,
                    "player_win_rate": float,
                    "robot_win_rate": float,
                    "draw_rate": float,
                    "avg_streak": float,
                    "runs": int,
                },
                ...
            ],
            "best_ai": str,           # hardest for players to beat
            "worst_ai": str,          # easiest for players to beat
            "best_strategy": str,     # most effective player strategy overall
            "worst_strategy": str,    # least effective player strategy overall
        }
    """
    if player_strategies is None:
        player_strategies = PLAYER_STRATEGIES
    if ai_opponents is None:
        ai_opponents = AI_OPPONENTS
    if output_path is None:
        output_path = OUTPUT_PATH

    all_results = {}
    combo_summaries = []
    start_time = time.time()

    for strategy in player_strategies:
        for ai_type in ai_opponents:
            combo_key = (strategy, ai_type)
            runs = []

            for _ in range(runs_per_combo):
                result = run_single_game(strategy, ai_type, rounds_per_run)
                if result is not None:
                    runs.append(result)

            all_results[combo_key] = runs

            if runs:
                total_pw = sum(r["player_wins"] for r in runs)
                total_rw = sum(r["robot_wins"] for r in runs)
                total_d = sum(r["draws"] for r in runs)
                total_all = total_pw + total_rw + total_d
                avg_streak = sum(r["final_streak"] for r in runs) / len(runs)

                combo_summaries.append({
                    "strategy": strategy,
                    "ai": ai_type,
                    "player_win_rate": total_pw / total_all if total_all > 0 else 0.0,
                    "robot_win_rate": total_rw / total_all if total_all > 0 else 0.0,
                    "draw_rate": total_d / total_all if total_all > 0 else 0.0,
                    "avg_streak": round(avg_streak, 1),
                    "runs": len(runs),
                })

    elapsed = time.time() - start_time

    total_runs = sum(len(r) for r in all_results.values())
    total_rounds = sum(
        len(r["rounds"])
        for runs in all_results.values()
        for r in runs
    )

    # --- Aggregate: best/worst AI and strategy ---
    ai_avg_wr = {}
    for s in combo_summaries:
        ai_avg_wr.setdefault(s["ai"], []).append(s["robot_win_rate"])

    strategy_avg_wr = {}
    for s in combo_summaries:
        strategy_avg_wr.setdefault(s["strategy"], []).append(s["player_win_rate"])

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    best_ai = max(ai_avg_wr, key=lambda k: _avg(ai_avg_wr[k])) if ai_avg_wr else "N/A"
    worst_ai = min(ai_avg_wr, key=lambda k: _avg(ai_avg_wr[k])) if ai_avg_wr else "N/A"
    best_strategy = max(strategy_avg_wr, key=lambda k: _avg(strategy_avg_wr[k])) if strategy_avg_wr else "N/A"
    worst_strategy = min(strategy_avg_wr, key=lambda k: _avg(strategy_avg_wr[k])) if strategy_avg_wr else "N/A"

    # --- Save to Excel ---
    if save_excel:
        save_results_to_excel(all_results, output_path)

    return {
        "elapsed_seconds": round(elapsed, 1),
        "total_rounds": total_rounds,
        "total_runs": total_runs,
        "combo_results": combo_summaries,
        "best_ai": best_ai,
        "worst_ai": worst_ai,
        "best_strategy": best_strategy,
        "worst_strategy": worst_strategy,
    }


# ==================================================================
# Main (CLI entry point)
# ==================================================================

def main():
    print("=" * 60)
    print("RPS Simulation Mode")
    print("=" * 60)
    print()
    print(f"Runs per combination:   {RUNS_PER_COMBO}")
    print(f"Rounds per run:         {ROUNDS_PER_RUN}")
    print(f"Player strategies:      {len(PLAYER_STRATEGIES)}")
    print(f"AI opponents:           {len(AI_OPPONENTS)}")
    total_combos = len(PLAYER_STRATEGIES) * len(AI_OPPONENTS)
    total_runs = total_combos * RUNS_PER_COMBO
    total_rounds = total_runs * ROUNDS_PER_RUN
    print(f"Total combinations:     {total_combos}")
    print(f"Total runs:             {total_runs}")
    print(f"Total rounds:           {total_rounds:,}")
    print()

    results = run_simulation()

    for s in results["combo_results"]:
        print(
            f"  {s['strategy']:15s} vs {s['ai']:12s}  |  "
            f"Player WR: {s['player_win_rate']:.1%}  "
            f"Avg streak: {s['avg_streak']:.1f}  "
            f"({s['runs']} runs)"
        )

    print()
    print(f"Simulation complete in {results['elapsed_seconds']}s")
    print(f"Generated {results['total_rounds']:,} rounds across {results['total_runs']} runs.")
    print()
    print(f"Strongest AI:          {results['best_ai']}")
    print(f"Weakest AI:            {results['worst_ai']}")
    print(f"Best player strategy:  {results['best_strategy']}")
    print(f"Worst player strategy: {results['worst_strategy']}")
    print()
    print("You can now re-run ml_training_script.py with simulation data.")


if __name__ == "__main__":
    main()

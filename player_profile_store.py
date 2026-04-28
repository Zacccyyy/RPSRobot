"""
Player Profile Store

Manages named player profiles for the clone system.
Each player gets a JSON file with their full round history
and computed pattern statistics.

Storage:
    ~/Desktop/CapStone/player_profiles/<name>.json   -  per-player data
    ~/Desktop/CapStone/player_research_log.xlsx      -  combined research Excel

Profile data includes:
    - All rounds played (gesture, opponent gesture, outcome)
    - Conditional response tables (after win/loss/draw × last move)
    - Transition frequencies (move A → move B)
    - Overall gesture frequencies
"""

import json
import os
from datetime import datetime
from pathlib import Path


GESTURES = ["Rock", "Paper", "Scissors"]
OUTCOMES = ["win", "lose", "draw"]

UPGRADE = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
DOWNGRADE = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}

MIN_ROUNDS_FOR_CLONE = 30


class PlayerProfileStore:

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.home() / "Desktop" / "CapStone"
        self.profiles_dir = self.base_dir / "player_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.excel_path = self.base_dir / "player_research_log.xlsx"

    def _profile_path(self, name):
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return self.profiles_dir / f"{safe_name.lower()}.json"

    def list_players(self):
        """Return list of (display_name, round_count) for all saved profiles."""
        players = []
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                name = data.get("player_name", path.stem)
                count = len(data.get("rounds", []))
                players.append((name, count))
            except Exception:
                continue
        return players

    def list_playable_clones(self):
        """Return names of players with enough data to clone."""
        return [
            name for name, count in self.list_players()
            if count >= MIN_ROUNDS_FOR_CLONE
        ]

    def load_profile(self, name):
        """Load a player profile. Returns dict or None."""
        path = self._profile_path(name)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def get_or_create_profile(self, name):
        """Load existing profile or create a new one."""
        profile = self.load_profile(name)
        if profile is not None:
            return profile

        profile = {
            "player_name": name,
            "created_at": datetime.now().isoformat(),
            "last_played": datetime.now().isoformat(),
            "rounds": [],
        }
        self._save_profile(name, profile)
        return profile

    def record_round(self, player_name, player_gesture, robot_gesture,
                     outcome, game_mode, round_number=0, emotion=None):
        """
        Record a single round to the player's profile.
        Called automatically during gameplay when a player name is set.

        emotion: optional dict from EmotionTracker.get_round_snapshot()
        """
        if not player_name or not player_name.strip():
            return

        profile = self.get_or_create_profile(player_name)

        round_data = {
            "timestamp": datetime.now().isoformat(),
            "player_gesture": player_gesture,
            "robot_gesture": robot_gesture,
            "outcome": outcome,
            "game_mode": game_mode,
            "round_number": round_number,
        }

        # Attach emotion data if available.
        if emotion and isinstance(emotion, dict):
            round_data["emotion"] = emotion.get("emotion", "Unknown")
            round_data["emotion_confidence"] = emotion.get("emotion_confidence", 0.0)
            round_data["smile_score"] = emotion.get("smile_score", 0.0)
            round_data["surprise_score"] = emotion.get("surprise_score", 0.0)
            round_data["frustration_score"] = emotion.get("frustration_score", 0.0)

        # Compute response type from previous round.
        if profile["rounds"]:
            prev = profile["rounds"][-1]
            prev_gesture = prev["player_gesture"]
            if player_gesture == prev_gesture:
                round_data["response_type"] = "stay"
            elif UPGRADE.get(prev_gesture) == player_gesture:
                round_data["response_type"] = "upgrade"
            else:
                round_data["response_type"] = "downgrade"
            round_data["previous_gesture"] = prev_gesture
            round_data["previous_outcome"] = prev["outcome"]
        else:
            round_data["response_type"] = "first"
            round_data["previous_gesture"] = None
            round_data["previous_outcome"] = None

        profile["rounds"].append(round_data)
        profile["last_played"] = datetime.now().isoformat()

        self._save_profile(player_name, profile)
        self._log_to_excel(player_name, round_data)

    def build_pattern_tables(self, name):
        """
        Build the statistical tables that define a player's style.
        These are used by the clone AI to reproduce their patterns.

        Returns dict with:
            - gesture_freq: {Rock: 0.4, Paper: 0.35, Scissors: 0.25}
            - outcome_response: {win: {stay: 0.6, upgrade: 0.2, downgrade: 0.2}, ...}
            - transition: {Rock: {Rock: 0.3, Paper: 0.5, Scissors: 0.2}, ...}
            - outcome_transition: {win: {Rock: {Rock: ..., ...}, ...}, ...}
            - round_count: int
        """
        profile = self.load_profile(name)
        if profile is None or not profile["rounds"]:
            return None

        rounds = profile["rounds"]

        # Overall frequency.
        gesture_counts = {g: 0 for g in GESTURES}
        for r in rounds:
            g = r["player_gesture"]
            if g in gesture_counts:
                gesture_counts[g] += 1

        total = max(sum(gesture_counts.values()), 1)
        gesture_freq = {g: c / total for g, c in gesture_counts.items()}

        # Outcome-conditioned response type.
        outcome_response = {
            o: {"stay": 0, "upgrade": 0, "downgrade": 0}
            for o in OUTCOMES
        }
        for r in rounds:
            rt = r.get("response_type")
            po = r.get("previous_outcome")
            if rt in ("stay", "upgrade", "downgrade") and po in OUTCOMES:
                outcome_response[po][rt] += 1

        # Normalise.
        for o in OUTCOMES:
            total_r = max(sum(outcome_response[o].values()), 1)
            outcome_response[o] = {
                k: v / total_r for k, v in outcome_response[o].items()
            }

        # Direct transition: after gesture X, probability of gesture Y.
        transition = {
            g: {g2: 0 for g2 in GESTURES} for g in GESTURES
        }
        for i in range(len(rounds) - 1):
            curr_g = rounds[i]["player_gesture"]
            next_g = rounds[i + 1]["player_gesture"]
            if curr_g in GESTURES and next_g in GESTURES:
                transition[curr_g][next_g] += 1

        for g in GESTURES:
            total_t = max(sum(transition[g].values()), 1)
            transition[g] = {g2: c / total_t for g2, c in transition[g].items()}

        # Outcome + gesture → next gesture transition.
        outcome_transition = {
            o: {g: {g2: 0 for g2 in GESTURES} for g in GESTURES}
            for o in OUTCOMES
        }
        for i in range(len(rounds) - 1):
            curr_g = rounds[i]["player_gesture"]
            curr_o = rounds[i]["outcome"]
            next_g = rounds[i + 1]["player_gesture"]
            if curr_g in GESTURES and curr_o in OUTCOMES and next_g in GESTURES:
                outcome_transition[curr_o][curr_g][next_g] += 1

        for o in OUTCOMES:
            for g in GESTURES:
                total_ot = max(sum(outcome_transition[o][g].values()), 1)
                outcome_transition[o][g] = {
                    g2: c / total_ot
                    for g2, c in outcome_transition[o][g].items()
                }

        return {
            "player_name": name,
            "round_count": len(rounds),
            "gesture_freq": gesture_freq,
            "outcome_response": outcome_response,
            "transition": transition,
            "outcome_transition": outcome_transition,
        }

    def _save_profile(self, name, profile):
        path = self._profile_path(name)
        try:
            with open(path, "w") as f:
                json.dump(profile, f, indent=2)
        except Exception as exc:
            print(f"[ProfileStore] Save error: {exc}")

    def save_ai_state(self, player_name, ai):
        """
        Persist the AI's learned bandit weights alongside the player profile.
        This allows the AI to resume learning from where it left off next session.
        Called when the game ends or the player quits.
        """
        if not player_name or not player_name.strip():
            return
        try:
            profile = self.get_or_create_profile(player_name)
            profile["ai_state"] = {
                "bandit":              getattr(ai, "_bandit", {}),
                "consecutive_wins":    getattr(ai, "_consecutive_wins", 0),
                "consecutive_losses":  getattr(ai, "_consecutive_losses", 0),
                "saved_at":            datetime.now().isoformat(),
            }
            self._save_profile(player_name, profile)
        except Exception as exc:
            print(f"[ProfileStore] AI state save error: {exc}")

    def load_ai_state(self, player_name, ai):
        """
        Restore persisted bandit weights into an AI instance.
        Called when a named player starts a session.
        Returns True if state was loaded, False otherwise.
        """
        if not player_name or not player_name.strip():
            return False
        try:
            profile = self.load_profile(player_name)
            if not profile:
                return False
            ai_state = profile.get("ai_state")
            if not ai_state:
                return False
            if hasattr(ai, "_bandit") and ai_state.get("bandit"):
                # Merge saved bandit — don't fully replace, blend with fresh prior
                for layer, saved in ai_state["bandit"].items():
                    if layer in ai._bandit and isinstance(saved, list) and len(saved) == 2:
                        # Blend: give 70% weight to saved history, 30% to fresh prior
                        ai._bandit[layer][0] = max(1.0, saved[0] * 0.7)
                        ai._bandit[layer][1] = max(1.0, saved[1] * 0.7)
            print(f"[ProfileStore] Loaded AI state for {player_name}")
            return True
        except Exception as exc:
            print(f"[ProfileStore] AI state load error: {exc}")
            return False

    def _log_to_excel(self, player_name, round_data):
        """Append round to the combined research Excel."""
        try:
            from openpyxl import Workbook, load_workbook
            from openpyxl.styles import Font, PatternFill, Alignment

            if self.excel_path.exists():
                try:
                    wb = load_workbook(self.excel_path)
                except Exception:
                    # File is corrupted — back it up and start fresh
                    backup = self.excel_path.with_suffix(".corrupted.xlsx")
                    try:
                        self.excel_path.rename(backup)
                        print(f"[ProfileStore] Corrupted Excel backed up to {backup.name}, creating fresh file.")
                    except Exception:
                        self.excel_path.unlink(missing_ok=True)
                    wb = None
            else:
                wb = None

            if wb is None:
                wb = Workbook()
                ws = wb.active
                ws.title = "All_Rounds"
                headers = [
                    "timestamp", "player_name", "player_gesture",
                    "robot_gesture", "outcome", "game_mode",
                    "round_number", "response_type",
                    "previous_gesture", "previous_outcome",
                    "emotion", "emotion_confidence",
                    "smile_score", "surprise_score", "frustration_score",
                ]
                ws.append(headers)
                header_fill = PatternFill("solid", fgColor="1F4E78")
                header_font = Font(color="FFFFFF", bold=True)
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
                ws.freeze_panes = "A2"

            ws = wb["All_Rounds"]

            # Auto-migrate: add emotion headers if missing from older workbook.
            current_headers = [c.value for c in ws[1]]
            if "emotion" not in current_headers:
                col = len(current_headers) + 1
                for i, h in enumerate(["emotion", "emotion_confidence",
                                       "smile_score", "surprise_score",
                                       "frustration_score"]):
                    cell = ws.cell(row=1, column=col + i, value=h)
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.font = Font(color="FFFFFF", bold=True)
                    cell.alignment = Alignment(horizontal="center")

            ws.append([
                round_data.get("timestamp", ""),
                player_name,
                round_data.get("player_gesture", ""),
                round_data.get("robot_gesture", ""),
                round_data.get("outcome", ""),
                round_data.get("game_mode", ""),
                round_data.get("round_number", 0),
                round_data.get("response_type", ""),
                round_data.get("previous_gesture", ""),
                round_data.get("previous_outcome", ""),
                round_data.get("emotion", ""),
                round_data.get("emotion_confidence", ""),
                round_data.get("smile_score", ""),
                round_data.get("surprise_score", ""),
                round_data.get("frustration_score", ""),
            ])

            wb.save(self.excel_path)
            wb.close()

        except Exception as exc:
            print(f"[ProfileStore] Excel log error: {exc}")

    def generate_all_player_reports(self):
        """
        Generate per-player analysis sheets in the research Excel.
        Each player gets their own tab with strategy breakdown.
        """
        try:
            from openpyxl import Workbook, load_workbook
            from openpyxl.styles import Font, PatternFill, Alignment

            if self.excel_path.exists():
                wb = load_workbook(self.excel_path)
            else:
                wb = Workbook()
                ws = wb.active
                ws.title = "All_Rounds"

            players = self.list_players()
            updated = 0

            for name, count in players:
                if count < 5:
                    continue

                tables = self.build_pattern_tables(name)
                if tables is None:
                    continue

                sheet_name = name[:28]
                if sheet_name in wb.sheetnames:
                    del wb[sheet_name]

                ws = wb.create_sheet(sheet_name)
                self._write_player_sheet(ws, name, tables)
                updated += 1

            wb.save(self.excel_path)
            wb.close()

            print(f"[ProfileStore] Updated {updated} player report sheets")
            return updated

        except Exception as exc:
            print(f"[ProfileStore] Report generation error: {exc}")
            return 0

    def _write_player_sheet(self, ws, name, tables):
        """Write a single player's analysis to a worksheet."""
        from openpyxl.styles import Font, PatternFill, Alignment

        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True, size=12)
        section_font = Font(bold=True, size=11)
        value_font = Font(size=10)

        row = 1

        # --- Title ---
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = ws.cell(row=row, column=1, value=f"Player Profile: {name}")
        cell.font = Font(bold=True, size=14, color="1F4E78")
        row += 1

        ws.cell(row=row, column=1, value=f"Total rounds: {tables['round_count']}")
        row += 2

        # --- Overall Strategy ---
        ws.cell(row=row, column=1, value="OVERALL STRATEGY").font = section_font
        row += 1

        freq = tables["gesture_freq"]
        favourite = max(freq, key=freq.get)
        least = min(freq, key=freq.get)

        ws.cell(row=row, column=1, value="Gesture")
        ws.cell(row=row, column=2, value="Frequency")
        ws.cell(row=row, column=3, value="Role")
        for c in range(1, 4):
            ws.cell(row=row, column=c).font = header_font
            ws.cell(row=row, column=c).fill = header_fill
            ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")
        row += 1

        counter = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
        for g in GESTURES:
            ws.cell(row=row, column=1, value=g)
            ws.cell(row=row, column=2, value=f"{freq[g]:.0%}")
            role = ""
            if g == favourite:
                role = "FAVOURITE"
            elif g == least:
                role = "Least used"
            ws.cell(row=row, column=3, value=role)
            row += 1

        row += 1

        # --- How to Beat Them ---
        ws.cell(row=row, column=1, value="HOW TO BEAT THIS PLAYER").font = section_font
        row += 1

        best_counter = counter[favourite]
        ws.cell(row=row, column=1, value=f"Primary strategy: Play {best_counter} often")
        ws.cell(row=row, column=2, value=f"Counters their {favourite} ({freq[favourite]:.0%} of throws)")
        row += 1

        # After-loss tendency.
        loss_resp = tables["outcome_response"].get("lose", {})
        loss_max = max(loss_resp, key=loss_resp.get) if loss_resp else "stay"
        ws.cell(row=row, column=1, value=f"After they lose: they tend to {loss_max}")
        ws.cell(row=row, column=2, value=f"({loss_resp.get(loss_max, 0):.0%} of the time)")
        row += 1

        win_resp = tables["outcome_response"].get("win", {})
        win_max = max(win_resp, key=win_resp.get) if win_resp else "stay"
        ws.cell(row=row, column=1, value=f"After they win: they tend to {win_max}")
        ws.cell(row=row, column=2, value=f"({win_resp.get(win_max, 0):.0%} of the time)")
        row += 2

        # --- Unique Traits ---
        ws.cell(row=row, column=1, value="UNIQUE TRAITS").font = section_font
        row += 1

        traits = self._compute_traits(tables)
        for trait in traits:
            ws.cell(row=row, column=1, value=trait)
            row += 1

        row += 1

        # --- After Outcome Response ---
        ws.cell(row=row, column=1, value="RESPONSE PATTERNS (after outcome)").font = section_font
        row += 1

        ws.cell(row=row, column=1, value="After...")
        ws.cell(row=row, column=2, value="Stay")
        ws.cell(row=row, column=3, value="Upgrade")
        ws.cell(row=row, column=4, value="Downgrade")
        for c in range(1, 5):
            ws.cell(row=row, column=c).font = header_font
            ws.cell(row=row, column=c).fill = header_fill
            ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")
        row += 1

        for outcome in OUTCOMES:
            resp = tables["outcome_response"].get(outcome, {})
            ws.cell(row=row, column=1, value=outcome.title())
            ws.cell(row=row, column=2, value=f"{resp.get('stay', 0):.0%}")
            ws.cell(row=row, column=3, value=f"{resp.get('upgrade', 0):.0%}")
            ws.cell(row=row, column=4, value=f"{resp.get('downgrade', 0):.0%}")
            row += 1

        row += 1

        # --- Transition Matrix ---
        ws.cell(row=row, column=1, value="MOVE TRANSITIONS (after X, plays Y)").font = section_font
        row += 1

        ws.cell(row=row, column=1, value="After...")
        for j, g in enumerate(GESTURES):
            ws.cell(row=row, column=j + 2, value=f"→ {g}")
        for c in range(1, 5):
            ws.cell(row=row, column=c).font = header_font
            ws.cell(row=row, column=c).fill = header_fill
            ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")
        row += 1

        trans = tables["transition"]
        for g in GESTURES:
            ws.cell(row=row, column=1, value=g)
            for j, g2 in enumerate(GESTURES):
                ws.cell(row=row, column=j + 2, value=f"{trans[g][g2]:.0%}")
            row += 1

        # Column widths.
        ws.column_dimensions["A"].width = 38
        ws.column_dimensions["B"].width = 22
        ws.column_dimensions["C"].width = 18
        ws.column_dimensions["D"].width = 18

    def _compute_traits(self, tables):
        """Compute human-readable unique traits for this player."""
        traits = []

        freq = tables["gesture_freq"]
        favourite = max(freq, key=freq.get)

        # Dominant gesture.
        if freq[favourite] > 0.50:
            traits.append(f"Heavy {favourite} player ({freq[favourite]:.0%} of throws)")
        elif freq[favourite] > 0.40:
            traits.append(f"Leans toward {favourite} ({freq[favourite]:.0%} of throws)")
        else:
            traits.append("Relatively balanced  -  no strong favourite")

        # Win-stay tendency.
        win_stay = tables["outcome_response"].get("win", {}).get("stay", 0)
        if win_stay > 0.55:
            traits.append(f"Win-stay player  -  repeats winning move {win_stay:.0%} of the time")
        elif win_stay < 0.25:
            traits.append(f"Win-shift player  -  rarely repeats after winning ({win_stay:.0%})")

        # Lose-shift tendency.
        lose_stay = tables["outcome_response"].get("lose", {}).get("stay", 0)
        if lose_stay > 0.45:
            traits.append(f"Stubborn after losses  -  stays with losing move {lose_stay:.0%}")
        elif lose_stay < 0.20:
            traits.append(f"Quick to change after losing  -  only stays {lose_stay:.0%}")

        # Upgrade after loss.
        lose_up = tables["outcome_response"].get("lose", {}).get("upgrade", 0)
        if lose_up > 0.50:
            traits.append(f"Upgrader  -  after losing, upgrades {lose_up:.0%} of the time")

        # Strongest transition.
        trans = tables["transition"]
        best_from, best_to, best_pct = None, None, 0
        for g in GESTURES:
            for g2 in GESTURES:
                if trans[g][g2] > best_pct:
                    best_pct = trans[g][g2]
                    best_from = g
                    best_to = g2

        if best_pct > 0.55:
            traits.append(
                f"Predictable sequence: after {best_from}, "
                f"plays {best_to} {best_pct:.0%} of the time"
            )

        # Draw response.
        draw_resp = tables["outcome_response"].get("draw", {})
        draw_max = max(draw_resp, key=draw_resp.get) if draw_resp else "stay"
        if draw_resp.get(draw_max, 0) > 0.50:
            traits.append(f"After draws, tends to {draw_max} ({draw_resp[draw_max]:.0%})")

        if not traits:
            traits.append("No strong identifiable patterns yet")

        return traits

    def export_csv(self, player_name, output_dir=None):
        """
        Export a player's round history to CSV.
        Returns the path written, or None on failure.
        """
        import csv
        from pathlib import Path

        profile = self.load_profile(player_name)
        if not profile:
            return None

        rounds = profile.get("rounds", [])
        if not rounds:
            return None

        if output_dir is None:
            output_dir = Path.home() / "Desktop" / "CapStone"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in player_name)
        path = output_dir / f"{safe_name}_export.csv"

        fieldnames = [
            "timestamp", "player_gesture", "robot_gesture",
            "outcome", "game_mode", "round_number",
            "response_type", "previous_gesture", "previous_outcome",
            "emotion", "emotion_confidence", "smile_score",
            "surprise_score", "frustration_score",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in rounds:
                    writer.writerow(r)
            return str(path)
        except Exception as exc:
            print(f"[ProfileStore] CSV export error: {exc}")
            return None

    def build_pattern_tables_filtered(self, name, mode_filter=None):
        """
        Same as build_pattern_tables but restricted to rounds of a specific game mode.
        mode_filter: None (all), "FairPlay", "Challenge", "Cheat", "Clone"
        Returns None if not enough data.
        """
        profile = self.load_profile(name)
        if profile is None or not profile["rounds"]:
            return None

        all_rounds = profile["rounds"]
        if mode_filter and mode_filter != "All":
            _norm_map = {"Fair Play": "FairPlay", "fair play": "FairPlay",
                         "Bluff Mode": "Cheat"}
            rounds = [r for r in all_rounds
                      if _norm_map.get(r.get("game_mode", ""), r.get("game_mode", "")) == mode_filter]
        else:
            rounds = all_rounds

        if not rounds:
            return None

        gesture_counts = {g: 0 for g in GESTURES}
        for r in rounds:
            g = r["player_gesture"]
            if g in gesture_counts:
                gesture_counts[g] += 1
        total = max(sum(gesture_counts.values()), 1)
        gesture_freq = {g: c / total for g, c in gesture_counts.items()}

        outcome_response = {o: {"stay": 0, "upgrade": 0, "downgrade": 0} for o in OUTCOMES}
        for r in rounds:
            rt = r.get("response_type")
            po = r.get("previous_outcome")
            if rt in ("stay", "upgrade", "downgrade") and po in OUTCOMES:
                outcome_response[po][rt] += 1
        for o in OUTCOMES:
            total_r = max(sum(outcome_response[o].values()), 1)
            outcome_response[o] = {k: v / total_r for k, v in outcome_response[o].items()}

        transition = {g: {g2: 0 for g2 in GESTURES} for g in GESTURES}
        for i in range(len(rounds) - 1):
            curr_g = rounds[i]["player_gesture"]
            next_g = rounds[i + 1]["player_gesture"]
            if curr_g in GESTURES and next_g in GESTURES:
                transition[curr_g][next_g] += 1
        for g in GESTURES:
            total_t = max(sum(transition[g].values()), 1)
            transition[g] = {g2: c / total_t for g2, c in transition[g].items()}

        outcome_transition = {
            o: {g: {g2: 0 for g2 in GESTURES} for g in GESTURES} for o in OUTCOMES
        }
        for i in range(len(rounds) - 1):
            curr_g = rounds[i]["player_gesture"]
            curr_o = rounds[i]["outcome"]
            next_g = rounds[i + 1]["player_gesture"]
            if curr_g in GESTURES and curr_o in OUTCOMES and next_g in GESTURES:
                outcome_transition[curr_o][curr_g][next_g] += 1
        for o in OUTCOMES:
            for g in GESTURES:
                total_ot = max(sum(outcome_transition[o][g].values()), 1)
                outcome_transition[o][g] = {g2: c / total_ot for g2, c in outcome_transition[o][g].items()}

        return {
            "player_name": name,
            "round_count": len(rounds),
            "mode_filter": mode_filter or "All",
            "gesture_freq": gesture_freq,
            "outcome_response": outcome_response,
            "transition": transition,
            "outcome_transition": outcome_transition,
        }

    def get_session_history(self, name, max_sessions=5):
        """
        Group rounds into sessions (match groups) and return the last max_sessions.
        A new session is detected when round_number resets (goes < previous round_number)
        or when the timestamp gap exceeds 10 minutes.

        Returns list of session dicts:
          {date, mode, rounds_played, wins, losses, draws, win_rate, avg_reaction_ms}
        """
        from datetime import datetime as _dt
        profile = self.load_profile(name)
        if profile is None or not profile["rounds"]:
            return []

        rounds = profile["rounds"]
        sessions = []
        current = []

        for i, r in enumerate(rounds):
            if i == 0:
                current.append(r)
                continue

            prev = rounds[i - 1]
            # Detect session break: round_number reset or >10 min gap
            gap_mins = 0
            try:
                t1 = _dt.fromisoformat(prev.get("timestamp", ""))
                t2 = _dt.fromisoformat(r.get("timestamp", ""))
                gap_mins = (t2 - t1).total_seconds() / 60
            except Exception:
                pass

            rn_reset = r.get("round_number", 0) < prev.get("round_number", 0)
            if rn_reset or gap_mins > 10:
                if current:
                    sessions.append(current)
                current = [r]
            else:
                current.append(r)

        if current:
            sessions.append(current)

        # Build summary dicts for last max_sessions
        result = []
        for sess in sessions[-max_sessions:]:
            wins   = sum(1 for r in sess if r.get("outcome") == "win")
            losses = sum(1 for r in sess if r.get("outcome") == "lose")
            draws  = sum(1 for r in sess if r.get("outcome") == "draw")
            total  = max(len(sess), 1)
            mode   = sess[0].get("game_mode", "?")
            ts     = sess[0].get("timestamp", "")
            try:
                date_str = _dt.fromisoformat(ts).strftime("%d %b  %H:%M")
            except Exception:
                date_str = ts[:16] if ts else "Unknown"

            # Reaction times stored as response_type=="first" won't have this — skip
            rt_vals = [r.get("reaction_ms") for r in sess
                       if r.get("reaction_ms") and r["reaction_ms"] < 3000]
            avg_rt = round(sum(rt_vals) / len(rt_vals)) if rt_vals else None

            result.append({
                "date":          date_str,
                "mode":          mode,
                "rounds_played": len(sess),
                "wins":          wins,
                "losses":        losses,
                "draws":         draws,
                "win_rate":      wins / total,
                "avg_reaction_ms": avg_rt,
            })

        return result

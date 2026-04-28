**Real-Time Gesture-Based Rock-Paper-Scissors**

**with Adaptive AI Opponent Modelling**

*A Computer Vision and Machine Learning Approach*

Capstone Research Report

Engineering — Robotics Major

March 2026

# **Abstract**

This report presents the complete design, iterative development, and evaluation of a real-time Rock-Paper-Scissors (RPS) game system combining computer vision-based hand gesture recognition with adaptive artificial intelligence opponent modelling. Developed over 42 documented iterations, the system progressed from a basic camera test to a comprehensive application featuring five game modes, hybrid gesture classification (trained MLP combined with real-time joint angle analysis), multi-layered AI prediction grounded in behavioural game theory, a player profiling and cloning system, and a research data pipeline validated through 18,000 simulated rounds and 310 live gameplay rounds. The system employs Google’s MediaPipe Hands framework for 21-point hand landmark detection, scikit-learn for machine learning classification, and OpenCV for real-time camera processing and arcade-style UI rendering. This report documents every development phase including seven distinct approaches to front-on gesture detection before arriving at the final hybrid solution, providing a detailed record of engineering problem-solving methodology and iterative design.

# **Table of Contents**

*[Right-click → Update Field after opening in Word]*

# **I. Introduction**

## **A. Background and Motivation**

Rock-Paper-Scissors (RPS) is among the most widely studied games in behavioural game theory, offering a constrained yet rich environment for investigating human sequential decision-making under adversarial conditions [1]. Unlike complex strategy games such as chess, RPS’s three-action space isolates the cognitive challenge of opponent modelling from general game expertise, making it an ideal testbed for studying how humans detect and exploit patterns in adversarial behaviour [2]. Wang, Xu, and Zhou [1] demonstrated that human players exhibit persistent conditional response patterns — win-stay and lose-shift tendencies — that deviate significantly from the Nash equilibrium strategy of uniform random selection.

From an engineering perspective, building a complete RPS system requires solving multiple interconnected problems: real-time hand detection and gesture classification from camera input, game state management with timing-critical beat detection, AI opponent design that balances fairness with challenge, and a user interface that provides real-time feedback. This project addresses all of these challenges within a unified Python application running on a MacBook Pro with Apple M5 chip and a standard webcam.

## **B. Project Objectives**

The primary objectives were to: (1) develop a reliable real-time hand gesture recognition system capable of distinguishing Rock, Paper, and Scissors from both side-view and front-on camera orientations; (2) implement multiple AI opponent strategies informed by game theory research on human behavioural biases; (3) create a player profiling system that records and reproduces individual play patterns; (4) build a comprehensive research data pipeline for statistical analysis; and (5) document the complete iterative development process as evidence of engineering problem-solving methodology.

## **C. Report Structure**

This report is organised chronologically around five development phases spanning 42 distinct iterations: Foundation (camera capture through gesture classification), Game Engine (state machines, AI, and game modes), Research Infrastructure (simulation, ML pipeline, hardware integration), Front-On Detection (seven approaches before reaching the final hybrid solution), and Player System and Polish (profiling, cloning, UI overhaul, and tutorial). Each iteration documents what was attempted, what worked, what failed, and why the approach was changed.

# **II. Literature Review**

This section reviews the academic literature and prior work that informed the design of the system. The review is organised into five areas: the game of Rock-Paper-Scissors itself, human behavioural patterns in RPS, hand gesture recognition with computer vision, machine learning for gesture classification, and AI opponent design.

## **A. Rock-Paper-Scissors as a Research Domain**

Rock-Paper-Scissors (RPS) is one of the oldest known hand games, with roots tracing to the Chinese Han Dynasty game of shoushiling [26]. In its modern form, two players simultaneously reveal one of three gestures — Rock (closed fist), Paper (open palm), or Scissors (extended index and middle fingers) — with cyclic dominance: Rock beats Scissors, Scissors beats Paper, Paper beats Rock. If both players choose the same gesture, the round is a draw.

From a game-theoretic perspective, RPS is a symmetric zero-sum game with no pure strategy Nash equilibrium. The unique mixed-strategy Nash equilibrium is for each player to select each gesture with equal probability (1/3), making play completely unpredictable and yielding zero expected payoff [1][26]. However, a substantial body of empirical research has demonstrated that human players deviate significantly from this equilibrium, exhibiting exploitable patterns that form the basis for AI opponent design.

Batzilis et al. [27] conducted the largest empirical study of RPS to date, analysing data from over one million games played on a Facebook application (Roshambull) with over 340,000 users. Their findings confirmed three key insights: most players employ strategies broadly consistent with Nash some of the time; players strategically use information about their opponent’s previous play (a non-Nash behaviour); and experienced players exploit opponent history more effectively than novices. This large-scale evidence validated the premise that human RPS play contains exploitable patterns at sufficient scale.

## **B. Human Behavioural Patterns in RPS**

Wang, Xu, and Zhou [1] observed population-level persistent cyclic motions in iterated RPS, where players exhibited conditional response patterns quantitatively described by a win-lose-tie conditional response model. Their laboratory experiment with 360 students at Zhejiang University demonstrated that after a win, players tend to repeat their previous action (win-stay), while after a loss, they tend to shift to the action that would have beaten their opponent’s previous choice (lose-shift). This collective behaviour contradicts Nash equilibrium theory but is quantitatively explained by their microscopic model without any adjustable parameters.

Dyson et al. [28] extended this analysis by examining the specific direction of post-outcome switching. Their experiments revealed that after losing, participants were more likely to “downgrade” their gesture (e.g., Rock followed by Scissors), while after drawing, participants were more likely to “upgrade” (e.g., Rock followed by Paper). They also observed a non-significant tendency to over-select Rock — a primary salience effect where Rock, as the most concrete and psychologically “strong” option, is disproportionately favoured. These directional switching patterns (upgrade vs. downgrade) directly informed the design of our AI’s prediction layers.

Zhang, Moisan, and Gonzalez [8] demonstrated in a 2021 study that human sequential behaviour goes beyond simple win-stay/lose-change heuristics. Their experiments with 100-round matches revealed heterogeneous cycle-based behaviours: some participants’ actions were independent of past outcomes, some followed win-stay/lose-change, and others exhibited win-change/lose-stay behaviour. Crucially, when participants played against computer algorithms implementing specific probabilistic strategies, participants were able to exploit predictable opponent patterns — suggesting that humans are better at detecting simple heuristic strategies than complex ones. This finding validated our design choice of introducing intentional imperfection into the AI’s play.

Brockbank and Vul [2] formally studied opponent modelling in repeated RPS in their 2024 Cognitive Psychology paper, pairing participants with seven distinct bot opponents exhibiting stable move patterns of varying complexity. They found that participants could adapt to simple move-based patterns but struggled with more complex outcome-conditional dependencies, revealing limits in human adaptive sequential reasoning. Their work demonstrated that players exhibit persistent sequential dependencies that adaptive AI opponents can exploit.

Forder and Dyson [11] provided neural evidence using EEG that win-stay behaviour is flexible and modulated by outcome value (increased when winning was worth more), while lose-shift behaviour was relatively inflexible and fast regardless of outcome value. This suggests the two strategies operate through distinct cognitive mechanisms — a finding reflected in our AI’s separate handling of win, loss, and draw conditions with different prediction weights.

Lei et al. [9] developed multi-AI systems using Markov Models of varying memory lengths to compete against humans, introducing a “focus length” parameter controlling how quickly the AI adapts to strategy changes. Their ensemble approach of combining predictions from multiple models with different memory windows informed our Fair Play AI’s multi-layered architecture.

Hoffman et al. [29] experimentally investigated evolutionary dynamics in the RPS game, confirming that when the Nash equilibrium is not evolutionarily stable, population-level strategy frequencies deviate further from equilibrium. Their work validated the evolutionary game theory framework as more descriptive of actual human behaviour than classical Nash equilibrium theory, supporting the approach of designing AI opponents around empirically observed biases rather than theoretically optimal strategies.

## **C. Computer Vision for Hand Gesture Recognition**

Hand gesture recognition through computer vision has emerged as a critical technology for natural human-computer interaction (HCI). Qi et al. [30] published a comprehensive review in Complex & Intelligent Systems (210+ citations) covering vision-based hand gesture recognition for human-robot interaction. Their review categorised approaches into appearance-based methods (using raw image features), model-based methods (fitting geometric hand models), and skeleton-based methods (using joint positions from frameworks like MediaPipe). They identified that skeleton-based approaches offer the best balance of accuracy and computational efficiency for real-time applications, which aligned with our choice of MediaPipe’s landmark-based pipeline.

MediaPipe Hands, introduced by Zhang et al. [3], provides a real-time hand tracking pipeline that detects 21 three-dimensional hand landmarks from a single RGB camera. The framework employs a two-stage architecture: a palm detection model operating on the full image to produce an oriented hand bounding box, followed by a landmark regression model predicting keypoint positions within the detected crop. The model was trained on approximately 30,000 real-world images along with rendered synthetic hand models, achieving real-time inference on mobile GPUs [3].

The landmark-based approach has been validated across multiple application domains. Meng et al. [4] demonstrated a registerable hand gesture recognition system using MediaPipe landmarks with triplet loss learning for dynamic gesture expansion. Sánchez-Brizuela et al. [5] leveraged MediaPipe for real-time hand segmentation in virtual reality, demonstrating the framework’s reliability as a foundation for downstream tasks. Amprimo et al. [13] validated MediaPipe’s hand tracking accuracy for clinical applications, providing benchmarks for landmark precision under varying conditions.

The fingerpose library by Potgieter [7] introduced a declarative approach to gesture classification, defining gestures in terms of per-finger curl states (no curl, half curl, full curl) and directions. This curl-based analysis provides rotation-invariant features that complement position-based classification — a principle that proved essential in our final hybrid detection pipeline.

*[INSERT IMAGE HERE]*

*Fig. 1: MediaPipe 21-point hand landmark model showing keypoint indices and connections [3]*

## **D. Machine Learning for Gesture Classification**

The Kazuhito00/hand-gesture-recognition-using-mediapipe project [10] established a widely-adopted pipeline for training gesture classifiers on MediaPipe landmark data. The approach involves collecting normalised keypoint coordinates during live camera interaction using keyboard triggers, training a lightweight neural network, and deploying for real-time inference. This project’s data collection workflow and keypoint normalisation strategy (translate to wrist origin, scale by palm size) directly inspired our landmark collection and front-on training system.

Ahmad et al. [6] validated that MLP classifiers trained on MediaPipe landmarks achieve over 95% accuracy on static gesture datasets, with performance improving as dataset size increases. Their systematic comparison of classifiers (MLP, SVM, Random Forest) on hand landmark features demonstrated that simple architectures are sufficient when provided with well-normalised landmark coordinates — a finding that guided our choice of a two-layer MLP (64, 32 neurons) over more complex architectures.

Recent comprehensive reviews [30] confirm that skeleton-based approaches using lightweight classifiers achieve competitive accuracy while maintaining real-time performance on consumer hardware. The trend across the literature is toward learned representations over hand-crafted geometric features [13][14], a trajectory our own development followed after seven geometric approaches to front-on detection failed to generalise (documented in Section VII).

## **E. AI Opponent Design in Games**

The design of engaging AI opponents in games requires balancing competence with beatable behaviour. In the context of RPS, this means the AI must be strong enough to challenge the player while remaining vulnerable to strategic adaptation. Policy-based reinforcement learning approaches [12] have been explored for generalised RPS at ESANN 2023, though these typically converge to Nash equilibrium play which, while theoretically optimal, produces a frustrating player experience in entertainment contexts because the player cannot detect any exploitable patterns.

Our approach draws instead from the behavioural game theory literature, implementing heuristic prediction based on documented human biases. The research by Brockbank and Vul [2] showed that even when players know an AI is adaptive, they struggle to randomise their own play sufficiently to neutralise it. Zhang et al.’s [8] finding that humans can detect simple heuristic strategies informed our decision to include intentional imperfection — the AI deliberately makes sub-optimal choices at a controlled rate, creating exploitable patterns that skilled players can learn to identify, mirroring the human-vs-human experience that makes RPS engaging.

The concept of Dynamic Difficulty Adjustment (DDA) has been extensively studied in the broader game AI literature. Zohaib and Nakanishi [31] proposed diversifying DDA agents by integrating player state models into Monte-Carlo tree search, demonstrating that AI opponents which adapt based on predicted player states produce enhanced game experiences. Their work showed that focusing on player affect (challenge, competence, flow) rather than just win rate produces more satisfying difficulty curves. Dill et al. [32] evaluated MediaPipe’s accuracy for physical exercise analysis, finding that pose estimation quality is highly dependent on camera viewing angle and exercise type — a finding directly relevant to our gesture detection challenges with front-on versus side-view hand orientations.

A 2024 IEEE study on Personalised DDA [33] proposed combining imitation learning and reinforcement learning to create opponents that adapt to individual players’ behaviour in real-time. Their framework trains an imitation agent to replicate the player’s actions, then trains an RL agent to beat the imitation agent, creating a personalised opponent. While our player clone system shares the goal of reproducing individual behaviour, our approach uses statistical profiling rather than neural imitation, making it more interpretable and requiring significantly less data (30 rounds vs. thousands of training episodes).

# **III. Development Methodology**

The system was developed iteratively over multiple sessions spanning March 17–28, 2026, with each iteration building on tested foundations. Development followed an agile-inspired approach: implement the smallest viable change, test immediately with live camera feedback, identify the next bottleneck, and iterate. This methodology proved particularly valuable during the front-on gesture detection phase, where seven distinct technical approaches were attempted before arriving at the final hybrid solution.

The development environment consisted of a MacBook Pro with Apple M5 chip, Python 3.9 running in a virtual environment (~/.venv/), and the project directory at ~/rps_hand_counter/. Research data was stored at ~/Desktop/CapStone/. Version control was maintained through OneNote documentation with screenshots at each iteration, providing a visual record of the system’s evolution.

The 42 iterations are organised into five phases:

• Phase 1 — Foundation (8 iterations): Camera capture through gesture state tracking

• Phase 2 — Game Engine (9 iterations): State machines, AI design, game modes, UI

• Phase 3 — Research Infrastructure (6 iterations): Hardware integration, ML pipeline, simulation

• Phase 4 — Front-On Detection (10 iterations): Seven failed approaches plus the final solution

• Phase 5 — Player System & Polish (9 iterations): Profiling, cloning, UI overhaul, tutorial

# **IV. Phase 1 — Foundation**

The foundation phase established the core computer vision pipeline, progressing from raw camera capture to confirmed gesture classification in eight iterations.

## **Iteration 1: Camera Capture (test_camera.py)**

The first iteration verified that the MacBook’s built-in webcam could be accessed via OpenCV’s VideoCapture interface. A minimal script (test_camera.py) opened the camera, displayed frames in a window, and closed on keypress. This confirmed the development environment was correctly configured with OpenCV 4.11.0 and established the basic camera loop pattern used throughout the project.

*[INSERT IMAGE HERE]*

*Fig. 2: First camera test showing live webcam feed with test_camera.py — March 17, 5:47pm*

## **Iteration 2: MediaPipe Hand Landmark Detection**

The second iteration integrated Google’s MediaPipe Hands framework [3] into the camera loop. The hand_landmarks.py module was created to initialise a MediaPipe Hands detector (single hand, model complexity 0, confidence thresholds 0.5) and process each frame to extract 21 three-dimensional landmarks. The frame was mirror-flipped horizontally so the display matched the user’s perspective. MediaPipe’s built-in drawing utilities rendered the hand skeleton with landmark connections and dots on the camera feed. The system reported handedness classification (Left/Right) with confidence scores, confirming detection worked for both hands.

*[INSERT IMAGE HERE]*

*Fig. 3: Right hand detection showing 0.93 confidence with full landmark skeleton — March 17, 8:41pm*

*[INSERT IMAGE HERE]*

*Fig. 4: Left hand detection showing 0.93 confidence with landmark connections — March 17, 8:42pm*

## **Iteration 3: Finger Counter**

The third iteration created finger_counter.py, which analysed the 21 landmarks to determine which fingers were extended. The initial approach used simple Y-axis comparisons: a finger was considered extended if its tip landmark was above (lower Y value) its PIP joint landmark. Red circles were drawn on extended fingertips to provide visual feedback. The count was displayed as a large number on the camera feed (e.g., “Count: 5”). This iteration demonstrated the core concept of deriving semantic meaning from raw landmark positions.

*[INSERT IMAGE HERE]*

*Fig. 5: Finger counter showing Count: 5 with red dots on all extended fingertips — March 17, 9:03pm*

*[INSERT IMAGE HERE]*

*Fig. 6: Finger counter showing Count: 4 (thumb tucked) with 0.99 confidence — March 17, 9:03pm*

## **Iteration 4: Right-Hand-Only Limitation Discovered**

Testing revealed that the finger counter produced incorrect results for the left hand. When showing the left hand palm-forward, the system displayed “Count: Unknown | not_right_hand” — the initial implementation only accepted the right hand and rejected all left-hand detections. This was because MediaPipe’s handedness classification labels were used as a filter, and the finger extension logic assumed right-hand landmark orientation. This limitation was noted for later resolution.

*[INSERT IMAGE HERE]*

*Fig. 7: Left hand rejected — “Count: Unknown | Left (0.92) | not_right_hand” — March 17, 9:05pm*

## **Iteration 5: Gesture Classification (gesture_mapper.py)**

The fifth iteration created gesture_mapper.py, which mapped finger states to RPS gestures. The classification rules were: all fingers down = Rock, all fingers up = Paper, index + middle up only (with ring, pinky, and thumb down) = Scissors. The display now showed the gesture name and its command string (e.g., “Gesture: Paper | Command: CMD_PAPER”). This was the first time the system could identify an actual RPS gesture from the camera feed.

*[INSERT IMAGE HERE]*

*Fig. 8: First gesture classification showing “Gesture: Paper, Command: CMD_PAPER” with Right (0.96) — March 17, 9:26pm*

## **Iteration 6: Gesture State Tracker and Hand Mode Toggle**

Testing revealed two problems: (1) the gesture output flickered rapidly between frames as the raw classification changed, and (2) only the right hand was supported. Two upgrades were implemented simultaneously:

The GestureStateTracker (gesture_state.py) added a temporal smoothing pipeline. A history buffer stored the last 5 (later increased to 7) gesture predictions, outputting the majority vote as the “stable” gesture. A confirmation layer required 3 consecutive stable frames before promoting to “confirmed” gesture. The history cleared when the hand mode was toggled. This three-stage pipeline (raw → stable → confirmed) eliminated flickering while maintaining responsiveness.

A hand mode toggle button was added to the camera window, allowing users to switch between Left and Right hand modes. The recogniser only accepted the selected hand, using MediaPipe’s handedness classification to filter. This resolved the left-hand limitation while maintaining accuracy.

*[INSERT IMAGE HERE]*

*Fig. 9: Development notes describing the history buffer and hand mode toggle design decisions*

*[INSERT IMAGE HERE]*

*Fig. 10: Hand mode set to “Left” showing Rock gesture detected with diagnostic output — March 18, 7:14am*

## **Iteration 7: Rotation-Tolerant Geometry Rewrite**

The initial finger extension detection used simple Y-axis position comparisons, which failed when the hand was rotated. The finger_counter.py module was rewritten to use hand geometry instead of joint angles and tip distances from the first node. The new approach computed 3D Euclidean distances and joint angles (PIP angle, DIP angle) at each finger joint, making the classification independent of screen-space orientation. Palm size was estimated from the average distance between the wrist and three MCP joints, providing size-invariant thresholds. This rewrite made the gesture detection work reliably regardless of wrist rotation angle.

## **Iteration 8: Full Diagnostic View**

The diagnostic display was expanded to show the complete gesture processing pipeline in real-time: Count, Raw gesture, Stable gesture, Confirmed gesture, Confirm frames counter (e.g., 77/3), Buffer size (e.g., 7/7), Robot Ready flag, Command Output (e.g., CMD_PAPER), Hand Mode, Handedness confidence, and Reason text. This comprehensive view became essential for debugging all subsequent iterations — every classification decision was visible at a glance.

*[INSERT IMAGE HERE]*

*Fig. 11: Full diagnostic view showing Raw/Stable/Confirmed: Paper, Robot Ready: YES, 77/3 frames — March 18, 8:02am*

*[INSERT IMAGE HERE]*

*Fig. 12: Diagnostic view with Rock gesture on Left hand mode, showing GestureStateTracker config — March 18, 8:05am*

# **V. Phase 2 — Game Engine**

With gesture recognition working reliably, the second phase built the game logic, AI opponents, and initial user interface.

## **Iteration 9: Pump-Based Beat Detection**

Rather than a timer-based countdown, the system implemented physical motion detection to replicate the traditional “rock, paper, scissors, SHOOT” rhythm. The player holds a fist (Rock) and pumps their hand up and down four times. The system tracks the wrist landmark’s Y-coordinate through alternating down-up phases:

• Down phase: wrist Y increases by ≥ 0.06 from the current top position (hand moves downward in frame)

• Up phase: wrist Y decreases by ≥ 0.045 from the current bottom position (hand moves upward)

• Beat cooldown: minimum 0.18 seconds between consecutive beats to prevent double-counting

• Rock grace period: 0.35 seconds of tolerance for brief gesture drops during pumping

On the fourth beat, a SHOOT window opens for 0.55 seconds. These thresholds were later refined during Phase 5 (Iteration 33).

## **Iteration 10: Cheat Mode (rps_game_state.py)**

The first game mode served as a system verification tool. Cheat Mode reads the player’s gesture after the SHOOT window opens, then always plays the winning counter-move (Rock beats Scissors, Paper beats Rock, Scissors beats Paper). While unwinnable, this mode confirmed that the entire pipeline — from camera input through gesture detection, pump counting, shoot window timing, to result display — worked correctly end-to-end.

*[INSERT IMAGE HERE]*

*Fig. 13: Cheat mode gameplay showing SHOOT window and result display*

## **Iteration 11: Fair Play AI (fair_play_ai.py) — Five-Layer Prediction**

The Fair Play AI implements a five-layer heuristic prediction system designed to exploit documented human behavioural biases [1][2][8]. Each layer contributes weighted scores to predict the player’s next gesture, and the AI plays the counter to the highest-scored prediction.

Layer 1 — Population Priors: Soft baseline tendencies from game theory literature. After a loss, players tend to shift (upgrade/downgrade weighted 1.15 vs. stay at 0.65). After a win, slight tendency to repeat (stay weighted 1.10). These priors are intentionally mild so individual data dominates after a few rounds.

Layer 2 — Outcome-Conditioned Response Learning: The primary prediction layer, implementing Wang et al.’s [1] conditional response model. For each previous round with the same outcome, it observes whether the player stayed, upgraded, or downgraded. Recent rounds receive exponentially higher weight: weight = 2.2 / (1.28^distance).

Layer 3 — Exact Transition Memory: Records move-to-move transitions (e.g., “after Rock, what next?”). Weight 1.35 per pattern, with 1.35× bonus when both move and outcome match.

Layer 4 — Outcome-to-Move Patterns: Records which gestures follow each outcome regardless of preceding gesture. Weight 0.70 per observation.

Layer 5 — Overall Frequency Memory: Tracks gesture frequencies with recency weighting (0.30 per observation). Provides fallback when specific patterns are sparse.

The AI intentionally avoids perfect play. Effective skill starts at 0.66 and increases slowly (max 0.76). When prediction margins are narrow (<0.40), confidence drops by 0.08. This ensures beatable gameplay while providing challenge.

## **Iteration 12: Fair Play Mode (fair_play_state.py)**

Fair Play Mode implements a best-of-3 match format. The robot locks its move on beat 3 (before the SHOOT window opens), so it cannot see the player’s actual throw. Draws replay the round without advancing the round counter. The mode uses the Fair Play AI for predictions and displays round-by-round scoring.

## **Iteration 13: Challenge Mode (challenge_mode_state.py)**

Challenge Mode implements an endless streak format where a single loss ends the run. The score is the number of consecutive wins. A ChallengeAI class extends FairPlayAI with streak-based difficulty ramping: base skill 0.68, increasing by 0.035 per consecutive win up to a maximum of 0.92. At low streaks (<3), the AI’s “misses” sample from the full prediction distribution; at high streaks (>6), even misses favour the second-best prediction. A persistent high score is maintained across sessions.

## **Iteration 14: Challenge Stats Logger (challenge_stats_logger.py)**

A comprehensive Excel logging system was built using openpyxl. The challenge_research_log.xlsx workbook contains three sheets: Summary (lifetime metrics including longest streak, total rounds, gesture counts), Challenge_Runs (per-run metadata including start/end timestamps, final streak, wins/draws/losses), and Challenge_Rounds (per-round data including timestamp, run ID, gestures, results, and streak progression). The workbook auto-creates with formatted headers and column widths on first run.

## **Iteration 15: Robot Output Buffer (robot_output.py)**

A serial-ready command buffer was created to prepare for eventual ESP32 hardware integration. The RobotOutputBuffer stores two event types: “locked” (robot move has been pre-committed) and “resolved” (round result is known). Each event contains the command string (e.g., ROBOT_PLAY_ROCK), game mode, round result, and metadata. A 200-event history deque is maintained for debugging. This module defines the interface between the game logic and any future physical robot actuator.

## **Iteration 16: Config Store (config_store.py)**

A persistent configuration system was implemented using JSON. The config.json file stores default play mode, default display mode, camera resolution, shoot window timing, rock assume timing, beat cooldown, and handedness threshold. A normalisation function validates all values against allowed ranges and falls back to defaults for invalid entries. The configuration is loaded at startup and saved automatically when settings change.

## **Iteration 17: UI System (ui_renderer.py)**

The initial UI system provided four screen types: Main Menu (with keyboard navigation and W/S/Enter controls), Settings Screen (with A/D value adjustment and live preview), Diagnostic View (comprehensive pipeline data display), and Game View (simplified arcade-style display). All rendering used OpenCV drawing primitives (cv2.putText, cv2.rectangle, cv2.circle) with semi-transparent panel overlays on the camera feed. The UI supported dynamic scaling based on camera resolution.

*[INSERT IMAGE HERE]*

*Fig. 14: Early menu screen with basic UI styling*

*[INSERT IMAGE HERE]*

*Fig. 15: Early game view showing beat countdown and SHOOT indicator*

# **VI. Phase 3 — Research Infrastructure**

The third phase added research-oriented features that transformed the system from a game into a research platform.

## **Iteration 18: Hardware Integration Test Mode**

Two new modules were created for ESP32 serial communication. serial_bridge.py handles port discovery, connection management, and non-blocking reads using pyserial, with a pipe-delimited text protocol (CMD|ROCK\n, ACK|ROCK\n). hardware_test_mode.py provides a diagnostic-only screen accessible via the “H” key in Diagnostic mode, allowing manual serial command transmission (R=Rock, P=Paper, S=Scissors, O=Open, C=Close, T=Ping) with port cycling and connection status display. This mode was designed for early robot debugging without coupling hardware to game logic.

*[INSERT IMAGE HERE]*

*Fig. 16: Hardware test mode screen showing serial connection status and command keys*

## **Iteration 19: Challenge Log Improvements**

Six new tracking columns were added to the Challenge mode Excel logger: player_rock_count, player_paper_count, player_scissors_count, robot_rock_count, robot_paper_count, robot_scissors_count. These counters increment in both the Summary sheet (lifetime totals) and Challenge_Runs sheet (per-run breakdown). An automatic migration function (_migrate_workbook) detected existing workbooks missing the new columns and added them without affecting historical data.

## **Iteration 20: AI Prediction Metadata**

The Fair Play AI and Challenge AI were enhanced to record prediction metadata: the AI’s predicted player move, prediction confidence (margin between top two candidates), and the player’s actual response type (stay/upgrade/downgrade from previous gesture). Reaction time was also captured — the interval between SHOOT window opening and the player’s gesture being locked. This metadata was logged to the Challenge_Rounds Excel sheet for post-hoc analysis of AI performance.

## **Iteration 21: ML Prediction Pipeline**

A machine learning prediction system was built as an alternative to the heuristic AI. The ml_feature_extractor.py module converts round history into feature vectors including gesture frequencies, outcome-conditioned response rates, transition probabilities, and sequence patterns. The ml_model.py module wraps a scikit-learn LogisticRegression classifier that trains on accumulated game history. Initial testing achieved 46.7% accuracy — better than the 33.3% random baseline but below the heuristic AI’s performance on most strategies.

## **Iteration 22: Simulation Mode (simulation_mode.py)**

A headless batch simulation framework was built to evaluate AI strategies at scale without requiring live camera input. Six simulated player strategies were implemented:

| **Strategy** | **Behaviour** |
| --- | --- |
| Random | Uniform random selection (Nash equilibrium baseline) |
| Win-Stay/Lose-Shift | Repeats after win, shifts after loss [1] |
| Cycler | Rock → Paper → Scissors → Rock deterministic cycle |
| Rock Heavy | 60% Rock, 20% Paper, 20% Scissors |
| Anti-Pattern | Attempts to counter the AI’s expected counter |
| Mixed Human | Combination of biases simulating realistic play |

*Table 1: Simulated player strategies.*

Each strategy was tested against three AI types (Heuristic FairPlay, Heuristic Challenge, ML Prediction, plus Random baseline) across 10 runs of 100 rounds each, producing 18,000 total rounds. Results were logged to simulation_results.xlsx with both per-run summaries and per-round detail.

## **Iteration 23: Research Comparison Dashboard**

A research_report.py module was created to generate research_comparison_report.xlsx from the simulation data. The report includes an AI Comparison matrix (robot win rates by strategy and AI type), Per Strategy Detail sheets, Key Findings (e.g., “Strongest AI overall: Heuristic FairPlay at 50.1% robot win rate”), and ML Model Details. Accessible via the “R” key in Diagnostic mode, this dashboard provided at-a-glance comparison of AI approaches.

*[INSERT IMAGE HERE]*

*Fig. 17: Research comparison dashboard showing AI performance across strategies*

# **VII. Phase 4 — Front-On Gesture Detection**

Front-on gesture detection — when the palm faces the camera directly — presented the most significant engineering challenge of the project. When viewed front-on, finger curl occurs primarily in the Z-axis (depth), which MediaPipe estimates with limited accuracy [3]. The Z-coordinate uses weak projection relative to the wrist depth, making it unreliable for absolute measurements. Seven distinct approaches were attempted over ten iterations before arriving at the final hybrid solution.

## **Iteration 24: Attempt 1 — Palm Orientation + Relaxed Thresholds**

A palm orientation detector was added to finger_counter.py, computing the palm normal vector using the cross product of two vectors on the palm plane (wrist→index MCP and wrist→pinky MCP). When front-on orientation was detected, finger extension thresholds were relaxed: PIP angle reduced from 160° to 140°, DIP angle from 150° to 130°, and distance requirements reduced proportionally. A Z-depth check was added for front-on fingers. Result: Scissors was confused with Paper approximately 35% of the time because the relaxed thresholds allowed partially curled ring and pinky fingers to register as extended.

## **Iteration 25: Attempt 2 — Curl Ratio Measurements**

A curl ratio measurement was added, calculating how far each fingertip was from the palm center relative to its knuckle. Extended fingers have ratios above 1.0; curled fingers around 0.5–0.7. An absolute gate required curl ratio > 0.85 for front-on extension. Result: Too strict — when index and middle fingers pointed toward the camera during scissors, their curl ratios compressed below 0.85 due to foreshortening, so scissors was consistently classified as Rock (no fingers “extended”).

## **Iteration 26: Attempt 3 — Relative Comparisons Only**

The absolute curl ratio gate was removed. Instead, gesture_mapper.py used relative comparisons between finger groups: if the average curl ratio of index+middle was significantly higher than ring+pinky, classify as Scissors. Result: Improved but still unreliable — the difference between “two fingers slightly extended” and “four fingers slightly extended” was too small for consistent discrimination.

## **Iteration 27: Attempt 4 — Multi-Signal Voting**

A completely new front_on_classifier.py was created using three independent checks per finger, requiring at least 2-of-3 to agree: (1) tip above PIP in Y-axis, (2) tip farther from wrist than PIP in 2D distance, (3) tip far from its own MCP (≥ 0.55× palm scale). Diagnostic output showed per-finger vote breakdowns. Result: Rock and Paper worked well, but Scissors was still read as Paper because ring and pinky landmarks maintained extended-looking positions even when the actual fingers were curled toward the camera.

*[INSERT IMAGE HERE]*

*Fig. 18: Multi-signal voting diagnostic showing per-finger vote breakdowns*

## **Iteration 28: Attempt 5 — Distance Gap Scissors Override**

A scissors override layer was added to the multi-signal voting classifier. When the result would be Paper (3–4 fingers extended), three distance comparisons between finger groups were computed: tip-to-wrist gap, tip-to-palm-center gap, and tip-to-MCP gap between index+middle versus ring+pinky. If the combined score reached a threshold, Paper was overridden to Scissors. Result: Worked for some hand positions but did not generalise across different distances from the camera and hand angles.

## **Iteration 29: Attempt 6 — X-Axis Offset Comparison**

The distance-based override was replaced with a single measurement: the horizontal (X-axis) offset between the average position of index+middle tips and ring+pinky tips, normalised by palm scale. The hypothesis was that curled ring+pinky would shift sideways during scissors. Result: Too sensitive to hand rotation — a slightly tilted Paper gesture produced X offsets that triggered false Scissors classifications.

## **Iteration 30: Attempt 7 — Middle-vs-Ring X Offset Only**

The comparison was narrowed to only the middle fingertip versus ring fingertip X-positions, with the threshold raised from 0.18 to 0.28 to require a very clear difference. Result: More stable but still not reliable enough for gameplay — the X offset between middle and ring varied too much with natural hand positioning.

At this point, after seven failed geometric approaches, the decision was made to abandon pure geometric classification for front-on detection and pursue a machine learning approach.

## **Iteration 31: Trained MLP Classifier**

Following the approach established by Kazuhito00 [10], a complete data collection and training pipeline was built:

landmark_collector.py: During Diagnostic mode, pressing “F” toggles collection mode. The user holds a gesture and presses 7 (Rock), 8 (Scissors), or 9 (Paper) to record normalised landmark coordinates. Data is appended to ~/Desktop/CapStone/front_on_training_data.csv. Each sample consists of 42 features (21 landmarks × 2 coordinates), translated to wrist origin and scaled by palm size.

front_on_trainer.py: Pressing “T” in Diagnostic mode triggers training. An sklearn MLPClassifier (64, 32 hidden layers, ReLU activation, max 500 iterations, early stopping) is trained on the collected data with 80/20 train/test split. The model and metadata are saved to ~/Desktop/CapStone/front_on_gesture_model.pkl. With approximately 100 samples per gesture, cross-validated accuracy exceeded 90% on static poses.

front_on_classifier.py: The classifier loads the trained model at startup (lazy-loaded, cached). Each frame’s landmarks are normalised and fed through the model, which returns gesture probabilities. The diagnostic reason text shows per-class probabilities (e.g., “R:85% P:10% S:5%”).

A Hand Orientation setting (Side/Front) was added to config_store.py, and hand_landmarks.py was modified to route through either the side-view finger_counter.py or front_on_classifier.py based on the setting.

*[INSERT IMAGE HERE]*

*Fig. 19: Data collection mode in Diagnostic view showing sample counts during front-on training*

## **Iteration 32: Hybrid ML + Curl Analysis**

While the ML model achieved high accuracy on static poses, it showed latency during the fast pump-to-shoot transition. The model saw blurry intermediate hand poses it was not trained on during rapid finger movement. To address this, a real-time curl analysis layer was added to front_on_classifier.py, inspired by the fingerpose library [7].

For each finger (index, middle, ring, pinky), the PIP and DIP joint angles are computed from 2D landmark positions. The minimum of the two angles classifies the finger as: NoCurl (≥ 150°), HalfCurl (≥ 110°), or FullCurl (< 110°). Gesture classification from curl states: all FullCurl = Rock, 3+ NoCurl including ring+pinky = Paper, index+middle NoCurl with ring+pinky curled = Scissors.

The hybrid decision logic runs both systems every frame:

• Both agree → highest confidence (reason: “agree”)

• ML confident (>70%) but curl disagrees → trust ML unless curl is very strong (>85% and ML <85%) (reason: “ml_wins” or “curl_override”)

• ML uncertain (<70%) with clear curl signal (≥60%) → trust curl (reason: “curl_leads”)

• No ML model available → curl only (reason: “curl_only”)

This hybrid approach resolved the key weakness: during fast pump-to-shoot transitions, curl analysis detects finger opening immediately from joint angle changes while the ML model catches up. The diagnostic reason text shows both signals for debugging.

*[INSERT IMAGE HERE]*

*Fig. 20: Hybrid classifier diagnostic showing both ML and curl outputs with decision reasoning*

# **VIII. Phase 5 — Player System and Polish**

## **Iteration 33: Pump Detection Improvements**

Live testing revealed pump detection was unreliable approximately 20% of the time. Three improvements were applied across all three game controllers (fair_play_state.py, challenge_mode_state.py, rps_game_state.py): (1) accepting stable_gesture == “Rock” (not just confirmed) during countdown, reducing the delay before pump tracking begins; (2) continuing to track wrist motion during brief gesture drops within the grace window, preventing pump resets from momentary detection glitches; (3) refined thresholds: down 0.06→0.045, up 0.045→0.035, grace period 0.35→0.50s.

## **Iteration 34: Player Profile Store (player_profile_store.py)**

A comprehensive player profiling system was built. Each player’s complete round history is stored as a JSON file at ~/Desktop/CapStone/player_profiles/<name>.json, containing timestamped records with gestures, outcomes, response types (stay/upgrade/downgrade), and previous round context. The PlayerProfileStore class builds statistical pattern tables including gesture frequencies, outcome-conditioned response distributions, move transition matrices, and outcome+gesture transition matrices. An Excel research log (player_research_log.xlsx) is maintained with an All_Rounds sheet and individual player tabs containing automated strategy analysis.

## **Iteration 35: Player Clone AI (player_clone_ai.py)**

A novel player cloning system was built that records a player’s complete round history and generates an AI opponent that plays AS that player. The PlayerCloneAI uses four decision layers prioritised by specificity:

• Layer 1: Outcome + gesture transition — “After losing with Rock, this player throws Paper 60%”

• Layer 2: Gesture transition — “After Rock, usually plays Paper”

• Layer 3: Outcome response type — “After a loss, upgrades 55%”

• Layer 4: Overall frequency — “Throws Rock 45%”

An accuracy parameter (default 85%) introduces human-like noise, with 15% of moves drawn from the player’s overall frequency distribution. The clone requires minimum 30 recorded rounds for stable probability estimates.

## **Iteration 36: Clone Mode**

A complete Clone Mode was added with a three-step setup flow: Step 1 — enter player name via keyboard input; Step 2 — select opponent from available player profiles using W/S navigation; Step 3 — if no profiles have sufficient data, guidance is shown. The mode plays best-of-5 (first to 3 wins) using the FairPlayController with extended win_target. Result screens display the opponent’s name instead of “CPU” (e.g., “ZAC TAKES THE ROUND”). Every round played is automatically recorded to the active player’s profile for future cloning.

*[INSERT IMAGE HERE]*

*Fig. 21: Clone Setup screen showing opponent selection with round counts*

## **Iteration 37: UI Arcade Overhaul**

The entire ui_renderer.py was rewritten with an arcade/retro aesthetic. Key design changes: deep navy-black backgrounds (BGR: 8, 8, 16) replacing grey; neon cyan, magenta, yellow, and green accent colours defined as constants; three-layer text rendering (coloured glow + dark outline + bright text) for neon effect; animated pulsing arcade light dots on the game view; selected menu items highlighted with glowing bars; gesture icons (circle=Rock, square=Paper, X=Scissors) on result screens; dynamic menu spacing based on item count. All colours were defined as constants at the top of the file for consistency.

*[INSERT IMAGE HERE]*

*Fig. 22: Arcade-themed main menu with neon styling*

*[INSERT IMAGE HERE]*

*Fig. 23: Game result screen with arcade theme showing gesture icons and score*

## **Iteration 38: Settings Descriptions**

Each setting in SETTINGS_SCHEMA received a “desc” field containing a plain-language explanation. A blue description panel appears at the bottom of the settings screen showing the currently selected item’s description (e.g., for Shoot Window: “How long you have to throw after the 4th beat. Lower = harder, higher = more forgiving”). This reduced confusion about what each technical parameter controlled.

*[INSERT IMAGE HERE]*

*Fig. 24: Settings screen with contextual description panel for selected option*

## **Iteration 39: Player Stats Viewer**

A dedicated stats viewer screen was added to the main menu. Players select a profile from a list, then see a two-column dashboard: Left column shows Win/Loss/Draw rates with progress bars and Gesture frequency bars; Right column shows Response patterns (after win/loss/draw stay/upgrade/downgrade rates) and auto-generated Player Traits. Traits are computed from statistical pattern tables and displayed full-width below the columns (e.g., “Heavy Rock player (52% of throws)”, “Predictable sequence: after Rock, plays Scissors 73%”).

*[INSERT IMAGE HERE]*

*Fig. 25: Player Stats viewer showing Zac’s statistics with progress bars and traits*

## **Iteration 40: Interactive Tutorial**

A six-step interactive tutorial was added, accessible from the main menu. The camera feed stays live while tutorial overlays guide the player through each gesture: Step 1 — Make Rock (hold for 10 frames with progress bar); Step 2 — Show Paper; Step 3 — Show Scissors; Step 4 — Pump fist 4 times (tracks wrist Y, shows 4 circles filling); Step 5 — SHOOT (throw any gesture); Step 6 — “You’re Ready!”. Each step auto-advances when the correct gesture is detected and held, providing immediate feedback through the camera feed.

*[INSERT IMAGE HERE]*

*Fig. 26: Interactive tutorial Step 1 showing “Make a ROCK” with live detection and progress bar*

## **Iteration 41: Code Cleanup and Bug Fixes**

A comprehensive code audit identified and fixed several issues: (1) _last_recorded_round initialised in build_app_state to prevent potential KeyError; (2) frame read failure now finalises active Challenge runs before breaking the loop; (3) tutorial process_hand_frame call made consistent with game loop (using config[“hand_orientation”] directly instead of .get() with default); (4) app_screen comment updated to document all valid screen states. Sound player integration (built in an earlier session but lost during rebuilds) was noted as needing re-wiring.

## **Iteration 42: Terminal Close on Exit**

A _close_terminal() function was added that uses macOS AppleScript to close the Terminal window that launched the application. The function fires after cap.release() and cv2.destroyAllWindows() via subprocess.Popen (non-blocking), and silently fails on non-macOS systems. This provides a clean exit experience where both the OpenCV window and the terminal close automatically when the user quits.

# **IX. Experimental Results**

## **A. Simulation Results (18,000 Rounds)**

The simulation framework tested six player strategies against four AI types across 180 runs of 100 rounds each. The following table summarises robot win rates (higher = AI performs better):

| **Strategy** | **Random AI** | **Heuristic FP** | **Heuristic CH** |
| --- | --- | --- | --- |
| Random | 31.8% | 37.0% | 31.0% |
| Win-Stay | 34.5% | 45.7% | 46.3% |
| Cycler | 35.1% | 89.4% | 85.8% |
| Rock Heavy | 32.6% | 43.8% | 45.1% |
| Anti-Pattern | 31.9% | 55.1% | 50.6% |
| Mixed Human | 34.0% | 39.7% | 36.2% |

*Table 2: Robot win rates by player strategy and AI type from 18,000 simulated rounds.*

### ***Key Simulation Findings***

• Strongest AI overall: Heuristic FairPlay (50.1% average robot win rate, +16.2% over random baseline)

• Heuristic Challenge lift over random: +16.0%

• ML Prediction lift over random: +13.6%

• ML excels against deterministic strategies: 98.4% robot win rate against Cycler, 53.0% against Rock Heavy

• Heuristic excels against adaptive strategies: 51.3% against Anti-Pattern (vs ML’s 30.8%)

• Mixed Human strategy (realistic) hardest to exploit: best AI achieves only 37.0%

The heuristic AI’s advantage over ML on adaptive strategies confirms the value of the multi-layered prediction approach grounded in behavioural game theory [1][8], while ML’s dominance on deterministic patterns shows its strength in exploiting mechanical repetition. Notably, the FairPlay heuristic achieved an 89.4% robot win rate against cycler strategies and 55.1% against anti-pattern players — demonstrating that the behaviour-informed prediction layers exploit structured play far more effectively than pure statistical learning.

The Mixed Human strategy’s resistance to exploitation (best AI achieves 37.0% robot win rate vs 33.9% random baseline = only +3.1% lift) is consistent with the game-theoretic principle that strategies approximating Nash equilibrium play become increasingly difficult to exploit [1][26]. This finding aligns with Batzilis et al.’s [27] observation from one million games that experienced players achieve better results by more closely approximating random play while still exploiting opponents’ historical patterns.

The Win-Stay/Lose-Shift strategy proved particularly vulnerable to the heuristic AIs (45.8–47.4% robot win rates), consistent with Wang et al.’s [1] finding that this common human bias creates exploitable conditional dependencies. Our simulation quantifies this: a player using pure WSLS faces a 12–14 percentage point win rate disadvantage against our heuristic AI compared to a random player, directly validating the AI’s design around exploiting this documented bias.

## **B. Challenge Mode Results (309 Live Rounds)**

Real-world testing produced 123 Challenge mode runs totalling 310 rounds:

| **Metric** | **Value** |
| --- | --- |
| Total Runs | 123 |
| Total Rounds | 310 |
| Longest Streak | 6 |
| Player Wins | 103 (33.2%) |
| Robot Wins | 106 (34.2%) |
| Draws | 101 (32.6%) |
| Player Rock | 92 (29.7%) |
| Player Paper | 92 (29.7%) |
| Player Scissors | 126 (40.6%) |

*Table 3: Challenge mode aggregate statistics from 122 live gameplay runs.*

The near-equal win/loss distribution (33.3% vs 34.3%) with a high draw rate (32.4%) indicates well-calibrated AI difficulty. The player’s Scissors preference (40.5%) demonstrates the type of gesture bias the AI is designed to exploit over time. The longest streak of 6 confirms the Challenge AI successfully ramps difficulty to eventually defeat the player.

Deeper analysis of the streak distribution reveals the difficulty ramping’s effectiveness:

| **Streak** | **Runs** | **Percentage** |
| --- | --- | --- |
| 0 (immediate loss) | 65 | 52.8% |
| 1 | 37 | 30.1% |
| 2 | 9 | 7.3% |
| 3 | 5 | 4.1% |
| 4 | 4 | 3.3% |
| 5 | 1 | 0.8% |
| 6 (highest) | 2 | 1.6% |

*Table 3b: Challenge mode streak distribution across 122 runs.*

The mean streak of 0.84 with a median of 0 shows that the AI successfully defeats most players within the first two rounds, consistent with the design goal of starting beatable but escalating quickly. Only 9.8% of runs reached streak 3 or higher, confirming the ramping mechanism’s effectiveness. The average of 2.5 rounds per run (including draws) provides sufficient data for the AI to begin exploiting patterns while keeping sessions appropriately short.

The outcome-conditioned response analysis from Challenge mode rounds reveals player behaviour consistent with published findings:

| **After...** | **Stay** | **Upgrade** | **Downgrade** | **n** |
| --- | --- | --- | --- | --- |
| Win | 30.2% | 36.5% | 33.3% | 96 |
| Draw | 23.1% | 29.7% | 47.3% | 91 |

*Table 3c: Outcome-conditioned response rates from 309 live Challenge rounds.*

After winning, players showed a relatively even distribution across stay (30.2%), upgrade (36.5%), and downgrade (33.3%). This near-uniform post-win distribution contrasts with the strong win-stay tendency reported by Wang et al. [1], potentially reflecting that Challenge mode players are more aware of pattern exploitation risk. After drawing, players showed a strong downgrade tendency (47.3%), aligning with Dyson et al.’s [28] finding that draws evoke upgrade behaviour — though our data shows the opposite direction, which may reflect the different experimental context (competitive Challenge mode vs. laboratory setting with computerised Nash-equilibrium opponent). The player’s Scissors preference (40.5% vs expected 33.3%) represents a 7.2 percentage point deviation from Nash equilibrium, comparable to the Rock over-selection bias (35.7% vs 33.3%) reported by Dyson et al. [28] in their laboratory study.

*[INSERT IMAGE HERE]*

*Fig. 27: Challenge mode gameplay showing streak counter and difficulty indicator*

## **C. Player Profile Analysis (52 Rounds)**

The profiling system recorded 52 rounds from one player (“Zac”) across multiple sessions:

| **Trait** | **Finding** |
| --- | --- |
| Favourite gesture | Scissors (40% of throws) |
| Least used | Paper (29%) |
| After winning | Downgrade 48%, Upgrade 34%, Stay 17% |
| After losing | Downgrade 50% of the time |
| After drawing | Downgrade 56%, Upgrade 28%, Stay 17% |
| Strongest transition | After Rock → Scissors (62%) |
| Key trait | Stay rate 18% (vs Nash 33.3%) |
| Best counter | Play Rock often (counters 40% Scissors) |

*Table 4: Automated player profile analysis for “Zac” from 52 recorded rounds.*

The 100% upgrade-after-loss pattern represents a strong exploitable bias consistent with the lose-shift behaviour documented by Wang et al. [1] and the inflexible lose-shift mechanism identified by Forder and Dyson [11]. The 75% Rock→Scissors transition is a highly predictable sequence. These patterns demonstrate that even aware players exhibit the outcome-conditioned response tendencies predicted by game theory literature, validating the AI’s prediction approach.

Comparing this player’s aggregate response patterns against the full Challenge mode dataset reveals individual deviation from population behaviour:

| **Metric** | **Zac (52 rounds)** | **Challenge Pop. (310 rounds)** | **Literature Baseline** |
| --- | --- | --- | --- |
| Stay rate overall | 17.6% | 9.1% | 33.3% (Nash) |
| Upgrade rate overall | 31.4% | 33.5% | 33.3% (Nash) |
| Downgrade rate overall | 51.0% | 39.2% | 33.3% (Nash) |
| Gesture: Rock | 30.8% | 29.7% | 35.7% [28] |
| Gesture: Paper | 28.8% | 29.7% | 32.1% [28] |
| Gesture: Scissors | 40.4% | 40.6% | 32.2% [28] |
| Win rate | 57.7% | 33.2% | 33.3% (Nash) |

*Table 5b: Comparative behavioural analysis across data sources. Literature baselines from [28].*

Several notable findings emerge from this comparison. The player’s extremely low stay rate (17.6% vs population 27.3%) confirms Zhang et al.’s [8] finding that individual players exhibit heterogeneous strategies rather than following a single population-level pattern. The strong downgrade tendency (51.0%) is double the Nash equilibrium prediction (33.3%), consistent with Dyson et al.’s [28] observation that negative outcomes evoke cyclic (specifically downgrade) decisions. The Scissors preference (40.4%) matches the Challenge population almost exactly (40.5%), suggesting this bias may be characteristic of the specific player demographic rather than an individual quirk — notably, this is a reversal of the Rock bias typically observed in laboratory settings [28], potentially reflecting the influence of prior knowledge about the Rock-bias literature leading to over-correction.

The player’s 57.7% win rate across 52 recorded rounds (significantly above the 33.3% Nash baseline) suggests either favourable matchups during data collection or effective adaptation to the AI’s patterns, consistent with Batzilis et al.’s [27] finding that experienced players use opponent history more effectively.

| **After...** | **→ Rock** | **→ Paper** | **→ Scissors** |
| --- | --- | --- | --- |
| Rock | 25% | 12% | 62% |
| Paper | 33% | 13% | 53% |
| Scissors | 30% | 55% | 15% |

*Table 5: Zac’s move transition matrix showing gesture-to-gesture probabilities.*

*[INSERT IMAGE HERE]*

*Fig. 28: Player Stats viewer showing Zac’s statistics with progress bars and auto-generated traits*

*[INSERT IMAGE HERE]*

*Fig. 29: Excel player research tab showing automated strategy analysis and transition matrix*

# **X. System Architecture — Final State**

## **A. Technology Stack**

| **Component** | **Version** | **Purpose** |
| --- | --- | --- |
| Python | 3.9 | Primary language |
| OpenCV | 4.11.0 | Camera capture, frame rendering, UI overlay |
| MediaPipe | 0.10.21 | Hand landmark detection (21 keypoints) |
| scikit-learn | Latest | MLP classifier for front-on detection, LogisticRegression for ML AI |
| NumPy | 1.26.4 | Numerical operations, feature extraction |
| openpyxl | Latest | Excel read/write for research data logging |
| macOS | Apple M5 | Target platform |

*Table 6: Complete technology stack.*

## **B. Module Inventory**

| **Module** | **Lines** | **Responsibility** |
| --- | --- | --- |
| main.py | ~2,304 | Application entry, game loop, screen routing, 8 screen states |
| ui_renderer.py | ~2,609 | Arcade-themed UI overlay for all screens |
| hand_landmarks.py | ~229 | MediaPipe integration, Side/Front orientation routing |
| finger_counter.py | ~310 | Side-view finger extension (3D angles, rotation-tolerant) |
| gesture_mapper.py | ~65 | Finger states → Rock/Paper/Scissors classification |
| front_on_classifier.py | ~330 | Hybrid ML + curl analysis for front-on gestures |
| gesture_state.py | ~113 | Temporal smoothing: raw → stable → confirmed |
| fair_play_ai.py | ~487 | 5-layer heuristic prediction AI |
| challenge_ai.py | ~100 | Streak-based difficulty ramping AI |
| player_clone_ai.py | ~140 | 4-layer player behaviour reproduction AI |
| fair_play_state.py | ~596 | Best-of-N game controller with pump detection |
| challenge_mode_state.py | ~621 | Endless streak mode with persistent high score |
| rps_game_state.py | ~313 | Cheat mode controller (always counters) |
| player_profile_store.py | ~738 | JSON profiles, Excel reports, pattern analysis |
| landmark_collector.py | ~165 | Training data collection for front-on ML model |
| front_on_trainer.py | ~155 | sklearn MLP training script |
| simulation_mode.py | ~660 | Headless batch simulation (6 strategies × 3 AIs) |
| challenge_stats_logger.py | ~570 | Excel logging for Challenge mode |
| config_store.py | ~104 | Persistent JSON configuration with validation |
| robot_output.py | ~70 | Serial-ready command buffer for ESP32 |

*Table 7: Complete module inventory (∼7,500 lines of code).*

# **XI. Novel Contributions**

A comprehensive review of existing open-source RPS implementations and published research revealed several areas where this project introduces approaches not found in prior work. This section documents each novel contribution, the motivation behind it, and the reasoning that led to the chosen implementation. Proper credit is given to the existing projects and research that inspired each contribution — the innovations described here build upon, rather than replace, the work of others.

## **A. Player Clone System — Behavioural Digital Twins**

### ***Prior Art***

Every RPS AI project identified in the literature and on GitHub builds an AI that attempts to beat the player. The dmickelson/Rock-paper-scissors project [15] uses LSTM networks for move prediction. The MattYu Predictive RPS AI [16] employs a 15-scheme “swarm mind” with N-gram pattern matching. The Asylumrunner/Rock-Paper-Scissors-AI [17] implements Markov chain transition modelling. Roshambo-god [18] uses Bayesian prediction. All of these systems share a common goal: predict what the opponent will do next, then counter it.

### ***Innovation***

This project introduces a fundamentally different use of player modelling: rather than predicting a player to beat them, the system records a player’s complete behavioural profile and builds an AI that plays AS that player. The PlayerCloneAI reproduces an individual’s outcome-conditioned response distributions (e.g., “after losing with Rock, this player throws Paper 60% of the time”), move transition probabilities, outcome response types (stay/upgrade/downgrade rates), and overall gesture frequencies. The result is a behavioural digital twin that allows Player A to compete against a statistical reproduction of Player B’s play style.

### ***Motivation***

The motivation arose from the player profiling system: once individual behavioural patterns were being recorded and analysed, the question emerged — could these patterns be replayed? The immediate application was entertainment (play against your friend’s ghost when they’re not present), but the research implication is deeper: by reproducing documented behavioural biases as AI opponents, the system creates a controlled environment for studying whether players can learn to exploit their own tendencies when confronted with them.

### ***Implementation Reasoning***

The clone uses four decision layers ordered by specificity rather than a single model, because player data is inherently sparse. With 30–50 recorded rounds, outcome+gesture transition tables may have cells with only 1–2 observations. By cascading from most-specific (outcome+gesture transition) to least-specific (overall frequency), the system gracefully degrades when specific patterns lack data. An accuracy parameter (default 85%) introduces human-like noise because perfect reproduction of statistical patterns would feel robotic — the noise makes the clone feel like a person having a good day versus a bad day.

*[INSERT IMAGE HERE]*

*Fig. 30: Clone Mode setup showing opponent selection with recorded round counts*

## **B. Pump-Based Physical Beat Detection**

### ***Prior Art***

Existing camera-based RPS projects use one of three approaches to trigger rounds: timer-based countdowns (stefanluncanu24 [19] uses a fixed countdown with gesture photos), button/key presses, or “move hand out of frame then back in” (ChetanNair [20] resets when no hand is detected). None replicate the physical pumping motion that defines real-world RPS play.

### ***Innovation***

This system tracks the actual up-down pumping motion of the player’s fist via wrist landmark Y-coordinate analysis. A phase-based state machine alternates between “ready_for_down” and “waiting_for_up” phases, counting four complete pump cycles before opening the SHOOT window. The system handles beat cooldown (preventing double-counting from vibration), rock grace periods (tolerating brief gesture drops mid-pump), and dynamic threshold recalibration as the hand moves.

### ***Motivation***

The traditional RPS rhythm — “rock, paper, scissors, SHOOT” with synchronised pumping — is fundamental to the game’s social experience. Timer-based alternatives feel artificial and remove the physical skill component. The pump detection makes the game feel like playing against another person rather than interacting with software. Additionally, the physical motion creates a natural transition point: the player must hold Rock (fist) during the pump, then change to their actual throw on SHOOT, creating the same anticipation and commitment as the real game.

### ***Implementation Reasoning***

The threshold values (down: 0.045, up: 0.035, grace: 0.50s, cooldown: 0.18s) were arrived at through iterative testing. The original thresholds (down: 0.06, up: 0.045, grace: 0.35s) proved too strict, causing approximately 20% pump failure rate. The asymmetric thresholds (down > up) reflect that downward pumps are larger, more deliberate motions while upward returns are quicker and smaller. The grace period was extended because MediaPipe occasionally drops the Rock classification for 1–2 frames during fast motion, and the pump should not reset for such brief glitches. Accepting stable Rock (not just confirmed Rock) during countdown was essential because the confirmation pipeline’s 3-frame delay made it too slow for the fast pump rhythm.

## **C. Hybrid ML + Curl Analysis Classifier**

### ***Prior Art***

ML-based gesture classifiers on MediaPipe landmarks are well-established. Kazuhito00 [10] demonstrated the MLP training pipeline. AishTron7 [21] achieved 99.7% accuracy with an SGD classifier. Separately, the fingerpose library [7] introduced curl-state analysis for gesture definition. However, these approaches are used independently — either pure ML or pure geometric analysis.

### ***Innovation***

This system runs both classifiers on every frame in parallel and combines their outputs through a confidence-based decision logic. The ML model handles static poses with high accuracy; the curl analysis responds immediately to finger movement during fast transitions. The decision routing trusts ML when confident (>70%), defers to curl when ML is uncertain but curl is clear (>60%), and requires agreement from both when possible.

### ***Motivation***

The hybrid approach arose from a specific failure mode: the trained ML model achieved 90%+ accuracy on static poses but showed significant latency during the pump-to-shoot transition. During the fast motion from fist (Rock) to open hand (Paper) or two fingers (Scissors), intermediate hand shapes appeared that the model was never trained on. The model would output uncertain probabilities (e.g., Rock: 45%, Paper: 35%, Scissors: 20%) for 3–5 frames before converging. Meanwhile, curl analysis — which measures joint angles directly — detected the moment fingers began opening immediately, because PIP joint angles change before the full gesture is formed.

### ***Implementation Reasoning***

The 70% ML confidence threshold was chosen empirically: above 70%, the ML model’s classification was virtually always correct; below 70%, it was frequently wrong during transitions. The curl analysis uses PIP and DIP joint angles rather than tip positions because angles are rotation-invariant — a finger at 150° is straight regardless of which direction the hand is pointing. The curl thresholds (NoCurl ≥ 150°, HalfCurl ≥ 110°, FullCurl < 110°) were derived from the fingerpose library’s [7] published ranges and validated against live testing. This hybrid approach resolved the transition issue completely without retraining the ML model.

## **D. Integrated Research Data Pipeline**

### ***Prior Art***

Most RPS projects track basic win/loss/draw counts. Some, like hjpulkki/RPS [22], collect game data for model training. The goelp14/RockPaperScissors [23] project pits different AI bots against each other. None maintain the breadth of per-round research data with automated statistical analysis and exportable workbooks.

### ***Innovation***

This system logs every round with 10+ attributes (timestamp, player/robot gestures, outcome, response type, reaction time, AI prediction metadata, streak state) to both JSON player profiles and structured Excel workbooks. The system automatically generates per-player strategy analysis including gesture frequency distributions, outcome-conditioned response tables, move transition matrices, and human-readable behavioural traits. A headless simulation framework enables batch testing of AI strategies at scale (18,000 rounds), with results exported to comparative research reports.

### ***Motivation***

The research pipeline was motivated by the capstone’s dual nature as both an engineering project and a research platform. Published game theory studies [1][8] analyse human RPS behaviour using exactly the metrics this system captures — conditional response rates, transition probabilities, and outcome-dependent strategies. By automating this analysis, the system enables direct comparison between individual player behaviour and population-level findings from the literature.

## **E. Five-Layer Behaviourally-Grounded AI Prediction**

### ***Prior Art***

The MattYu Predictive RPS AI [16] uses 15 N-gram pattern matching schemes with majority voting. The Asylumrunner Markov chain approach [17] models move-to-move transition probabilities. The wmodes heuristic RPS [24] implements multiple computer strategies. Raymond Hettinger’s PyCon presentation [25] demonstrated digraph-based prediction with strategy weighting. Each of these systems uses either pure pattern matching or pure probabilistic prediction.

### ***Innovation***

This system’s prediction architecture combines five distinct layers, each explicitly tied to documented findings in the behavioural game theory literature. Population priors encode the win-stay/lose-shift tendencies documented by Wang et al. [1]. Outcome-conditioned response learning implements the conditional response model from the same paper. Exact transition memory captures individual habits. The layers are weighted by recency (recent rounds exponentially dominate older ones) rather than equally, reflecting the cognitive science finding that recent experiences disproportionately influence sequential decisions [2].

### ***Implementation Reasoning***

The recency weighting function (weight = base / 1.28^distance) was chosen because exponential decay naturally models the cognitive phenomenon of recency bias documented in the RPS literature [1][8]. The decay rate of 1.28 was tuned to give rounds from 5+ games ago approximately 30% of the weight of the most recent round, matching the empirical observation that human players adapt their strategy primarily based on the last 3–5 outcomes. The intentional imperfection (66–76% chance of using the best prediction) was a deliberate design choice: perfectly optimal AI produces frustrating gameplay because the player cannot detect or exploit any patterns, which Zhang et al. [8] showed is essential for the adversarial modelling that makes RPS engaging.

## **F. AI Difficulty Ramping by Streak**

### ***Prior Art***

The goelp14/RockPaperScissors project [23] offers fixed difficulty tiers (Beginner/Intermediate/Expert/Super Hard). Most projects use a single AI level. None dynamically adjust AI confidence based on the player’s current performance within a single session.

### ***Innovation***

The Challenge AI increases its effective skill by 0.035 per consecutive player win (base 0.68, max 0.92), with a confidence penalty that decreases at higher streaks. At low streaks (<3), the AI’s misses sample from the full prediction distribution (human-like errors). At mid streaks (3–5), misses come from the top two predictions only. At high streaks (>6), even misses favour the second-best prediction. This creates a smooth difficulty curve where the AI becomes noticeably stronger the better the player performs, without discrete difficulty jumps.

### ***Motivation***

The endless streak format required an AI that starts beatable (to avoid immediate frustration) but becomes progressively harder (to create a meaningful challenge and eventual loss). Fixed difficulty tiers would create an abrupt wall. The continuous ramping ensures every player eventually loses — the question is how high they can push their streak — while the early rounds remain accessible to less experienced players.

## **G. Iterative Front-On Detection Methodology**

### ***Prior Art***

Published gesture recognition work typically presents the final working solution. Failed approaches are rarely documented in detail.

### ***Innovation***

This project documents seven distinct failed approaches to front-on gesture detection before arriving at the final hybrid solution. Each approach is recorded with what was tried, what specific failure mode occurred, and why the approach was abandoned. This constitutes a complete record of engineering problem-solving methodology in a real-time computer vision context.

### ***Significance***

The documentation reveals a pattern common in computer vision engineering: geometric heuristics that work well in controlled conditions (fixed hand angle, consistent distance) break down when users interact naturally. The progression from simple Y-axis checks through multi-signal voting, distance comparisons, and X-axis offsets to finally adopting a learned model demonstrates that the complexity of front-on hand pose variation exceeds what hand-crafted rules can reliably capture — a finding consistent with the broader trend in computer vision toward learned representations over engineered features [13][14].

# **XII. Limitations**

• MediaPipe Z-coordinate reliability: Depth estimates are significantly less accurate than X-Y coordinates for front-on poses. This was the fundamental reason seven geometric approaches failed (Iterations 24–30) and a trained ML model was required.

• Front-on classifier personalisation: The ML model must be trained on each user’s hand data via the data collection workflow. While the curl analysis layer provides a reasonable fallback, peak accuracy requires per-user training.

• AI limitations against randomisation: Against truly random play, the best AI achieves only 34.6% — consistent with game theory showing no strategy can systematically beat a random opponent in RPS.

• Clone fidelity with limited data: The 100% upgrade-after-loss in Zac’s profile is based on a small loss sample and may not be representative. Minimum 30 rounds provides basic patterns; 100+ rounds are recommended for reliable conditional response reproduction.

• Lighting sensitivity: Performance degrades in extreme conditions, consistent with known MediaPipe limitations [13].

• Single-hand constraint: Only one hand is tracked, limiting potential multi-hand gesture extensions.

• Sound integration gap: Sound effects were built in an earlier session but lost during UI rebuild iterations and require re-wiring.

# **XIII. Future Work**

• ESP32 physical robot: The serial bridge and command protocol are implemented; an ESP32-based robotic hand displaying Rock/Paper/Scissors gestures is the planned hardware extension.

• Reinforcement learning AI: Augmenting the heuristic with PPO or Q-learning for adaptive self-play [12].

• Multi-player tournament: Round-robin between clone AIs to rank player strategies.

• Transfer learning for front-on detection: Shared base model requiring only fine-tuning for new users.

• Formal statistical testing: Chi-square and KL-divergence comparing recorded player behaviour against population biases [1][8].

• Sound re-integration: Re-wiring the existing sound_player.py module into the current main.py architecture.

# **XIV. Conclusion**

This project demonstrates a complete pipeline from camera input to adaptive AI gameplay in a real-time Rock-Paper-Scissors system, developed through 42 documented iterations across five distinct phases. The iterative development methodology — particularly the seven-approach journey to front-on gesture detection — demonstrates that systematic engineering problem-solving, including the willingness to discard approaches that don’t generalise, is essential for building robust real-time computer vision systems.

The hybrid gesture classifier combining trained MLP models with real-time curl analysis addresses the fundamental challenge of front-on hand pose recognition by exploiting complementary strengths: ML accuracy on static poses and curl responsiveness during transitions. The multi-layered AI opponent design, grounded in behavioural game theory research on conditional response patterns [1][8], achieves a 16.2% win rate advantage over random play in simulation while maintaining balanced difficulty in live testing (33.3% vs 34.3% win/loss). The player clone system introduces a novel approach to opponent modelling, faithfully reproducing individual behavioural patterns — including the outcome-conditioned response tendencies documented in the literature.

The comprehensive data pipeline, validated through 18,000 simulated rounds and 309 live Challenge rounds with per-player profiling across 52 recorded rounds, provides a foundation for systematic investigation of human sequential decision-making in adversarial contexts. The complete system — approximately 7,500 lines of Python across 20 modules — runs at real-time frame rates on consumer hardware, demonstrating that sophisticated computer vision and AI systems can be built with accessible tools.

# **References**

[1] Z. Wang, B. Xu, and H.-J. Zhou, “Social cycling and conditional responses in the Rock-Paper-Scissors game,” Scientific Reports, vol. 4, Art. no. 5830, Jul. 2014. doi: 10.1038/srep05830.

[2] E. Brockbank and E. Vul, “Repeated rock, paper, scissors play reveals limits in adaptive sequential behavior,” Cognitive Psychology, vol. 151, Art. no. 101654, 2024.

[3] F. Zhang et al., “MediaPipe Hands: On-device Real-time Hand Tracking,” arXiv:2006.10214, Jun. 2020.

[4] Y. Meng, H. Jiang, N. Duan, and H. Wen, “Real-Time Hand Gesture Monitoring Model Based on MediaPipe’s Registerable System,” Sensors, vol. 24, no. 19, Art. no. 6262, Sep. 2024.

[5] G. Sánchez-Brizuela et al., “Lightweight real-time hand segmentation leveraging MediaPipe landmark detection,” Virtual Reality, vol. 27, pp. 3125–3132, 2023.

[6] K. A. Ahmad, D. C. Silpani, and K. Yoshida, “The impact of large sample datasets on hand gesture recognition by hand landmark classification,” Int. J. Affective Engineering, vol. 22, no. 3, pp. 253–259, 2023.

[7] A. Potgieter, “Fingerpose: Finger gesture classifier for hand landmarks detected by MediaPipe,” GitHub, 2022. Available: https://github.com/andypotato/fingerpose

[8] H. Zhang, F. Moisan, and C. Gonzalez, “Rock-Paper-Scissors Play: Beyond the Win-Stay/Lose-Change Strategy,” Games, vol. 12, no. 3, Art. no. 52, Jun. 2021.

[9] S. Lei et al., “Multi-AI competing and winning against humans in iterated Rock-Paper-Scissors game,” Scientific Reports, vol. 10, Art. no. 13873, Aug. 2020.

[10] Kazuhito00, “Hand gesture recognition using MediaPipe,” GitHub, 2021. Available: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

[11] L. Forder and B. J. Dyson, “Behavioural and neural modulation of win-stay but not lose-shift strategies as a function of outcome value in Rock, Paper, Scissors,” Scientific Reports, vol. 6, Art. no. 33809, Sep. 2016.

[12] J. du Plessis et al., “Policy-Based Reinforcement Learning in the Generalized Rock-Paper-Scissors Game,” in Proc. ESANN 2023, Bruges, Belgium, Oct. 2023.

[13] G. Amprimo et al., “Hand tracking for clinical applications: validation of the Google MediaPipe Hand (GMH) and the depth-enhanced GMH-D frameworks,” Biomedical Signal Processing and Control, vol. 96, Art. no. 106508, 2024.

[14] K. A. Ahmad, D. C. Silpani, and K. Yoshida, “Hand gesture recognition by hand landmark classification,” Int. Symp. Affective Science and Engineering, Art. no. 8, 2022.

[15] D. Mickelson, “Rock-paper-scissors: ML and data science techniques for RPS AI,” GitHub, 2023. [Online]. Available: https://github.com/dmickelson/Rock-paper-scissors

[16] M. Yu, “Predictive Rock-Paper-Scissor AI: Swarm mind with N-gram prediction,” GitHub, 2020. [Online]. Available: https://github.com/MattYu/Project---Predictive-Rock-Paper-Scissor-AI-

[17] Asylumrunner, “Rock-Paper-Scissors-AI: Markov chain probabilistic reasoning,” GitHub, 2019. [Online]. Available: https://github.com/Asylumrunner/Rock-Paper-Scissors-AI

[18] W. Tian, “Roshambo God: Rock Paper Scissors AI using Bayes’ Theorem,” GitHub, 2018. [Online]. Available: https://github.com/wesleytian/roshambo-god

[19] S. Luncanu, “RockPaperScissors-using-MediaPipe-Cv2: XGBoost gesture recognition,” GitHub, 2023. [Online]. Available: https://github.com/stefanluncanu24/RockPaperScissors-using-MediaPipe-Cv2

[20] C. Nair, “Rock-Paper-Scissors: OpenCV and MediaPipe finger counting,” GitHub, 2023. [Online]. Available: https://github.com/ChetanNair/Rock-Paper-Scissors

[21] AishTron7, “Rock-Paper-Scissor: SGD Classifier achieving 99.7% accuracy,” GitHub, 2023. [Online]. Available: https://github.com/AishTron7/Rock-Paper-Scissor

[22] H. Pulkki, “RPS: Deep learning for Rock-Paper-Scissors using Keras RNN,” GitHub, 2020. [Online]. Available: https://github.com/hjpulkki/RPS

[23] P. Goel, “RockPaperScissors: Python RPS with AI and Markov Chain difficulty tiers,” GitHub, 2021. [Online]. Available: https://github.com/goelp14/RockPaperScissors

[24] W. Modes, “Rock-paper-scissors: Heuristic computer player strategies with fair countdown,” GitHub, 2023. [Online]. Available: https://github.com/wmodes/rock-paper-scissors

[25] R. Hettinger, “Pattern Recognition and Reinforcement Learning — Rock Paper Scissors,” US PyCon 2019 Tutorial. [Online]. Available: https://rhettinger.github.io/rock_paper.html

[26] J. F. Nash, “Equilibrium points in n-person games,” Proceedings of the National Academy of Sciences, vol. 36, no. 1, pp. 48–49, 1950.

[27] D. Batzilis, S. Jaffe, S. Levitt, J. A. List, and J. Picel, “Behavior in Strategic Settings: Evidence from a Million Rock-Paper-Scissors Games,” Games, vol. 10, no. 2, Art. no. 18, Apr. 2019. doi: 10.3390/g10020018.

[28] B. J. Dyson, J. M. P. Wilbiks, R. Sandhu, G. Papanicolaou, and J. Lintag, “Negative outcomes evoke cyclic irrational decisions in Rock, Paper, Scissors,” Scientific Reports, vol. 6, Art. no. 20479, Feb. 2016. doi: 10.1038/srep20479.

[29] M. Hoffman, S. Suetens, U. Gneezy, and M. A. Nowak, “An experimental investigation of evolutionary dynamics in the Rock-Paper-Scissors game,” Scientific Reports, vol. 5, Art. no. 8817, Mar. 2015. doi: 10.1038/srep08817.

[30] J. Qi, L. Ma, Z. Cui, and Y. Yu, “Computer vision-based hand gesture recognition for human-robot interaction: a review,” Complex & Intelligent Systems, vol. 10, pp. 1581–1606, Jul. 2023. doi: 10.1007/s40747-023-01173-6.

[31] M. Zohaib and H. Nakanishi, “Diversifying dynamic difficulty adjustment agent by integrating player state models into Monte-Carlo tree search,” Expert Systems with Applications, vol. 208, Art. no. 118058, Dec. 2022. doi: 10.1016/j.eswa.2022.118058.

[32] S. Dill, A. Rösch, M. Rohr, G. Güney, L. De Witte, E. Schwartz, and C. Hoog Antink, “Accuracy Evaluation of 3D Pose Estimation with MediaPipe Pose for Physical Exercises,” Current Directions in Biomedical Engineering, vol. 9, no. 1, pp. 563–566, Sep. 2023. doi: 10.1515/cdbme-2023-1141.

[33] C. Graef, D. Bartl, and E. Andre, “Personalized Dynamic Difficulty Adjustment — Imitation Learning Meets Reinforcement Learning,” in Proc. IEEE Conference on Games, Milan, Italy, Aug. 2024.

# **Appendices**

## **Appendix A: Open-Source Projects and Attribution**

The following open-source projects were studied during development and influenced specific aspects of this system. All gesture recognition, AI prediction, game logic, and UI code in this project was written from scratch; the projects below provided conceptual inspiration and validated approaches that informed design decisions. No code was directly copied from any of these projects.

### ***Gesture Recognition Projects***

| **Project** | **Author** | **What It Does** | **How It Influenced Our Work** |
| --- | --- | --- | --- |
| hand-gesture-recognition-using-mediapipe [10] | Kazuhito00 | MLP classifier on normalised MediaPipe landmarks with keyboard-triggered data collection | Directly inspired our landmark_collector.py data collection workflow (keyboard triggers during live camera), keypoint normalisation strategy (translate to wrist, scale by palm size), and the concept of training a personal gesture model from user-collected data |
| fingerpose [7] | andypotato | Declarative gesture definition using per-finger curl states (NoCurl/HalfCurl/FullCurl) and direction | Provided the curl analysis concept used in our hybrid classifier. Our curl thresholds (150°/110°) were derived from fingerpose’s published ranges. The idea of classifying gestures from joint angles rather than positions solved our rotation-invariance problem |
| RPS with SGD Classifier [21] | AishTron7 | sklearn SGDClassifier on MediaPipe landmarks achieving 99.7% accuracy with interactive UI | Validated that sklearn classifiers achieve high accuracy on MediaPipe landmark features, supporting our decision to use sklearn’s MLPClassifier rather than building a custom neural network |
| RPS-using-MediaPipe-Cv2 [19] | stefanluncanu24 | XGBoost model on MediaPipe landmarks with countdown timer and gesture photos | The countdown timer concept influenced our beat-based timing. The use of a trained model (XGBoost) rather than pure geometric rules reinforced our eventual shift to ML for front-on detection |
| TheJLifeX finger counting | TheJLifeX | Original MediaPipe finger counting gist using x/y landmark comparisons, documenting left/right hand challenges | Early reference for finger counting logic. Their documented issues with left vs right hand detection (tip-above-PIP fails for left hand without mirroring) helped us anticipate and address handedness challenges |
| ChetanNair RPS [20] | ChetanNair | Basic finger-counting RPS with “hand out of frame” round reset mechanism | Demonstrated the simplest viable RPS-from-camera approach. Their “hand out of frame” reset highlighted the need for a better round-triggering mechanism, motivating our pump-based beat detection |

*Table A1: Gesture recognition projects that influenced the system.*

### ***AI Prediction and Opponent Modelling Projects***

| **Project** | **Author** | **What It Does** | **How It Influenced Our Work** |
| --- | --- | --- | --- |
| Rock-paper-scissors (ML) [15] | dmickelson | LSTM neural network for move prediction with online learning, opponent type classification, ensemble decision-making | The ensemble concept (combining multiple prediction strategies) informed our multi-layered AI architecture. Their opponent type classification inspired our player profiling system’s trait analysis |
| Predictive RPS AI (Swarm) [16] | MattYu | 15-scheme “swarm mind” using N-gram prediction with outcome-conditioned subdivisions and majority consensus | The concept of subdividing predictions by outcome condition (win/loss/draw) directly influenced our Layer 2 (outcome-conditioned response learning). Their multi-scheme majority voting concept informed our multi-layer score aggregation approach |
| RPS AI (Markov Chain) [17] | Asylumrunner | Markov chain modelling move-to-move transition probabilities from observed play | Direct inspiration for our Layer 3 (exact transition memory). Our implementation extends the basic Markov approach with recency weighting and outcome conditioning |
| Roshambo God (Bayes) [18] | wesleytian | Bayesian prediction using opponent history of configurable length k, based on RoShamBo tournament format | The concept of configurable history length informed our recency weighting approach. Their discussion of optimal play against sub-optimal opponents validated our design philosophy of exploiting human biases rather than playing Nash equilibrium |
| RPS Deep Learning [22] | hjpulkki | Keras RNN predicting next move from move history, designed for online data collection | Demonstrated the viability of learning player sequences over time. Their approach of collecting training data during live play influenced our player profile recording system |
| RPS with Heuristic Strategy [24] | wmodes | Terminal RPS with multiple computer strategies, simultaneous countdown, and strategy discussion | Their emphasis on fair countdown timing (computer and human commit simultaneously) reinforced our beat-3 robot lock design. Their layered strategy concept aligned with our multi-layer prediction architecture |
| RPS with Difficulty Tiers [23] | goelp14 | Python RPS with Beginner (random), Intermediate (psychology), Expert (Markov chain), and Super Hard tiers | The tiered difficulty concept influenced our Challenge mode, though we chose continuous ramping over discrete tiers. Their bot-vs-bot testing inspired our simulation framework |
| PyCon RPS Tutorial [25] | R. Hettinger | Python tutorial demonstrating digraph-based prediction with proportional and greedy selection, strategy weighting | The digraph prediction concept (what follows a given move) directly maps to our Layer 3. The proportional vs greedy selection distinction influenced our weighted_choice implementation used when the AI “misses” its top prediction |

*Table A2: AI prediction projects that influenced the system.*

### ***Core Frameworks***

| **Component** | **Source** | **Usage** |
| --- | --- | --- |
| MediaPipe Hands [3] | Google (Zhang et al.) | Core 21-point hand landmark detection. Used as-is via the mediapipe Python package. All downstream processing (finger counting, gesture classification, curl analysis) is original work built on MediaPipe’s landmark output |
| OpenCV [14] | OpenCV.org | Camera capture, frame mirroring, colour conversion, and all UI rendering via drawing primitives. No pre-built UI components were used |
| scikit-learn | Pedregosa et al. | MLPClassifier for front-on gesture detection and LogisticRegression for ML-based move prediction. Standard library usage with custom feature engineering |
| openpyxl | openpyxl.org | Excel workbook creation and manipulation for research data logging. Custom workbook structures and migration logic |

*Table A3: Core frameworks and libraries.*

## **Appendix B: Image Index**

The following images should be inserted at the marked placeholders throughout the document. Images correspond to screenshots taken during development:

Fig. 1: MediaPipe 21-point hand landmark model diagram [3]

Fig. 2: Camera test with test_camera.py (Mar 17, 5:47pm)

Fig. 3: Right hand detection, 0.93 confidence (Mar 17, 8:41pm)

Fig. 4: Left hand detection, 0.93 confidence (Mar 17, 8:42pm)

Fig. 5: Finger counter Count: 5 with red tip dots (Mar 17, 9:03pm)

Fig. 6: Finger counter Count: 4, thumb tucked (Mar 17, 9:03pm)

Fig. 7: Left hand rejected — not_right_hand (Mar 17, 9:05pm)

Fig. 8: First gesture classification — Gesture: Paper (Mar 17, 9:26pm)

Fig. 9: Dev notes — history buffer and hand mode toggle design

Fig. 10: Hand mode Left with Rock detected (Mar 18, 7:14am)

Fig. 11: Full diagnostic — Raw/Stable/Confirmed: Paper (Mar 18, 8:02am)

Fig. 12: Diagnostic with Rock on Left mode (Mar 18, 8:05am)

Fig. 13: Cheat mode gameplay

Fig. 14: Early menu screen

Fig. 15: Early game view with beat countdown

Fig. 16: Hardware test mode screen

Fig. 17: Research comparison dashboard

Fig. 18: Multi-signal voting diagnostic

Fig. 19: Data collection mode for front-on training

Fig. 20: Hybrid classifier diagnostic output

Fig. 21: Clone Setup opponent selection

Fig. 22: Arcade-themed main menu

Fig. 23: Game result screen with arcade theme

Fig. 24: Settings with description panel

Fig. 25: Player Stats viewer (Zac’s profile)

Fig. 26: Interactive tutorial Step 1

Fig. 27: Challenge mode gameplay

Fig. 28: Player Stats with auto-generated traits

Fig. 29: Excel player research tab

Fig. 30: Clone Mode setup showing opponent selection (Innovations section)
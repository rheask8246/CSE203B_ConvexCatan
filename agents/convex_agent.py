"""
ConvexAgent v5 — Fairness-first with early-game development and 2-step lookahead.

- Early game (VP < 3): prioritize production to reach 4-6 VP
- Mid game: enforce fairness (alpha = min(1.0, vp/4))
- Node blocking: prevent leader from taking high-impact nodes (BLOCK_WEIGHT=1.5)
- Impact weight: critical nodes weighted by dice probability (IMPACT_WEIGHT=0.3)
- Robber: block hex with largest total production
- Roads: max fairness of reachable nodes
- Trade: maximize hand entropy, penalize duplicate stacks
- 2-step lookahead for settlement placement
"""

from catanatron.models.player import Player
from catanatron.models.enums import ActionType, ActionPrompt

from agents.fairness_scoring import (
    _production_matrix,
    _production_matrix_with_robber,
    W,
    fairness_node_scores,
    fairness_road_scores,
    robber_hex_scores,
    entropy_trade_suggestion,
    city_fairness_score,
    _reachable_from_road,
    _our_vp,
)
from catanatron.models.board import get_edges


class ConvexAgent(Player):
    """Fairness-first agent: minimize global inequality, early game builds economy."""

    def __init__(self, color, lambda_value=None, mu_value=None,
                 self_floor=None, is_bot=True):
        super().__init__(color, is_bot=is_bot)
        self.lambda_value = lambda_value
        self.mu_value = mu_value
        self.self_floor = self_floor

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        state = game.state

        if state.is_initial_build_phase:
            return self._initial(actions, game)

        if state.current_prompt != ActionPrompt.PLAY_TURN:
            robber = [a for a in actions if a.action_type == ActionType.MOVE_ROBBER]
            if robber:
                return self._best_robber_action(robber, game)
            discard = [a for a in actions if a.action_type == ActionType.DISCARD]
            if discard:
                return self._best_discard_action(discard, game)
            return actions[0]

        for a in actions:
            if a.action_type == ActionType.ROLL:
                return a

        build_action = self._best_build_decision(actions, game)
        if build_action is not None:
            return build_action

        trade = self._best_trade_action(actions, game)
        if trade is not None:
            return trade

        for a in actions:
            if a.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                return a

        for a in actions:
            if a.action_type == ActionType.END_TURN:
                return a

        return actions[0]

    def _initial(self, actions, game):
        A = _production_matrix(game)
        vp = _our_vp(game, self.color)
        settle = [a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        road = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]

        node_scores = fairness_node_scores(game, self.color, A, W, vp, use_lookahead=True)

        if settle:
            return max(settle, key=lambda a: float(node_scores[a.value]))

        if road:
            reachable = {
                tuple(sorted([a.value[0], a.value[1]])): _reachable_from_road(game, a.value[0], a.value[1])
                for a in road
            }
            road_scores = fairness_road_scores(game, self.color, A, W, vp, reachable)
            return max(road, key=lambda a: road_scores.get(tuple(sorted([a.value[0], a.value[1]])), float('-inf')))

        return actions[0]

    def _best_robber_action(self, robber_actions, game):
        A = _production_matrix(game)
        scores = robber_hex_scores(game, A, W)
        return max(robber_actions, key=lambda a: scores.get(a.value[0], 0.0))

    def _best_discard_action(self, discard_actions, game):
        from catanatron.models.enums import RESOURCES
        import numpy as np
        state = game.state
        try:
            hand = state.hands.get(self.color, {})
            counts = np.array([hand.get(r, 0) for r in RESOURCES], dtype=float)
        except Exception:
            return discard_actions[0]

        def entropy_after(a):
            try:
                idx = list(RESOURCES).index(a.value)
                c = counts.copy()
                c[idx] = max(0.0, c[idx] - 1)
                total = c.sum()
                if total <= 0:
                    return 0.0
                p = c / total
                p = p[p > 0]
                return float(-np.sum(p * np.log(p)))
            except Exception:
                return 0.0

        return max(discard_actions, key=entropy_after)

    def _best_build_decision(self, actions, game):
        A = _production_matrix_with_robber(game)
        vp = _our_vp(game, self.color)
        node_scores = fairness_node_scores(game, self.color, A, W, vp, use_lookahead=True)

        settlements = [a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        cities = [a for a in actions if a.action_type == ActionType.BUILD_CITY]
        roads = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]

        candidates = []

        for a in settlements:
            n = a.value
            s = float(node_scores[n]) if 0 <= n < len(node_scores) else float('-inf')
            candidates.append((s, a))

        for a in cities:
            s = city_fairness_score(game, self.color, a.value, A, W)
            candidates.append((s, a))

        for a in roads:
            u, v = a.value[0], a.value[1]
            reachable = {tuple(sorted([u, v])): _reachable_from_road(game, u, v)}
            road_scores = fairness_road_scores(game, self.color, A, W, vp, reachable)
            s = road_scores.get(tuple(sorted([u, v])), float('-inf'))
            candidates.append((s, a))

        if not candidates:
            return None

        priority = {
            ActionType.BUILD_SETTLEMENT: 0,
            ActionType.BUILD_CITY: 1,
            ActionType.BUILD_ROAD: 2,
        }
        best_score, best_action = max(
            candidates,
            key=lambda x: (x[0], -priority.get(x[1].action_type, 9)),
        )
        return best_action

    def _best_trade_action(self, actions, game):
        trade_actions = [a for a in actions if a.action_type == ActionType.MARITIME_TRADE]
        if not trade_actions:
            return None

        suggestion = entropy_trade_suggestion(game, self.color, duplicate_penalty=0.5)
        if suggestion is None:
            return None

        give_res, recv_res = suggestion

        for a in trade_actions:
            try:
                v = list(a.value) if hasattr(a.value, "__iter__") else []
                if len(v) >= 2 and v[0] == give_res and v[-1] == recv_res:
                    return a
            except Exception:
                pass

        for a in trade_actions:
            try:
                v = list(a.value) if hasattr(a.value, "__iter__") else []
                if len(v) >= 1 and v[0] == give_res:
                    return a
            except Exception:
                pass

        return None

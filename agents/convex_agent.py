"""
ConvexAgent — Fair resource allocation via convex optimization (LP).

Uses the convex solver (CVXPY) for:
- Initial placement: fair allocation LP (solve_initial)
- Mid-game builds: same LP with remaining budgets (solve_build + score_action)
- Robber: block hex contributing most to leader's production
- Trade: LP to maximize total resources + hand balance (solve_trade_and_build)

Falls back to fairness_gain_scores when the LP is infeasible.
"""

from catanatron.models.player import Player
from catanatron.models.enums import ActionType, ActionPrompt

from agents.convex_solver import (
    solve_initial,
    solve_build,
    score_action,
    score_robber_hexes_fairness,
    get_best_maritime_trade,
)


class ConvexAgent(Player):
    """Fair resource allocation agent using convex optimization (LP)."""

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
        """Initial placement: LP-based fair allocation scores."""
        settle = [a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        road = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]

        scores = solve_initial(
            game, self.color,
            lambda_value=self.lambda_value,
            mu_value=self.mu_value,
            self_floor=self.self_floor,
        )

        if settle:
            return max(settle, key=lambda a: float(scores[a.value]))

        if road:
            return max(
                road,
                key=lambda a: (float(scores[a.value[0]]) + float(scores[a.value[1]])) / 2,
            )

        return actions[0]

    def _best_robber_action(self, robber_actions, game):
        """Block the hex contributing most to the leader's production."""
        scores = score_robber_hexes_fairness(game)
        return max(robber_actions, key=lambda a: scores.get(a.value[0], 0.0))

    def _best_discard_action(self, discard_actions, game):
        """Discard to maximize hand entropy (balanced resource distribution)."""
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
        """Mid-game builds: LP-based node/edge scores + score_action."""
        build = [
            a for a in actions
            if a.action_type in (
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD,
            )
        ]
        if not build:
            return None

        node_scores, edge_scores, edge_to_idx = solve_build(
            game, self.color,
            lambda_value=self.lambda_value,
            mu_value=self.mu_value,
            self_floor=self.self_floor,
        )
        return max(
            build,
            key=lambda a: score_action(a, node_scores, edge_scores, edge_to_idx),
        )

    def _best_trade_action(self, actions, game):
        """Maritime trade: LP maximizes total resources + hand balance."""
        trade_actions = [a for a in actions if a.action_type == ActionType.MARITIME_TRADE]
        if not trade_actions:
            return None

        suggestion = get_best_maritime_trade(game, self.color)
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

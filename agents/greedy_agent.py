from collections import Counter
from catanatron import Player
from catanatron.models.enums import ActionType

#score(l) = α · pip_score(l) + β · diversification_bonus(l)

PIP_WEIGHTS = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}

ALPHA = 1.0
BETA = 0.6

def get_catan_map(game):
    """
    Robustly locate the map object across catanatron versions.
    We need an object with `.adjacent_tiles` mapping (node_id -> list of tiles).
    """
    #Have seen the map under various names in different versions, so try multiple candidates.
    candidates = [
        getattr(game, "catan_map", None),
        getattr(game, "board", None),
        getattr(getattr(game, "state", None), "board", None),
        getattr(getattr(getattr(game, "state", None), "board", None), "map", None),
        getattr(getattr(getattr(game, "state", None), "board", None), "catan_map", None),
    ]

    for obj in candidates:
        if obj is None:
            continue
        #board and map objects I've seen both have adjacent_tiles, so check for that directly
        if hasattr(obj, "adjacent_tiles"):
            return obj
        if hasattr(obj, "map") and hasattr(obj.map, "adjacent_tiles"):
            return obj.map

    raise AttributeError(
        "Could not find map object with `.adjacent_tiles`. "
        "Inspect `dir(game)`, `dir(game.state)`, and `dir(game.state.board)`."
    )

def pip_score(catan_map, node_id):
    score = 0
    for tile in catan_map.adjacent_tiles.get(node_id, []):
        if tile.number in PIP_WEIGHTS:
            score += PIP_WEIGHTS[tile.number]
    return score


def diversification_bonus(catan_map, node_id, portfolio):
    bonus = 0
    for tile in catan_map.adjacent_tiles.get(node_id, []):
        if tile.resource:
            bonus += 1 / (1 + portfolio.get(tile.resource, 0))
    return bonus


def score_node(catan_map, node_id, portfolio):
    return ALPHA * pip_score(catan_map, node_id) + \
           BETA * diversification_bonus(catan_map, node_id, portfolio)


class GreedyAgent(Player):
    """
    Greedy baseline:
    - Settlement: highest pip + diversification
    - City: upgrade highest pip node
    - Road: choose road whose endpoint has best node score
    - Otherwise: roll or end turn
    """

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        catan_map = get_catan_map(game)

        #Roll
        for a in actions:
            if a.action_type == ActionType.ROLL:
                return a

        #Build city
        city_actions = [a for a in actions if a.action_type == ActionType.BUILD_CITY]
        if city_actions:
            return max(city_actions, key=lambda a: pip_score(catan_map, a.value))

        #Build settlement
        settlement_actions = [a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        if settlement_actions:
            portfolio = self._portfolio(game)
            return max(settlement_actions, key=lambda a: score_node(catan_map, a.value, portfolio))

        #Build road
        road_actions = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]
        if road_actions:
            portfolio = self._portfolio(game)
            best = road_actions[0]
            best_score = -1
            for a in road_actions:
                try:
                    n1, n2 = a.value
                    s = max(score_node(catan_map, n1, portfolio),
                            score_node(catan_map, n2, portfolio))
                except Exception:
                    s = 0
                if s > best_score:
                    best_score = s
                    best = a
            return best

        #End turn
        for a in actions:
            if a.action_type == ActionType.END_TURN:
                return a

        return actions[0]

    def _portfolio(self, game):
        """
        Count resources adjacent to this player's existing settlements/cities.
        """
        state = game.state
        catan_map = get_catan_map(game)
        portfolio = Counter()

        buildings = state.buildings_by_color[self.color]
        settlement_nodes = buildings.get("SETTLEMENT", [])
        city_nodes = buildings.get("CITY", [])

        for node in settlement_nodes + city_nodes:
            for tile in catan_map.adjacent_tiles.get(node, []):
                if tile.resource:
                    portfolio[tile.resource] += 1

        return portfolio
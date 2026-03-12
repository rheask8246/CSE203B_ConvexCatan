"""
Fair resource allocation LP for Catan.

Implements the primal formulation from the paper:
- Continuous relaxation of discrete settlement placement (x_pl in [0,1])
- Board: a_lk = expected production rate of resource k at location l (dice probs)
- Decision vars: x_pl = extent player p occupies location l
- Constraints: placement budget (sum_l x_pl = 2 init / <= remaining mid-game),
  location capacity (sum_p x_pl <= 1), distance rule (sum_p(x_pl + x_pm) <= 1 for adj (l,m))
- Objective: max t - lambda * sum_p ||rho_p - rho_bar_p * 1||^2  s.t. t <= v_p for all p
  where rho_p = x_p @ A, v_p = rho_p @ w, rho_bar_p = mean(rho_p)
"""

import numpy as np
import cvxpy as cp

from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.map import NUM_NODES
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ActionType

N_RESOURCES = 5
LAMBDA = 0.5
W = np.ones(N_RESOURCES)


def _production_matrix(game):
    """A[l,k] = a_lk = expected production rate of resource k at location l."""
    cmap = game.state.board.map
    A = np.zeros((NUM_NODES, N_RESOURCES))
    for node_id, prod in cmap.node_production.items():
        if 0 <= node_id < NUM_NODES:
            for k, res in enumerate(RESOURCES):
                A[node_id, k] = prod.get(res, 0.0)
    return A


def _occupied(game):
    """Nodes that already have a building."""
    out = set()
    for color in game.state.colors:
        for btype in (SETTLEMENT, CITY):
            nodes = game.state.buildings_by_color.get(color, {}).get(btype, [])
            out.update(n for n in nodes if 0 <= n < NUM_NODES)
    return out


def _adjacent_pairs():
    """Pairs (u,v) with u < v that are adjacent on the board."""
    edges = get_edges()
    return [(min(a, b), max(a, b)) for a, b in edges if a < NUM_NODES and b < NUM_NODES]


def run_lp(game, our_color, budgets, exact_budget=False, A_override=None):
    """
    Solve fair allocation LP: maximin across all players + balance penalty.
    exact_budget: if True, use sum_l x_pl = budget (initial placement); else <= (mid-game).
    Returns node scores for our player (our slice of the fair allocation).
    """
    state = game.state
    colors = list(state.colors)
    P = len(colors)
    our_idx = colors.index(our_color)

    # Allow an override of the production matrix so that, for example,
    # robber-adjusted production (A^rob) can be injected without changing
    # the LP structure.
    A = A_override if A_override is not None else _production_matrix(game)
    L = NUM_NODES
    adj = _adjacent_pairs()
    occ = _occupied(game)

    x = cp.Variable((P, L), nonneg=True)
    rho = x @ A  # rho_pk = sum_l a_lk x_pl
    v = rho @ W  # v_p = sum_k w_k rho_pk
    t = cp.Variable()

    # Penalty: sum_p ||rho_p - rho_bar_p * 1||^2, rho_bar_p = mean(rho_p)
    penalty = sum(cp.sum_squares(rho[p] - cp.sum(rho[p]) / N_RESOURCES) for p in range(P))

    if exact_budget:
        constraints = [cp.sum(x[p]) == budgets[p] for p in range(P)]
    else:
        constraints = [cp.sum(x[p]) <= budgets[p] for p in range(P)]
    constraints += [t <= v[p] for p in range(P)]  # t = min over p of v_p
    constraints += [cp.sum(x[:, l]) <= 1 for l in range(L)]  # location capacity
    constraints += [cp.sum(x[:, u]) + cp.sum(x[:, v]) <= 1 for u, v in adj]  # distance rule
    constraints += [x <= 1]  # x_pl in [0,1]
    for l in occ:
        constraints.append(cp.sum(x[:, l]) == 0)

    prob = cp.Problem(cp.Maximize(t - LAMBDA * penalty), constraints)
    try:
        prob.solve(verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
            return x.value[our_idx]
    except Exception:
        pass
    return None


def solve_initial(game, our_color, A_override=None):
    """Node scores for initial placement (exactly 2 settlements each).

    If A_override is provided, it is used as the production matrix instead of
    recomputing it from the board. This lets us plug in robber-aware or other
    adjusted production models while keeping the LP itself unchanged.
    """
    budgets = [2] * len(game.state.colors)
    scores = run_lp(
        game,
        our_color,
        budgets,
        exact_budget=True,
        A_override=A_override,
    )
    if scores is None:
        A = A_override if A_override is not None else _production_matrix(game)
        scores = A @ W
    return scores


def solve_build(game, our_color, A_override=None):
    """Node and edge scores for mid-game builds. Called each turn when we have build options.

    If A_override is provided, it is used as the production matrix instead of
    recomputing it from the board, mirroring solve_initial.
    """
    state = game.state
    budgets = []
    for color in state.colors:
        b = state.buildings_by_color.get(color, {})
        n = len(b.get(SETTLEMENT, [])) + len(b.get(CITY, []))
        budgets.append(max(0, 5 - n))

    node_scores = run_lp(
        game,
        our_color,
        budgets,
        exact_budget=False,
        A_override=A_override,
    )
    if node_scores is None:
        A = A_override if A_override is not None else _production_matrix(game)
        node_scores = A @ W

    edges = get_edges()
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    edge_scores = np.array([
        (node_scores[a] + node_scores[b]) / 2
        for a, b in edges
    ])
    return node_scores, edge_scores, edge_to_idx


def score_action(action, node_scores, edge_scores, edge_to_idx):
    """Score a build action from LP solution."""
    try:
        if action.action_type == ActionType.BUILD_SETTLEMENT or action.action_type == ActionType.BUILD_CITY:
            n = action.value
            return float(node_scores[n]) if 0 <= n < len(node_scores) else 0.0
        if action.action_type == ActionType.BUILD_ROAD:
            e = tuple(sorted(action.value))
            i = edge_to_idx.get(e)
            return float(edge_scores[i]) if i is not None else 0.0
    except (IndexError, KeyError, TypeError):
        pass
    return 0


def score_robber_hexes_fairness(game):
    """
    Fairness-aligned heuristic scores for robber moves.

    For each hex index h, we approximate how much blocking h would reduce the
    disparity between players' expected production, and return a score c_h
    equal to that reduction: higher is better.

    This does not solve a CVXPY problem; it is a deterministic scoring rule
    that is consistent with the fairness objective used in run_lp.
    """
    state = game.state
    colors = list(state.colors)
    P = len(colors)
    if P == 0:
        return {}

    cmap = state.board.map
    tiles = list(cmap.tiles)
    num_tiles = len(tiles)

    # Map tile objects to indices so we can accumulate contributions.
    tile_to_idx = {tile: idx for idx, tile in enumerate(tiles)}

    # Dice probabilities for sums of two six-sided dice.
    dice_probs = {
        2: 1 / 36.0,
        3: 2 / 36.0,
        4: 3 / 36.0,
        5: 4 / 36.0,
        6: 5 / 36.0,
        7: 6 / 36.0,
        8: 5 / 36.0,
        9: 4 / 36.0,
        10: 3 / 36.0,
        11: 2 / 36.0,
        12: 1 / 36.0,
    }

    # contrib[color][tile_idx] = approximate contribution of that hex to player's score.
    contrib = {color: np.zeros(num_tiles, dtype=float) for color in colors}

    for color in colors:
        buildings = state.buildings_by_color.get(color, {})
        for btype, weight in ((SETTLEMENT, 1.0), (CITY, 2.0)):
            for node_id in buildings.get(btype, []):
                tiles_here = cmap.adjacent_tiles.get(node_id, [])
                for tile in tiles_here:
                    idx = tile_to_idx.get(tile)
                    if idx is None:
                        continue
                    number = getattr(tile, "number", None)
                    prob = dice_probs.get(number, 0.0)
                    contrib[color][idx] += weight * prob

    # Baseline scalar scores v_p are sums over tiles for each player.
    v = np.array(
        [contrib[color].sum() for color in colors],
        dtype=float,
    )
    baseline_range = float(v.max() - v.min()) if len(v) > 0 else 0.0

    # If everyone already has equal production, all robber moves are neutral.
    if baseline_range <= 0:
        return {idx: 0.0 for idx in range(num_tiles)}

    scores = {}
    for h in range(num_tiles):
        # New scores after blocking hex h: subtract its contribution for each player.
        new_v = np.array(
            [v_p - contrib[color][h] for v_p, color in zip(v, colors)],
            dtype=float,
        )
        new_range = float(new_v.max() - new_v.min())
        scores[h] = baseline_range - new_range

    return scores

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


def gini(values):
    """Compute Gini coefficient for a 1D numeric vector."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mean_v = float(np.mean(arr))
    if mean_v <= 0:
        return 0.0
    diff = np.abs(arr[:, None] - arr[None, :]).sum()
    n = arr.size
    return float(diff / (2.0 * n * n * mean_v))


def shannon_entropy(values):
    """Compute Shannon entropy for a non-negative vector (natural log)."""
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def summarize_production(rho):
    """Summarize fairness/efficiency metrics from per-player production vectors rho."""
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError("rho must be a 2D matrix with shape (players, resources)")
    v = rho @ W
    weakest = float(np.min(v)) if v.size else 0.0
    strongest = float(np.max(v)) if v.size else 0.0
    max_min_ratio = float(strongest / weakest) if weakest > 1e-12 else float("inf")
    entropies = np.array([shannon_entropy(rho[p]) for p in range(rho.shape[0])], dtype=float)
    gaps = strongest - v
    return {
        "production_by_player": v,
        "gini": gini(v),
        "max_min_ratio": max_min_ratio,
        "total_expected_production": float(np.sum(v)),
        "mean_entropy": float(np.mean(entropies)) if entropies.size else float("nan"),
        "entropy_by_player": entropies,
        "gap_from_max_by_player": gaps,
    }


def expected_production_gini(game):
    """Gini of expected production across players from current buildings.
    Uses settlements (1x) and cities (2x) production. What the LP directly optimizes."""
    A = _production_matrix(game)
    state = game.state
    colors = list(state.colors)
    v = []
    for color in colors:
        rho_p = np.zeros(N_RESOURCES, dtype=float)
        for btype, mult in ((SETTLEMENT, 1.0), (CITY, 2.0)):
            nodes = state.buildings_by_color.get(color, {}).get(btype, [])
            for n in nodes:
                if 0 <= n < A.shape[0]:
                    rho_p += mult * A[n, :]
        v.append(float(np.dot(rho_p, W)))
    return gini(v), np.array(v)


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


def run_lp(game, our_color, budgets, exact_budget=False, A_override=None, lambda_value=None):
    """
    Solve fair allocation LP: maximin across all players + balance penalty.
    exact_budget: if True, use sum_l x_pl = budget (initial placement); else <= (mid-game).
    Returns allocation details for all players.
    """
    state = game.state
    colors = list(state.colors)
    P = len(colors)

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

    lambda_eff = LAMBDA if lambda_value is None else float(lambda_value)
    prob = cp.Problem(cp.Maximize(t - lambda_eff * penalty), constraints)
    try:
        prob.solve(verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
            capacity_start = 2 * P
            capacity_duals = np.array([
                float(c.dual_value) if c.dual_value is not None else 0.0
                for c in constraints[capacity_start: capacity_start + L]
            ])
            return {
                "status": prob.status,
                "x": np.asarray(x.value, dtype=float),
                "rho": np.asarray(rho.value, dtype=float),
                "v": np.asarray(v.value, dtype=float),
                "t": float(t.value),
                "penalty": float(penalty.value),
                "objective": float(prob.value),
                "lambda": lambda_eff,
                "capacity_duals": capacity_duals,
            }
    except Exception:
        pass
    return None

def solve_initial(game, our_color, A_override=None, lambda_value=None):
    """Node scores for initial placement (exactly 2 settlements each).

    If A_override is provided, it is used as the production matrix instead of
    recomputing it from the board. This lets us plug in robber-aware or other
    adjusted production models while keeping the LP itself unchanged.
    """
    budgets = [2] * len(game.state.colors)
    scores = run_lp(game, our_color, budgets, exact_budget=True, A_override=A_override, lambda_value=lambda_value)
    if scores is None:
        A = A_override if A_override is not None else _production_matrix(game)
        scores = A @ W
    return scores


def solve_build(game, our_color, A_override=None, lambda_value=None):
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
        lambda_value=lambda_value
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


def solve_initial_all_players(game, A_override=None, lambda_value=None):
    """Solve initial-placement LP and return production/fairness metrics for all players."""
    budgets = [2] * len(game.state.colors)
    details = run_lp(game, budgets, exact_budget=True, A_override=A_override, lambda_value=lambda_value)
    if details is None:
        A = A_override if A_override is not None else _production_matrix(game)
        P = len(game.state.colors)
        board_resource_mean = np.mean(A, axis=0)
        rho = np.tile(2.0 * board_resource_mean, (P, 1))
        summary = summarize_production(rho)
        return {
            "status": "fallback",
            "rho": rho,
            "v": summary["production_by_player"],
            "t": float(np.min(summary["production_by_player"])),
            "penalty": 0.0,
            "objective": float(np.min(summary["production_by_player"])),
            "capacity_duals": np.zeros(NUM_NODES, dtype=float),
            "metrics": summary,
        }

    summary = summarize_production(details["rho"])
    details["metrics"] = summary
    return details


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


def solve_trade_and_build(game, our_color, alpha=1.0, beta=0.1):
    """
    Per-turn convex trading helper for a single player (Option A).

    This sets up a small convex problem for the acting player that decides how
    much to trade in/out of each resource this turn, with a fairness-aligned
    objective on their post-trade hand. It does NOT execute actions or change
    global placements; it only returns suggested net trades.

    Variables (length = N_RESOURCES):
      - t_in[r]  >= 0 : amount of resource r traded in
      - t_out[r] >= 0 : amount of resource r traded out

    Constraints:
      - q[r] = q0[r] + t_in[r] - t_out[r]   (post-turn hand)
      - q[r] >= 0                           (cannot overspend)
      - sum_r t_in[r] <= (1 / rho) * sum_r t_out[r]  (aggregated trade ratio)

    Objective:
      maximize   alpha * sum_r q[r]  -  beta * || q - mean(q) * 1 ||_2^2

    This keeps the problem convex (linear constraints, concave objective) and
    encourages both higher total resources and a more balanced hand. The
    returned trades can be mapped to concrete MARITIME_TRADE actions by the
    agent if desired.
    """
    state = game.state

    # Current hand q0[r] for our_color.
    q0 = np.zeros(N_RESOURCES, dtype=float)
    try:
        hand = state.hands.get(our_color, {})
        for idx, res in enumerate(RESOURCES):
            q0[idx] = hand.get(res, 0)
    except Exception:
        # If hand structure is different, fall back to zeros (no trades).
        return None, None

    # Best available trade ratio rho from ports or bank.
    rho = 4.0  # default 4:1 bank trade
    try:
        port_resources = state.board.get_player_port_resources(our_color)
        if port_resources:
            # If player has a generic 3:1 port (represented as None), use 3:1.
            if None in port_resources:
                rho = 3.0
            else:
                # Has at least one 2:1 specific port; approximate with 2:1.
                rho = 2.0
    except Exception:
        pass

    # Decision variables: trades in/out of each resource.
    t_in = cp.Variable(N_RESOURCES, nonneg=True)
    t_out = cp.Variable(N_RESOURCES, nonneg=True)

    # Post-turn hand.
    q = q0 + t_in - t_out

    constraints = []
    # Cannot overspend.
    constraints.append(q >= 0)
    # Aggregated trade-ratio constraint.
    constraints.append(cp.sum(t_in) <= (1.0 / rho) * cp.sum(t_out))

    # Fairness-aligned objective on the hand: more total resources + better balance.
    if N_RESOURCES > 0:
        q_mean = cp.sum(q) / N_RESOURCES
        imbalance = cp.sum_squares(q - q_mean)
    else:
        imbalance = 0.0

    objective = cp.Maximize(alpha * cp.sum(q) - beta * imbalance)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and t_in.value is not None:
            return t_in.value, t_out.value
    except Exception:
        pass

    return None, None

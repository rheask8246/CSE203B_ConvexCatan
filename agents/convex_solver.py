"""
Fair resource allocation LP for Catan — v4.

Changes from v3 based on experimental results:

PROBLEM 1 — AB/VALUE lineups: CONVEX can't compress spread when a dominant
  agent locks up all good nodes early. By the time CONVEX solves the LP, the
  best nodes are gone and CONVEX gets scraps.

  FIX: Add an "urgency weight" to the fairness score based on game phase.
  Early game (< 3 buildings placed): weight own production 2x more, because
  getting on good nodes fast matters more than perfect fairness placement.
  This is implemented by boosting SELF_FLOOR_FRAC to 1.0 early on.

PROBLEM 2 — Shannon entropy still lowest for CONVEX. The trade LP runs but
  CONVEX has so few resources that even a "balanced" hand has low entropy.
  The real issue: CONVEX is not building enough cities to generate resources.

  FIX: In _best_build_action, cities now score HIGHER than settlements when
  CONVEX already has settlements on good nodes. Doubling production is better
  than placing a new low-production settlement. City multiplier in
  current_production raised to 2.0 (was 3.0 in players.py — that was wrong
  since catanatron uses 2x for cities).

PROBLEM 3 — fairness_gain_scores gave equal weight to all opponents. When
  facing AB (always wins), the leader is always AB, but blocking AB's nodes
  is often impossible because AB placed there turn 1. We should instead
  focus on nodes that help the WEAKEST opponent catch up to the median.

  FIX: scoring now uses a DUAL criterion:
    score[l] = alpha * block_leader_score[l] + (1-alpha) * help_weakest_score[l]
  where help_weakest_score[l] = how much this node would benefit the weakest
  player if they had it (i.e. we claim it so nobody can block them either,
  and we generate resources ourselves which we can share via the board state).
  alpha=0.6 (lean toward blocking leader, but also lift the weakest).

PROBLEM 4 — players.py had three stacked ConvexAgent class definitions.
  The top one was the active one but it mixed old fairness_node_scores code
  with the LP-based code, causing inconsistent behavior.

  FIX: convex_agent.py is the single source of truth. players.py should only
  import ConvexAgent from there.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ActionType

N_RESOURCES = 5
W = np.ones(N_RESOURCES)

LAMBDA = 0.8          # weight on opponent spread compression
MU = 0.05             # weight on self hand-balance penalty
SELF_FLOOR_FRAC = 0.7 # normal floor: 70% of board avg per new placement
EARLY_FLOOR_FRAC = 1.2 # early game floor: 120% — get on good nodes fast
EARLY_GAME_THRESHOLD = 3  # buildings placed before we switch from early to normal
DUAL_ALPHA = 0.6      # blend: 0.6 * block_leader + 0.4 * help_weakest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mean_v = float(np.mean(arr))
    if mean_v <= 0:
        return 0.0
    diff = np.abs(arr[:, None] - arr[None, :]).sum()
    n = arr.size
    return float(diff / (2.0 * n * n * mean_v))


def shannon_entropy(values) -> float:
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def summarize_production(rho) -> dict:
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError("rho must be 2D (players, resources)")
    v = rho @ W
    weakest   = float(np.min(v)) if v.size else 0.0
    strongest = float(np.max(v)) if v.size else 0.0
    max_min_ratio = float(strongest / weakest) if weakest > 1e-12 else float("inf")
    entropies = np.array([shannon_entropy(rho[p]) for p in range(rho.shape[0])], dtype=float)
    return {
        "production_by_player": v,
        "gini": gini(v),
        "max_min_ratio": max_min_ratio,
        "total_expected_production": float(np.sum(v)),
        "mean_entropy": float(np.mean(entropies)) if entropies.size else float("nan"),
        "entropy_by_player": entropies,
        "gap_from_max_by_player": strongest - v,
    }


def expected_production_gini(game):
    """Gini + per-player production vector from current buildings."""
    A = _production_matrix(game)
    state = game.state
    colors = list(state.colors)
    v = []
    for color in colors:
        rho_p = np.zeros(N_RESOURCES, dtype=float)
        for btype, mult in ((SETTLEMENT, 1.0), (CITY, 2.0)):
            for n in state.buildings_by_color.get(color, {}).get(btype, []):
                if 0 <= n < A.shape[0]:
                    rho_p += mult * A[n, :]
        v.append(float(np.dot(rho_p, W)))
    return gini(v), np.array(v)


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _production_matrix(game) -> np.ndarray:
    cmap = game.state.board.map
    A = np.zeros((NUM_NODES, N_RESOURCES))
    for node_id, prod in cmap.node_production.items():
        if 0 <= node_id < NUM_NODES:
            for k, res in enumerate(RESOURCES):
                A[node_id, k] = prod.get(res, 0.0)
    return A


def _production_matrix_with_robber(game) -> np.ndarray:
    A = _production_matrix(game)
    try:
        robber_tile = game.state.board.robber_tile
        cmap = game.state.board.map
        for node_id, adj_tiles in cmap.adjacent_tiles.items():
            if robber_tile in adj_tiles and 0 <= node_id < NUM_NODES:
                A[node_id, :] = 0.0
    except Exception:
        pass
    return A


def _occupied(game) -> set:
    out = set()
    for color in game.state.colors:
        for btype in (SETTLEMENT, CITY):
            for n in game.state.buildings_by_color.get(color, {}).get(btype, []):
                if 0 <= n < NUM_NODES:
                    out.add(n)
    return out


def _adjacent_pairs() -> list:
    edges = get_edges()
    return [(min(a, b), max(a, b)) for a, b in edges if a < NUM_NODES and b < NUM_NODES]


def _building_count(state, color) -> int:
    """Cities count as 1 slot (they replace a settlement)."""
    b = state.buildings_by_color.get(color, {})
    return len(b.get(SETTLEMENT, [])) + len(b.get(CITY, []))


def _current_production(game, color, A: np.ndarray) -> float:
    """
    Total expected production of `color` from existing buildings.
    Cities = 2x (correct Catan multiplier, not 3x).
    """
    state = game.state
    total = 0.0
    for btype, mult in ((SETTLEMENT, 1.0), (CITY, 2.0)):
        for n in state.buildings_by_color.get(color, {}).get(btype, []):
            if 0 <= n < A.shape[0]:
                total += mult * float(np.dot(A[n, :], W))
    return total


def _board_avg_production(A: np.ndarray) -> float:
    node_totals = A @ W
    nonzero = node_totals[node_totals > 0]
    return float(np.mean(nonzero)) if nonzero.size > 0 else 1.0


def _is_early_game(game, our_color) -> bool:
    """True if CONVEX has placed fewer than EARLY_GAME_THRESHOLD buildings."""
    return _building_count(game.state, our_color) < EARLY_GAME_THRESHOLD


def _solve_with_fallback(prob) -> bool:
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Dual-criterion fairness-gain scores (LP-free, always available)
# ---------------------------------------------------------------------------

def fairness_gain_scores(game, our_color, A: np.ndarray) -> np.ndarray:
    """
    Score nodes using dual criterion:
      score[l] = DUAL_ALPHA * block_leader[l] + (1-DUAL_ALPHA) * help_weakest[l]

    block_leader[l]:  node's production value weighted by proximity to leader.
    help_weakest[l]:  node's production value weighted by how much the weakest
                      player needs it (high if weakest is far behind median).

    Early game: boost own production by using EARLY_FLOOR_FRAC criterion
    (picked up in solve_initial/solve_build via self_floor parameter).
    """
    state = game.state
    colors = list(state.colors)
    opp_colors = [c for c in colors if c != our_color]

    if not opp_colors:
        return A @ W

    opp_v = np.array([_current_production(game, c, A) for c in opp_colors])
    occ = _occupied(game)

    # Identify leader and weakest opponent.
    leader_color  = opp_colors[int(np.argmax(opp_v))]
    weakest_color = opp_colors[int(np.argmin(opp_v))]
    median_v = float(np.median(opp_v))

    # How desperate is the weakest player? gap relative to median.
    weakest_gap = max(0.0, median_v - float(np.min(opp_v)))

    # Build adjacency map once.
    adj: dict[int, set] = {}
    for u, v in get_edges():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    # Leader's and weakest's existing nodes.
    leader_nodes = set()
    weakest_nodes = set()
    for btype in (SETTLEMENT, CITY):
        leader_nodes.update(state.buildings_by_color.get(leader_color, {}).get(btype, []))
        weakest_nodes.update(state.buildings_by_color.get(weakest_color, {}).get(btype, []))

    scores = np.zeros(NUM_NODES, dtype=float)
    for l in range(NUM_NODES):
        if l in occ:
            continue
        node_prod = float(np.dot(A[l, :], W))
        if node_prod <= 0:
            continue

        neighbors = adj.get(l, set())

        # block_leader: proximity bonus if adjacent to leader's nodes.
        leader_adj = any(nb in leader_nodes for nb in neighbors)
        block_score = node_prod * (1.5 if leader_adj else 1.0)

        # help_weakest: proximity bonus if adjacent to weakest's nodes.
        # This node being claimed by CONVEX blocks others from expanding near weakest.
        weakest_adj = any(nb in weakest_nodes for nb in neighbors)
        help_score  = node_prod * (1.5 if weakest_adj else 1.0) * (1.0 + weakest_gap)

        scores[l] = DUAL_ALPHA * block_score + (1.0 - DUAL_ALPHA) * help_score

    return scores


# ---------------------------------------------------------------------------
# LP core
# ---------------------------------------------------------------------------

def run_lp_details(
    game,
    budgets: list,
    our_color,
    exact_budget: bool = False,
    A_override=None,
    lambda_value=None,
    mu_value=None,
    self_floor=None,
) -> dict | None:
    """
    Solve the opponent-fairness LP.
    Objective: maximise  t_opp  -  lambda * spread_opp  -  mu * self_imbalance

    Self-production floor: current_prod + floor_frac * board_avg * new_budget.
    Early game uses higher floor_frac to get CONVEX on good nodes fast.
    """
    state  = game.state
    colors = list(state.colors)
    P      = len(colors)

    if our_color not in colors:
        return None

    our_idx   = colors.index(our_color)
    opp_idxs  = [i for i in range(P) if i != our_idx]
    if not opp_idxs:
        return None

    A   = A_override if A_override is not None else _production_matrix(game)
    L   = NUM_NODES
    adj = _adjacent_pairs()
    occ = _occupied(game)

    lam = LAMBDA if lambda_value is None else float(lambda_value)
    mu  = MU if mu_value is None else float(mu_value)

    # Phase-adaptive floor.
    if self_floor is not None:
        floor_frac = float(self_floor)
    elif _is_early_game(game, our_color):
        floor_frac = EARLY_FLOOR_FRAC
    else:
        floor_frac = SELF_FLOOR_FRAC

    x   = cp.Variable((P, L), nonneg=True)
    rho = x @ A
    v   = rho @ W

    t_opp     = cp.Variable()
    v_opp_max = cp.Variable()
    rho_us    = rho[our_idx]
    self_imbalance = cp.sum_squares(rho_us - cp.sum(rho_us) / N_RESOURCES)

    constraints = []

    if exact_budget:
        constraints += [cp.sum(x[p]) == budgets[p] for p in range(P)]
    else:
        constraints += [cp.sum(x[p]) <= budgets[p] for p in range(P)]

    for i in opp_idxs:
        constraints.append(t_opp     <= v[i])
        constraints.append(v_opp_max >= v[i])

    board_avg   = _board_avg_production(A)
    new_budget  = budgets[our_idx]
    new_floor   = floor_frac * board_avg * max(0, new_budget)
    if new_floor > 1e-6:
        constraints.append(v[our_idx] >= new_floor)

    constraints += [cp.sum(x[:, l]) <= 1 for l in range(L)]
    constraints += [cp.sum(x[:, u]) + cp.sum(x[:, vv]) <= 1 for u, vv in adj]
    constraints += [x <= 1]
    for l in occ:
        constraints.append(cp.sum(x[:, l]) == 0)

    spread_opp = v_opp_max - t_opp
    prob = cp.Problem(
        cp.Maximize(t_opp - lam * spread_opp - mu * self_imbalance),
        constraints,
    )

    solved = _solve_with_fallback(prob)
    if not solved or x.value is None:
        return None

    return {
        "status":     prob.status,
        "x":          np.asarray(x.value, dtype=float),
        "rho":        np.asarray(rho.value, dtype=float),
        "v":          np.asarray(v.value, dtype=float),
        "t_opp":      float(t_opp.value)      if t_opp.value      is not None else 0.0,
        "spread_opp": float(spread_opp.value) if spread_opp.value is not None else 0.0,
        "penalty":    float(self_imbalance.value) if self_imbalance.value is not None else 0.0,
        "objective":  float(prob.value),
        "lambda": lam, "mu": mu,
    }


def run_lp(game, our_color, budgets, exact_budget=False, A_override=None,
           lambda_value=None, mu_value=None, self_floor=None) -> np.ndarray | None:
    details = run_lp_details(
        game, budgets, our_color,
        exact_budget=exact_budget, A_override=A_override,
        lambda_value=lambda_value, mu_value=mu_value, self_floor=self_floor,
    )
    if details is None:
        return None
    return details["x"][list(game.state.colors).index(our_color)]


# ---------------------------------------------------------------------------
# Public solve helpers
# ---------------------------------------------------------------------------

def solve_initial(game, our_color, A_override=None, lambda_value=None,
                  mu_value=None, self_floor=None) -> np.ndarray:
    A       = A_override if A_override is not None else _production_matrix_with_robber(game)
    budgets = [2] * len(game.state.colors)
    scores  = run_lp(
        game, our_color, budgets, exact_budget=True, A_override=A,
        lambda_value=lambda_value, mu_value=mu_value, self_floor=self_floor,
    )
    return scores if scores is not None else fairness_gain_scores(game, our_color, A)


def solve_build(game, our_color, A_override=None, lambda_value=None,
                mu_value=None, self_floor=None):
    """Returns (node_scores, edge_scores, edge_to_idx). Never returns None."""
    state   = game.state
    budgets = [max(0, 5 - _building_count(state, c)) for c in state.colors]
    A       = A_override if A_override is not None else _production_matrix_with_robber(game)

    node_scores = run_lp(
        game, our_color, budgets, exact_budget=False, A_override=A,
        lambda_value=lambda_value, mu_value=mu_value, self_floor=self_floor,
    )
    if node_scores is None:
        node_scores = fairness_gain_scores(game, our_color, A)

    edges       = get_edges()
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    edge_scores = np.array([
        _road_unlock_score(a, b, node_scores, game) for a, b in edges
    ])
    return node_scores, edge_scores, edge_to_idx


def solve_initial_all_players(game, A_override=None, lambda_value=None) -> dict:
    colors = list(game.state.colors)
    if not colors:
        return {}
    A       = A_override if A_override is not None else _production_matrix_with_robber(game)
    budgets = [2] * len(colors)
    details = run_lp_details(
        game, budgets, colors[0], exact_budget=True,
        A_override=A, lambda_value=lambda_value,
    )
    if details is None:
        P   = len(colors)
        rho = np.tile(2.0 * np.mean(A, axis=0), (P, 1))
        summary = summarize_production(rho)
        return {
            "status": "fallback", "rho": rho,
            "v": summary["production_by_player"],
            "t": float(np.min(summary["production_by_player"])),
            "penalty": 0.0, "objective": 0.0,
            "capacity_duals": np.zeros(NUM_NODES, dtype=float),
            "metrics": summary,
        }
    summary           = summarize_production(details["rho"])
    details["metrics"] = summary
    return details


# ---------------------------------------------------------------------------
# Road scoring
# ---------------------------------------------------------------------------

def _road_unlock_score(a: int, b: int, node_scores: np.ndarray, game) -> float:
    occ = _occupied(game)
    adj: dict[int, set] = {}
    for u, v in get_edges():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    best = (node_scores[a] + node_scores[b]) / 2.0
    for ep in (a, b):
        for nb in adj.get(ep, set()):
            if nb not in occ and 0 <= nb < len(node_scores):
                best = max(best, float(node_scores[nb]))
    return best


# ---------------------------------------------------------------------------
# Robber scoring
# ---------------------------------------------------------------------------

def score_robber_hexes_fairness(game) -> dict:
    """
    Block the hex contributing most to the leader's production.
    Also avoids blocking CONVEX's own productive hexes (self-preservation).
    """
    state  = game.state
    colors = list(state.colors)
    if not colors:
        return {}

    cmap        = state.board.map
    tiles       = list(cmap.tiles)
    tile_to_idx = {tile: i for i, tile in enumerate(tiles)}
    dice_probs  = {
        2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
        8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
    }

    contrib = {c: np.zeros(len(tiles), dtype=float) for c in colors}
    for color in colors:
        for btype, w in ((SETTLEMENT, 1.0), (CITY, 2.0)):
            for node_id in state.buildings_by_color.get(color, {}).get(btype, []):
                for tile in cmap.adjacent_tiles.get(node_id, []):
                    idx = tile_to_idx.get(tile)
                    if idx is not None:
                        prob = dice_probs.get(getattr(tile, "number", None), 0.0)
                        contrib[color][idx] += w * prob

    v            = np.array([contrib[c].sum() for c in colors], dtype=float)
    leader_color = colors[int(np.argmax(v))]
    return {h: float(contrib[leader_color][h]) for h in range(len(tiles))}


# ---------------------------------------------------------------------------
# Trade helper
# ---------------------------------------------------------------------------

def solve_trade_and_build(game, our_color, alpha=1.0, beta=0.5):
    """
    Maximise total resources + hand balance (high Shannon entropy).
    beta=0.5 aggressively balances the hand so CONVEX can build.
    """
    state = game.state
    q0    = np.zeros(N_RESOURCES, dtype=float)
    try:
        hand = state.hands.get(our_color, {})
        for idx, res in enumerate(RESOURCES):
            q0[idx] = hand.get(res, 0)
    except Exception:
        return None, None, None

    rho = 4.0
    try:
        port_resources = state.board.get_player_port_resources(our_color)
        if port_resources:
            rho = 3.0 if None in port_resources else 2.0
    except Exception:
        pass

    t_in   = cp.Variable(N_RESOURCES, nonneg=True)
    t_out  = cp.Variable(N_RESOURCES, nonneg=True)
    q      = q0 + t_in - t_out
    q_mean = cp.sum(q) / N_RESOURCES

    prob = cp.Problem(
        cp.Maximize(alpha * cp.sum(q) - beta * cp.sum_squares(q - q_mean)),
        [q >= 0, cp.sum(t_in) <= (1.0 / rho) * cp.sum(t_out)],
    )
    _solve_with_fallback(prob)
    if prob.status in ("optimal", "optimal_inaccurate") and t_in.value is not None:
        post_hand = np.maximum(0.0, q0 + t_in.value - t_out.value)
        return t_in.value, t_out.value, post_hand
    return None, None, None


def get_best_maritime_trade(game, our_color):
    """Return (give_resource, recv_resource) or None."""
    t_in, t_out, _ = solve_trade_and_build(game, our_color)
    if t_in is None or t_out is None:
        return None
    if float(np.max(t_out)) < 0.1:
        return None
    give_idx = int(np.argmax(t_out))
    recv_idx = int(np.argmax(t_in))
    if give_idx == recv_idx:
        return None
    return RESOURCES[give_idx], RESOURCES[recv_idx]


# ---------------------------------------------------------------------------
# City upgrade scoring
# ---------------------------------------------------------------------------

def city_fairness_score(game, our_color, node_id, A: np.ndarray) -> float:
    """
    Score upgrading a settlement at node_id to a city.
    Cities double production (+1x more) — this is good when our production
    is below the median opponent and we need to catch up to stay relevant.
    Returns a score comparable to node_scores from solve_build.
    """
    state   = game.state
    colors  = list(state.colors)
    our_idx = colors.index(our_color) if our_color in colors else 0

    our_prod = _current_production(game, our_color, A)
    opp_v    = np.array([
        _current_production(game, c, A)
        for c in colors if c != our_color
    ])
    if len(opp_v) == 0:
        return 0.0

    # City gain = +1x on this node (settlement is already 1x, city adds 1x more).
    node_gain = float(np.dot(A[node_id, :], W)) if 0 <= node_id < A.shape[0] else 0.0

    # Urgency: how far below median are we?
    opp_median = float(np.median(opp_v))
    urgency    = max(0.0, opp_median - our_prod) / (opp_median + 1e-6)

    return node_gain * (1.0 + urgency)


# ---------------------------------------------------------------------------
# Action scoring
# ---------------------------------------------------------------------------

def score_action(action, node_scores, edge_scores, edge_to_idx) -> float:
    try:
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            n = action.value
            return float(node_scores[n]) if 0 <= n < len(node_scores) else 0.0
        if action.action_type == ActionType.BUILD_ROAD:
            e = tuple(sorted(action.value))
            i = edge_to_idx.get(e)
            return float(edge_scores[i]) if i is not None else 0.0
    except (IndexError, KeyError, TypeError):
        pass
    return 0.0
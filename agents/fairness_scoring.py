"""
Fairness-first scoring for ConvexAgent.
Minimizes global inequality; early game prioritizes production to reach 4-6 VP.
"""

from __future__ import annotations

import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY

N_RESOURCES = 5
W = np.ones(N_RESOURCES)
CITY_MULT = 3.0   # cities increase inequality faster
BLOCK_WEIGHT = 1.5
IMPACT_WEIGHT = 0.3
DICE_PROBS = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
              8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}


def _production_matrix(game):
    cmap = game.state.board.map
    A = np.zeros((NUM_NODES, N_RESOURCES))
    for node_id, prod in cmap.node_production.items():
        if 0 <= node_id < NUM_NODES:
            for k, res in enumerate(RESOURCES):
                A[node_id, k] = prod.get(res, 0.0)
    return A


def _production_matrix_with_robber(game):
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


def production_inequality(v):
    """Global inequality metric. Lower = more equal."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return 0.0
    total = sum(abs(v[i] - v[j]) for i in range(n) for j in range(n))
    mean = float(np.mean(v)) + 1e-6
    return total / (2.0 * n * n * mean)


def current_production(game, color, A, W_arr):
    """Expected production per turn. Settlement=1x, City=3x."""
    state = game.state
    total = 0.0
    for btype, mult in ((SETTLEMENT, 1.0), (CITY, CITY_MULT)):
        for node_id in state.buildings_by_color.get(color, {}).get(btype, []):
            if 0 <= node_id < A.shape[0]:
                total += mult * float(np.dot(A[node_id], W_arr))
    return total


def _occupied(game):
    occ = set()
    for color in game.state.colors:
        for btype in (SETTLEMENT, CITY):
            for n in game.state.buildings_by_color.get(color, {}).get(btype, []):
                occ.add(n)
    return occ


def _node_production_gain(A, W_arr, node_id):
    return float(np.dot(A[node_id], W_arr))


def _node_impact(A, W_arr, cmap, node_id):
    """expected_production * dice_probability for highest tile on node."""
    tiles = cmap.adjacent_tiles.get(node_id, [])
    best_prob = 0.0
    for tile in tiles:
        p = DICE_PROBS.get(getattr(tile, "number", 7), 0.0)
        best_prob = max(best_prob, p)
    base = _node_production_gain(A, W_arr, node_id)
    return base * best_prob


def _reachable_from_road(game, node_a, node_b):
    occ = _occupied(game)
    adj = {}
    for u, v in get_edges():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    reachable = set()
    for ep in (node_a, node_b):
        for nb in adj.get(ep, set()):
            if nb not in occ and 0 <= nb < NUM_NODES:
                reachable.add(nb)
    return reachable


def _our_vp(game, our_color):
    try:
        idx = list(game.state.colors).index(our_color)
        key = f"P{idx}_ACTUAL_VICTORY_POINTS"
        return int(game.state.player_state.get(key, 0))
    except Exception:
        return 0


def fairness_node_scores(
    game,
    our_color,
    A,
    W_arr,
    vp,
    block_weight=BLOCK_WEIGHT,
    impact_weight=IMPACT_WEIGHT,
    use_lookahead=True,
):
    """
    Score nodes: early game (vp<3) prioritize production; else fairness.
    alpha = min(1.0, vp/4) blends fairness vs production.
    Blocking: inequality_if_leader_takes - inequality_if_we_take.
    Impact: expected_production * dice_prob.
    2-step lookahead: simulate best next settlement after placing.
    """
    state = game.state
    cmap = state.board.map
    colors = list(state.colors)
    our_idx = colors.index(our_color)
    occ = _occupied(game)

    v = np.array([current_production(game, c, A, W_arr) for c in colors])
    opp_idxs = [i for i in range(len(colors)) if i != our_idx]

    leader_idx = opp_idxs[int(np.argmax([v[i] for i in opp_idxs]))] if opp_idxs else our_idx

    alpha = min(1.0, vp / 4.0)

    scores = np.full(NUM_NODES, -np.inf)

    for node_id in range(NUM_NODES):
        if node_id in occ:
            continue

        gain = _node_production_gain(A, W_arr, node_id)

        v_ours = v.copy()
        v_ours[our_idx] += gain
        inequality_if_we_take = production_inequality(v_ours)

        v_leader = v.copy()
        v_leader[leader_idx] += gain
        inequality_if_leader_takes = production_inequality(v_leader)

        blocking_bonus = inequality_if_leader_takes - inequality_if_we_take

        fairness_score = -inequality_if_we_take + block_weight * blocking_bonus
        production_score = gain
        impact = _node_impact(A, W_arr, cmap, node_id)

        base_score = alpha * fairness_score + (1.0 - alpha) * production_score
        base_score += impact_weight * impact

        if use_lookahead:
            occ_new = occ | {node_id}
            v_after = v_ours.copy()
            best_future = -np.inf
            for n2 in range(NUM_NODES):
                if n2 in occ_new:
                    continue
                g2 = _node_production_gain(A, W_arr, n2)
                v2 = v_after.copy()
                v2[our_idx] += g2
                ineq2 = production_inequality(v2)
                best_future = max(best_future, -ineq2)
            if best_future > -np.inf:
                base_score += 0.5 * best_future

        scores[node_id] = base_score

    return scores


def fairness_road_scores(game, our_color, A, W_arr, vp, reachable_nodes):
    """road_score = max(node_scores[n] for n in reachable_nodes)."""
    node_scores = fairness_node_scores(
        game, our_color, A, W_arr, vp,
        use_lookahead=False,
    )
    road_scores = {}
    for road, nodes in reachable_nodes.items():
        best = -np.inf
        for n in nodes:
            if 0 <= n < len(node_scores) and np.isfinite(node_scores[n]):
                best = max(best, float(node_scores[n]))
        road_scores[road] = best
    return road_scores


def robber_hex_scores(game, A, W_arr):
    """
    For each hex: score = sum(production of buildings around hex).
    Choose hex with highest score (blocks largest production cluster).
    """
    state = game.state
    cmap = state.board.map
    tiles = list(cmap.tiles)
    tile_to_idx = {tile: i for i, tile in enumerate(tiles)}
    scores = np.zeros(len(tiles))

    for color in state.colors:
        for btype, mult in ((SETTLEMENT, 1.0), (CITY, CITY_MULT)):
            for node_id in state.buildings_by_color.get(color, {}).get(btype, []):
                node_val = float(np.dot(A[node_id], W_arr)) if 0 <= node_id < A.shape[0] else 0.0
                for tile in cmap.adjacent_tiles.get(node_id, []):
                    idx = tile_to_idx.get(tile)
                    if idx is not None:
                        prob = DICE_PROBS.get(getattr(tile, "number", 7), 0.0)
                        scores[idx] += mult * prob * (node_val if node_val > 0 else 1.0)

    return {h: float(scores[h]) for h in range(len(tiles))}


def entropy_trade_suggestion(game, our_color, duplicate_penalty=0.5):
    """
    Maximize hand entropy, penalize duplicate stacks.
    Returns (give_resource, recv_resource) or None.
    """
    from catanatron.models.enums import RESOURCES

    state = game.state
    try:
        hand = state.hands.get(our_color, {})
        counts = np.array([float(hand.get(r, 0)) for r in RESOURCES])
    except Exception:
        return None

    total = counts.sum()
    if total < 4:
        return None

    rho = 4.0
    try:
        port_resources = state.board.get_player_port_resources(our_color)
        if port_resources:
            rho = 3.0 if None in port_resources else 2.0
    except Exception:
        pass

    best_give, best_recv = None, None
    best_score = -np.inf

    for give_idx in range(N_RESOURCES):
        if counts[give_idx] < 1:
            continue
        for recv_idx in range(N_RESOURCES):
            if give_idx == recv_idx:
                continue
            c_new = counts.copy()
            c_new[give_idx] -= 1
            c_new[recv_idx] += 1
            total_new = c_new.sum()
            if total_new <= 0:
                continue
            p = c_new / total_new
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p)))
            dup_penalty = duplicate_penalty * (c_new.max() / (total_new + 1e-6))
            score = entropy - dup_penalty
            if score > best_score:
                best_score = score
                best_give = RESOURCES[give_idx]
                best_recv = RESOURCES[recv_idx]

    return (best_give, best_recv) if best_give is not None else None


def city_fairness_score(game, our_color, node_id, A, W_arr):
    """Score city upgrade: settlement 1x -> city 3x, so +2x gain."""
    gain = 2.0 * _node_production_gain(A, W_arr, node_id)
    v = np.array([current_production(game, c, A, W_arr) for c in game.state.colors])
    our_idx = list(game.state.colors).index(our_color)
    v_new = v.copy()
    v_new[our_idx] += gain
    return -production_inequality(v_new)

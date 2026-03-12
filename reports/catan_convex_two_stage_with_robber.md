## A Two-Stage Fairness-Aware Convex Model for Catan with Robber

This note extends the primal formulation in `Project Outline.pdf` by:

- Recapping the **fair allocation LP** for initial settlement placement.
- Explaining how to incorporate **robber blocking** via fixed parameters while keeping the problem convex.
- Describing a **two-stage convex procedure** to choose both fair placements and a fairness-aligned robber move, without introducing non-convex bilinear terms.

It is written to match the notation of the outline and the implementation in `CSE203B_ConvexCatan/agents/convex_solver.py`.

---

### 1. Base Fair-Allocation Model (Recap)

Let:

- $L$ be the set of legal settlement locations (board intersections).
- $K = \{1,\dots,5\}$ index the five resource types.
- $P$ be the number of players, indexed by $p \in \{1,\dots,P\}$.

**Board model and expected production.**

For each location $l \in L$ and resource $k \in K$, the constant
$a_{lk} \ge 0$ denotes the expected production rate of resource $k$ at
location $l$, computed from dice probabilities and adjacent hex resource types.
In the code, these coefficients form the matrix

- `A[l,k]` in `convex_solver._production_matrix(game)`.

**Decision variables.**

For each player $p$ and location $l$, introduce a relaxed placement variable

$$
x_{pl} \in [0,1],
$$

which represents the extent to which player $p$ occupies location $l$.
In the actual game $x_{pl} \in \{0,1\}$, but relaxing to $[0,1]$ yields a
convex problem. In code, all players’ variables are stacked into a matrix

- `x[p,l]` as a CVXPY variable of shape `(P, L)`.

**Constraints.**

The feasible region is given by affine (linear) constraints.

**Placement budget per player (initial phase)**  

Each player must place exactly two settlements:
$$
\sum_{l \in L} x_{pl} = 2 \quad \forall p.
$$

In mid-game, this is relaxed to:
$$
\sum_{l \in L} x_{pl} \le \text{budget}_p,
$$
where $\text{budget}_p$ depends on how many settlements/cities player $p$ already has.

**Location capacity constraint**  

Each location can be used by at most one player:
$$
\sum_{p=1}^P x_{pl} \le 1 \quad \forall l.
$$

**Distance constraint (relaxed)**  

Let $E \subseteq L \times L$ be the set of adjacent location pairs. To enforce
the “no adjacent settlements” rule in relaxed form, impose:
$$
\sum_{p=1}^P (x_{pl} + x_{pm}) \le 1 \quad \forall (l,m) \in E.
$$

In code, `get_edges()` returns all adjacent pairs, and these constraints are
added as:

```python
cp.sum(x[:, u]) + cp.sum(x[:, v]) <= 1
```

**Occupied locations (mid-game)**  

If a node already contains a building, then no LP mass may be put there:
$$
\sum_p x_{pl} = 0 \quad \forall l \text{ already occupied}.
$$

This is implemented by `_occupied(game)` and an equality constraint for each
occupied node.

All of these constraints are linear, so they define a convex polyhedral set.

**Expected resource allocation and scores.**

For each player $p$, the expected resource production vector is

$$
\mu_{pk} = \sum_{l \in L} a_{lk} x_{pl} \quad \forall k \in K,
$$

or in vector form $\mu_p = A^\top x_p$. The scalar score is

$$
v_p = \sum_{k \in K} w_k \mu_{pk},
$$

with fixed weights $w_k \ge 0$ (in code, `W = np.ones(N_RESOURCES)` so all
resources are weighted equally).

**Fairness objective.**

To promote fairness, introduce an auxiliary variable $t$ representing the
minimum expected production across players and impose

$$
v_p \ge t \quad \forall p.
$$

To encourage balanced access across resource types for each player, penalize
the within-player resource imbalance via a convex quadratic term:

$$
\bar\mu_p = \frac{1}{|K|} \sum_{k \in K} \mu_{pk}, \quad
\text{imbalance}_p = \|\mu_p - \bar\mu_p \mathbf{1}\|_2^2.
$$

The primal problem is then

$$
\max_{x,t} \quad
  t - \lambda \sum_{p=1}^P \|\mu_p - \bar\mu_p \mathbf{1}\|_2^2,
$$

subject to the constraints above, where $\lambda \ge 0$ trades off fairness
against efficiency. This objective is concave (linear term minus a convex
quadratic), and all constraints are linear, so the problem is a convex
optimization problem with a unique global optimum.

In `convex_solver.run_lp`, this is implemented as:

- `rho = x @ A` (our $\mu_p$),
- `v = rho @ W` (our $v_p$),
- `t` as a scalar CVXPY variable,
- `penalty = sum(cp.sum_squares(rho[p] - cp.sum(rho[p])/N_RESOURCES) for p in range(P))`,
- objective: `cp.Maximize(t - LAMBDA * penalty)`,
- constraints: budgets, capacity, distance, existing occupancy, and `t <= v[p]`.

The helper functions `solve_initial` and `solve_build` simply choose budgets,
call `run_lp`, and then extract node and edge scores that the agent uses
greedily for discrete placement.

---

### 2. Modeling the Robber via Fixed Parameters

The robber blocks production on whichever hex it occupies: when a 7 is rolled or
a Knight is played, the active player chooses a hex; that hex no longer produces
resources until the robber moves again.

We first model the robber’s effect as a **fixed parameter** rather than a
decision variable, so that the main LP remains convex.

Let:

- $\rho_h \in [0,1]$ denote the fraction of time hex $h$ is (effectively) blocked by the robber.

We adjust the production coefficients to obtain a robber-aware matrix
$A^{\text{rob}} = [a_{lk}^{\text{rob}}]$:

$$
a^{\text{rob}}_{lk} =
  \sum_{h \in \text{adj}(l)}
    P(\text{dice} = n_h) \cdot (1 - \rho_h) \cdot \mathbf{1}(\text{resource}(h)=k),
$$

where:

- $n_h$ is the dice number on hex $h$,
- $\text{adj}(l)$ is the set of hexes adjacent to intersection $l$,
- $\mathbf{1}(\text{resource}(h)=k)$ is 1 if hex $h$ produces resource $k$.

Crucially, in this “robber-aware but not robber-optimizing” formulation, the
$\rho_h$ are treated as **exogenous parameters**, not variables of the LP.
Intuitively, they encode a scenario about how often each hex is blocked (for
example, from empirical data or a fixed robber policy).

Because $\rho_h$ only appear through the coefficients $a^{\text{rob}}_{lk}$,
the decision variables $x_{pl}$ still enter the objective and constraints
**linearly**. We simply solve the same fair-allocation LP with $A$ replaced
by $A^{\text{rob}}$:

$$
\mu^{\text{rob}}_{pk} = \sum_{l \in L} a^{\text{rob}}_{lk} x_{pl}, \quad
v^{\text{rob}}_p = \sum_{k \in K} w_k \mu^{\text{rob}}_{pk},
$$

and the objective

$$
\max_{x,t} \quad
  t - \lambda \sum_{p=1}^P \|\mu^{\text{rob}}_p - \bar\mu^{\text{rob}}_p \mathbf{1}\|_2^2
$$

subject to

$$
v^{\text{rob}}_p \ge t \ \forall p
$$

and the same constraints on $x_{pl}$ as before. For any fixed choice of
$(\rho_h)_h$, this remains a convex optimization problem.

In implementation terms, this corresponds to building a modified production
matrix `A_rob[l,k]` and passing it as an `A_override` argument into
`run_lp`, `solve_initial`, or `solve_build`, without changing the LP structure.

---

### 3. Why Joint Robber-and-Placement Optimization is Non-Convex

If we instead try to optimize over both $x_{pl}$ and $\rho_h$ **jointly**,
then the robber-adjusted coefficients become functions of $\rho$:

$$
a^{\text{rob}}_{lk}(\rho) =
  \sum_{h \in \text{adj}(l)}
    P(\text{dice} = n_h) \cdot (1 - \rho_h) \cdot \mathbf{1}(\text{resource}(h)=k).
$$

Plugging into the expected production,

$$
\mu^{\text{rob}}_{pk} =
  \sum_l a^{\text{rob}}_{lk}(\rho) x_{pl},
$$

we obtain terms of the form $(1-\rho_h) x_{pl}$, i.e. products of two decision
variables. These bilinear terms make the objective and constraints **non-convex**
in $(x,\rho)$. Consequently, a single-stage optimization over both settlements
and robber locations does not fit within the convex framework of CSE203B.

To keep the formulation convex, we treat placements and robber as **separate but
coupled** convex problems.

---

### 4. Two-Stage Fairness-Aligned Robber Optimization (Option A)

The key idea is to:

1. Use the existing LP to compute a **fair placement** for all players.
2. Given those placements (or the realized game state they induce), solve a
   **small convex problem** that chooses a robber location to improve fairness,
   while treating the placements as fixed.

#### Stage 1: Fair Placement Optimization

Stage 1 is exactly the primal formulation described in the outline, optionally
using a robber-aware matrix $A^{\text{rob}}$ if a fixed robber scenario is
assumed. It produces:

- A fractional allocation $x^\star$,
- Expected production vectors $\mu_p^\star$,
- Scalar fairness scores $v_p^\star$.

In code, this is the role of `run_lp` (called by `solve_initial` at setup and
`solve_build` mid-game), which re-solves the LP as the board state evolves.

#### Stage 2: Robber Choice Given Placements

Given a fixed allocation pattern (either the fractional $x^\star$ or the
discrete game state after rounding), we can define a **fairness-improvement
score** for each candidate robber hex $h$.

Let:

- $v_p^\star$ be the baseline scores under the current allocations.
- For each hex $h$, define $v_p^{(h)}$ as the scores we would obtain if
  the robber is placed on hex $h$ and its production is blocked (approximated
  using expected contributions from that hex to each player).

A simple fairness metric is the **range** of player scores:

$$
\text{range} = \max_p v_p - \min_p v_p.
$$

We want the robber to **shrink** this range, pulling advantaged players closer
to the rest. For each hex $h$, define:

$$
c_h = \big(\max_p v_p^\star - \min_p v_p^\star\big)
      - \big(\max_p v_p^{(h)} - \min_p v_p^{(h)}\big).
$$

Here, $c_h > 0$ means that blocking $h$ reduces disparity; larger $c_h$
means a bigger fairness gain.

Now we introduce robber decision variables $\rho_h$ on the simplex:

- $\rho_h \ge 0$ for all hexes $h$,
- $\sum_h \rho_h = 1$.

We then solve the **robber LP**:

$$
\max_{\rho} \quad \sum_h c_h \rho_h
\quad \text{subject to} \quad
\sum_h \rho_h = 1,\; \rho_h \ge 0 \ \forall h.
$$

This is a linear program in $\rho$ because the $c_h$ are constants computed
from the fixed allocations. In practice, the optimizer simply chooses the hex
with the largest $c_h$, i.e. the robber location that most improves fairness
according to the chosen metric.

We can equivalently think of this as performing an **argmax** over hexes:

$$
h^\star = \arg\max_h c_h.
$$

The important point is that fairness is still defined in terms of the same
quantities $v_p$ and imbalances $\|\mu_p - \bar\mu_p \mathbf{1}\|^2$; we
are now using those quantities to select a robber move, not to adjust placements
and robber simultaneously.

#### Convexity preserved

- Stage 1 remains the original convex fair-allocation LP in $x_{pl}$.
- Stage 2 is a convex (indeed linear) optimization over $\rho$ (or an argmax
  over $c_h$), with $x^\star$ treated as data.
- At no point do we introduce bilinear decision-variable products like
  $x_{pl}\rho_h$. Instead, we use the solution of one convex problem as
  input to another.

This two-stage design gives a **robber-optimizing story** that is fully
compatible with convex optimization theory.

---

### 5. Initial Phase vs Mid-Game Usage

In practice, the two-stage framework can be used differently in the **initial
placement** and **mid-game** phases.

#### Initial Phase

- **Stage 1** is used to compute fair initial placements:
  - Each player’s budget is 2 settlements.
  - The LP is solved once for the starting board.
  - The relaxation $x^\star$ is rounded (or scores are used) to choose actual
    settlement and road actions.
- **Stage 2** (robber) is mostly analytical at this stage:
  - Before any 7 is rolled, the robber sits on a fixed hex (often the desert).
  - We can still ask: “Given these optimal starting placements, which hex would a
    fairness-aligned robber prefer to block?”.
  - This is insightful for understanding robustness, but typically not applied
    in live gameplay at t = 0.

#### Mid-Game

In mid-game, the same structure separates **build** and **robber** decisions:

- On **build prompts** (player can build settlement/city/road):
  - Stage 1 is re-solved with updated budgets and occupancy to score potential
    builds; the agent chooses the highest-scoring build.
  - Stage 2 is inactive (no robber move is being chosen on that prompt).

- On **robber prompts** (7 rolled or Knight played):
  - Placements and current buildings are treated as fixed (realized board).
  - Stage 2 is applied:
    - A fairness metric (e.g. range of $v_p$) is computed under the current
      state.
    - For each candidate robber hex, an approximate updated fairness value is
      computed, giving $c_h$.
    - The robber is moved to the hex with highest $c_h$, aligning robber
      behavior with the fairness objective.

This separation keeps decisions convex while providing a principled way for both
placement and robber behavior to reflect the same fairness criterion.

---

### 6. Connection to Implementation

In the `CSE203B_ConvexCatan` code:

- The fair-allocation LP and its budgets, capacity, and distance constraints are
  implemented in `agents/convex_solver.py` (`run_lp`, `solve_initial`,
  `solve_build`).
- The `ConvexAgent` uses the LP’s node and edge scores to choose discrete build
  actions.
- A helper such as `score_robber_hexes_fairness(game)` can implement Stage 2 by
  assigning a fairness-improvement score to each candidate robber hex based on
  approximate changes in player production, and the agent can choose the robber
  move with the highest score.

Thus, the two-stage theory maps directly to a practical convex-optimization
driven policy for both placements and robber decisions.


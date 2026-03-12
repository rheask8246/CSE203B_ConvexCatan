## A Two-Stage Fairness-Aware Convex Model for Catan with Robber and Trading

This document merges and extends:

- `catan_convex_two_stage_with_robber.md`
- `catan_convex_two_stage_with_robber_and_trade.md`

It presents a unified convex framework with:

- **Stage 1**: a fair-allocation LP over relaxed placements,
- **Stage 2-R**: a fairness-aligned robber choice, and
- **Stage 2-T**: a fairness-aligned per-turn trading subproblem.

The notation follows the project outline and matches the implementation in
`CSE203B_ConvexCatan/agents/convex_solver.py` and `agents/players.py`.

---

### 1. Base Fair-Allocation Model (Stage 1)

Let:

- $L$ be the set of legal settlement locations (board intersections).
- $K = \{1,\dots,5\}$ index the five resource types.
- $P$ be the number of players, indexed by $p \in \{1,\dots,P\}$.

**Board model and expected production.**

For each location $l \in L$ and resource $k \in K$, the constant
$a_{lk} \ge 0$ denotes the expected production rate of resource $k$ at
location $l$, computed from dice probabilities and adjacent hex resource types.
In code, these coefficients form the matrix

- `A[l,k]` in `convex_solver._production_matrix(game)`.

**Decision variables.**

For each player $p$ and location $l$, we use a relaxed placement variable
$x_{p,l}$ with

- $0 \le x_{p,l} \le 1$ for all players $p$ and locations $l$,

which represents the extent to which player $p$ occupies location $l$.
In the actual game $x_{p,l} \in \{0,1\}$, but relaxing to $[0,1]$ yields a
convex problem. In code, all players’ variables are stacked into a matrix

- `x[p,l]` as a CVXPY variable of shape `(P, L)`.

**Constraints.**

The feasible region is given by affine (linear) constraints.

**Placement budget per player (initial phase)**  

Each player must place exactly two settlements:

- for all $p$:  
  $\displaystyle \sum_{l \in L} x_{p,l} = 2$.

In mid-game, this is relaxed to an inequality:

- for all $p$:  
  $\displaystyle \sum_{l \in L} x_{p,l} \le \text{budget}_p,$

where $\text{budget}_p$ depends on how many settlements/cities player $p$
already has.

**Location capacity constraint**  

Each location can be used by at most one player:

- for all $l$:  
  $\displaystyle \sum_{p=1}^P x_{p,l} \le 1$.

**Distance constraint (relaxed)**  

Let $E \subseteq L \times L$ be the set of adjacent location pairs. To enforce
the “no adjacent settlements” rule in relaxed form, impose:

- for all $(l,m) \in E$:  
  $\displaystyle \sum_{p=1}^P \bigl(x_{p,l} + x_{p,m}\bigr) \le 1$.

In code, `get_edges()` returns all adjacent pairs, and these constraints are
added as:

```python
cp.sum(x[:, u]) + cp.sum(x[:, v]) <= 1
```

**Occupied locations (mid-game)**  

If a node already contains a building, then no LP mass may be put there:

- for every already-occupied location $l$:  
  $\displaystyle \sum_p x_{p,l} = 0$.

This is implemented by `_occupied(game)` and an equality constraint for each
occupied node.

All of these constraints are linear, so they define a convex polyhedral set.

**Expected resource allocation and scores.**

For each player $p$, the expected resource production vector is

- for each resource $k \in K$:  
  $\mu_{p,k} = \sum_{l \in L} a_{l,k} \, x_{p,l}$.

In vector form, this is $\mu_p = A^\top x_p$. The scalar score is

- $v_p = \sum_{k \in K} w_k \, \mu_{p,k}$,

with fixed weights $w_k \ge 0$ (in code, `W = np.ones(N_RESOURCES)` so all
resources are weighted equally).

**Fairness objective.**

To promote fairness, introduce an auxiliary variable $t$ representing the
minimum expected production across players and impose the constraint

- for all $p$: $v_p \ge t$.

To encourage balanced access across resource types for each player, penalize
the within-player resource imbalance via a convex quadratic term. Define

- $\displaystyle \bar\mu_p = \frac{1}{|K|} \sum_{k \in K} \mu_{p,k}$ (the mean of player $p$’s resource vector),
- $\text{imbalance}_p = \|\mu_p - \bar\mu_p \mathbf{1}\|_2^2$.

The Stage 1 problem is:

maximize

$$
t \;-\; \lambda \sum_{p=1}^P \bigl\|\mu_p - \bar\mu_p \mathbf{1}\bigr\|_2^2,
$$

subject to the constraints above, where $\lambda \ge 0$ trades off fairness
against efficiency.

This objective is concave (linear term minus a convex quadratic), and all
constraints are linear, so the problem is a convex optimization problem with a
unique global optimum.

In `convex_solver.run_lp`, this is implemented via:

- `rho = x @ A` (our $\mu_p$),
- `v = rho @ W` (our $v_p$),
- `t` as a scalar CVXPY variable,
- `penalty = sum(cp.sum_squares(rho[p] - cp.sum(rho[p])/N_RESOURCES) for p in range(P))`,
- objective: `cp.Maximize(t - LAMBDA * penalty)`,
- constraints: budgets, capacity, distance, existing occupancy, and `t <= v[p]`.

The helper functions `solve_initial` and `solve_build` choose budgets, call
`run_lp`, and then extract node and edge scores that the agent uses greedily for
discrete placement.

---

### 2. Robber as Fixed Parameters (Robber-Aware Stage 1 Variant)

The robber blocks production on whichever hex it occupies: when a 7 is rolled
or a Knight is played, the active player chooses a hex; that hex no longer
produces resources until the robber moves again.

We first model the robber’s effect as a **fixed parameter** rather than a
decision variable, so that Stage 1 remains convex.

Let:

- $\rho_h \in [0,1]$ denote the fraction of time hex $h$ is (effectively) blocked by the robber.

We adjust the production coefficients to obtain a robber-aware matrix
$A^{\text{rob}} = [a_{lk}^{\text{rob}}]$:

$$
a^{\text{rob}}_{l,k} =
  \sum_{h \in \text{adj}(l)}
    P(\text{dice} = n_h) \cdot (1 - \rho_h) \cdot \mathbf{1}(\text{resource}(h)=k),
$$

where:

- $n_h$ is the dice number on hex $h$,
- $\text{adj}(l)$ is the set of hexes adjacent to intersection $l$,
- $\mathbf{1}(\text{resource}(h)=k)$ is 1 if hex $h$ produces resource $k$.

In this “robber-aware but not robber-optimizing” variant, the $\rho_h$ are
treated as **exogenous parameters**, not LP variables. Because they only appear
through $a^{\text{rob}}\_{l,k}$, the decision variables $x_{p,l}$ still enter
the objective and constraints linearly. We simply solve the same Stage 1 problem
with $A$ replaced by $A^{\text{rob}}$:

$$
\mu^{\text{rob}}_{p,k} = \sum_{l \in L} a^{\text{rob}}_{l,k} x_{p,l}, \quad
v^{\text{rob}}_p = \sum_{k \in K} w_k \mu^{\text{rob}}_{p,k},
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

and the same constraints on $x_{p,l}$ as before. For any fixed choice of
$(\rho_h)_h$, this remains a convex optimization problem.

In implementation terms, this corresponds to building a modified production
matrix `A_rob[l,k]` and passing it as an `A_override` argument into `run_lp`,
`solve_initial`, or `solve_build`, without changing the LP structure.

---

### 3. Two-Stage Fairness-Aligned Robber Optimization (Stage 2-R)

If we try to optimize over both $x_{p,l}$ and $\rho_h$ jointly, the
robber-adjusted coefficients become functions of $\rho$:

$$
a^{\text{rob}}_{l,k}(\rho) =
  \sum_{h \in \text{adj}(l)}
    P(\text{dice} = n_h) \cdot (1 - \rho_h) \cdot \mathbf{1}(\text{resource}(h)=k).
$$

Plugging into the expected production,

$$
\mu^{\text{rob}}_{p,k} =
  \sum_l a^{\text{rob}}_{l,k}(\rho) x_{p,l},
$$

we obtain terms of the form $(1-\rho_h) x_{p,l}$, i.e. products of two decision
variables. These bilinear terms make the objective and constraints non-convex
in $(x,\rho)$.

To keep the framework convex, we treat placements and robber as **separate but
coupled** convex problems by introducing a second stage.

**Stage 1 (recap)** produces:

- A fractional allocation $x^\star$,
- Expected production vectors $\mu_p^\star$,
- Scalar scores $v_p^\star$.

**Stage 2-R: Robber choice given placements.**

Given a fixed allocation pattern (either the fractional $x^\star$ or the
realized discrete game state), we can define a
**fairness-improvement score** for each candidate robber hex $h$.

Let:

- $v_p^\star$ be the baseline scores under the current allocations.
- For each hex $h$, define $v_p^{(h)}$ as the scores we would obtain if
  the robber is placed on hex $h$ and its production is blocked (approximated
  using expected contributions from that hex to each player).

A simple fairness metric is the range of player scores:

$$
\text{range} = \max_p v_p - \min_p v_p.
$$

We want the robber to shrink this range, pulling advantaged players closer
to the rest. For each hex $h$, define:

$$
c_h = \big(\max_p v_p^\star - \min_p v_p^\star\big)
      - \big(\max_p v_p^{(h)} - \min_p v_p^{(h)}\big).
$$

Here, $c_h > 0$ means that blocking $h$ reduces disparity; larger $c_h$
means a bigger fairness gain.

We now introduce robber decision variables $\rho_h$ on the simplex:

- $\rho_h \ge 0$ for all hexes $h$,
- $\sum_h \rho_h = 1$.

We then solve the robber LP:

$$
\max_{\rho} \quad \sum_h c_h \rho_h
\quad \text{subject to} \quad
\sum_h \rho_h = 1,\; \rho_h \ge 0 \ \forall h.
$$

This is a linear program in $\rho$ because the $c_h$ are constants computed
from the fixed allocations. In practice, the optimizer simply chooses the hex
with the largest $c_h$, i.e. the robber location that most improves fairness
according to the chosen metric:

$$
h^\star = \arg\max_h c_h.
$$

Stage 1 remains the original convex fair-allocation LP in $x_{p,l}$. Stage 2-R
is a convex (indeed linear) optimization over $\rho$ (or an argmax over $c_h$),
with $x^\star$ treated as data. At no point do we introduce bilinear products
like $x_{p,l}\rho_h$.

---

### 4. Stage 2-T: Fairness-Aligned Trading on a Single Turn

We now extend the framework with a trading component that is solved
per-turn for the acting player. This is Stage 2-T, a local convex problem
inspired by the single-player trading formulation.

Trading is modeled for a single player $p^\ast$ (e.g. our ConvexAgent) as a
separate LP or QP that runs on that player’s turn, given:

- The current hand of resources $q_r^0$.
- The available trade ratios from ports and the bank.
- The global fairness narrative, which we approximate with a local
fairness-aligned objective on the player’s post-trade resource hand.

We do not modify Stage 1; trading is a local decision that updates this turn’s
resource state and suggested builds.

#### 4.1 Trading variables and resource balance

Fix the acting player $p^\ast$. For each resource type $r$, introduce:

- $t^{\text{in}}_r \ge 0$: amount of resource $r$ traded in (received).
- $t^{\text{out}}_r \ge 0$: amount of resource $r$ traded out (given up).

Denote the current hand before the turn by $q_r^0$ for each resource $r$.
For this per-turn subproblem, we define:

- Post-trade quantity:

$$
q_r = q_r^0 + t^{\text{in}}_r - t^{\text{out}}_r
$$

- Non-negativity:

$$
q_r \ge 0 \quad \text{for all } r \quad (\text{cannot overspend}).
$$

These are linear in the decision variables $t^{\text{in}}_r, t^{\text{out}}_r$.

#### 4.2 Trade-ratio constraint

Let $\rho$ be the best trade ratio available to player $p^\ast$ this turn
(2:1, 3:1, or 4:1, based on ports and bank trades). We impose an aggregated
trade constraint:

$$
\sum_r t^{\text{in}}_r \;\le\; \frac{1}{\rho} \sum_r t^{\text{out}}_r.
$$

This captures the idea that, on average, the player must give up at least
$\rho$ cards to receive 1 card, aggregated across resources. More detailed
per-resource models are possible but this maintains linearity and simplicity.

#### 4.3 Fairness-aware objective for trading

To remain aligned with the global fairness objective but keep the per-turn
problem simple, we use a hand-level surrogate. One natural choice is:

- Encourage more total resources while penalizing imbalance across resources.

Let $q$ be the vector of post-trade quantities $(q_r)_r$. Define:

- Total resources: $\sum_r q_r$.
- Mean per-resource quantity: $\bar q = \frac{1}{|K|} \sum_r q_r$.
- Imbalance: $\text{imbalance} = \|q - \bar q \mathbf{1}\|_2^2$.

We can then solve:

$$
\max_{t^{\text{in}}, t^{\text{out}}} \;\; \alpha \sum_r q_r \;-\; \beta \, \|q - \bar q \mathbf{1}\|_2^2
$$

subject to:

- $q_r = q_r^0 + t^{\text{in}}_r - t^{\text{out}}_r$,
- $q_r \ge 0$ for all $r$,
- $\sum_r t^{\text{in}}_r \le \frac{1}{\rho} \sum_r t^{\text{out}}_r$,
- $t^{\text{in}}_r, t^{\text{out}}_r \ge 0$ for all $r$.

Here $\alpha,\beta \ge 0$ are tunable weights. The objective is concave
(linear term minus convex quadratic) and the constraints are linear, so this
per-turn trading problem is a convex optimization problem.

In code, `solve_trade_and_build` in `agents/convex_solver.py` implements a
special case of this idea:

- It reads `q0` from the game state, chooses $\rho$ from ports/bank, defines
  `t_in` and `t_out` as nonnegative CVXPY variables, sets
  `q = q0 + t_in - t_out`, and imposes:
  - `q >= 0`,
  - `sum(t_in) <= (1.0 / rho) * sum(t_out)`.
- It then maximizes
  `alpha * sum(q) - beta * sum_squares(q - q_mean)` with `q_mean = sum(q) / N_RESOURCES`,
  and returns the optimal `t_in` and `t_out` (or `None` if no solution).

This provides a **fairness-motivated local trading decision** for the agent.

---

### 5. Putting It Together: Three Stages per Turn

Conceptually, a fairness-aware ConvexAgent that handles placements, robber,
and trading within this convex framework operates as follows:

1. **Stage 1 (global fair placement LP)**  
   - Run periodically (e.g., at initial setup and periodically mid-game) to:
     - Compute or update fair allocation scores over $x_{p,l}$.
     - Produce node and edge scores for guiding build actions.
     - Maintain fairness metrics $\mu_p, v_p, t$.

2. **Stage 2-T (per-turn trading LP)**  
   - On each turn for player $p^\ast$, when trading is possible:
     - Solve the local trading problem with variables $t^{\text{in}}_r, t^{\text{out}}_r$
       (and optionally relaxed build variables), subject to:
       - Resource-balance constraints per resource $r$,
       - Aggregated trade-ratio constraint using the available ports and bank,
       - Optional per-turn action limits (e.g., at most one settlement, etc.).
     - Use a fairness-aligned scalar objective (e.g., maximize total resources
       minus hand imbalance) to find the best net trades.
     - Map the resulting `t_in`, `t_out` into concrete trade actions
       (e.g. MARITIME_TRADE) and choose builds consistent with the recommended
       resource usage.

3. **Stage 2-R (robber LP / argmax)**  
   - Whenever a robber move is required (after a 7 or Knight):
     - Use current allocations or game state to compute fairness-improvement
       scores $c_h$ per hex (e.g. reduction in range of $v_p$).
     - Solve the linear robber problem (or simply take $\arg\max_h c_h$) to
       select a fairness-aligned robber location.

Throughout, each stage is convex on its own:

- Stage 1: concave objective in $x_{p,l}, t$ with linear constraints.
- Stage 2-T: concave objective in $t^{\text{in}}_r, t^{\text{out}}_r$ with linear constraints.
- Stage 2-R: linear objective in $\rho_h$ with simplex constraints.

The stages are coupled only via constants computed from previous stages
($\mu_p, v_p, c_h, q_r^0$), never through bilinear products of decision
variables. This design keeps the full system within the scope of convex
optimization while modeling key aspects of Catan: resource allocation,
robber blocking, and trading.


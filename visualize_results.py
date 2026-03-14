"""Create fairness-focused plots for comparing Catan agents.

Since CONVEX does not optimise for winning, plots are organised around
fairness metrics (production Gini, VP Gini, Shannon entropy, max/min ratio)
rather than win rate. Win rate is still shown as a sanity check.

Example:
    python visualize_results.py --input-dir results/eval_20260306_120000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Catan fairness evaluation outputs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load(path: Path, index_col=None) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Warning: {path} not found — skipping dependent plots.")
        return None
    return pd.read_csv(path, index_col=index_col)


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _lineup_order_by(df: pd.DataFrame, col: str, asc: bool = True) -> list:
    """Sort lineups by median of col."""
    return (
        df.groupby("lineup")[col].median()
        .sort_values(ascending=asc)
        .index.tolist()
    )


def _agent_order_by(df: pd.DataFrame, col: str, asc: bool = False) -> list:
    return (
        df.groupby("agent")[col].mean()
        .sort_values(ascending=asc)
        .index.tolist()
    )


def _shannon_entropy(row: pd.Series) -> float:
    res_cols = ["wood_in_hand", "brick_in_hand", "sheep_in_hand", "wheat_in_hand", "ore_in_hand"]
    vals = row[res_cols].to_numpy(dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _add_entropy(players: pd.DataFrame) -> pd.DataFrame:
    needed = {"wood_in_hand", "brick_in_hand", "sheep_in_hand", "wheat_in_hand", "ore_in_hand"}
    if needed.issubset(players.columns):
        players = players.copy()
        players["shannon_entropy"] = players.apply(_shannon_entropy, axis=1)
    return players


# ---------------------------------------------------------------------------
# Consistent agent colour palette (colorblind-safe)
# ---------------------------------------------------------------------------

PALETTE = {
    "R":      "#6C757D",
    "GREEDY": "#2196F3",
    "MCTS":   "#FF9800",
    "WR":     "#9C27B0",
    "AB":     "#F44336",
    "VALUE":  "#4CAF50",
    "CONVEX": "#E91E63",
}

# Highlight CONVEX lineups differently from baselines.
def _lineup_color(lineup: str) -> str:
    return "#E91E63" if "CONVEX" in lineup else "#607D8B"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    analysis_dir = args.analysis_dir or (input_dir / "analysis")
    outdir = args.outdir or (input_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(input_dir / "player_metrics.csv")
    players = _coerce_numeric(players, [
        "prod_gini_final", "prod_max_min_ratio", "turns_until_equalized",
    ])
    players = _add_entropy(players)

    games = _safe_load(input_dir / "game_metrics.csv")
    if games is not None:
        games = _coerce_numeric(games, [
            "prod_gini_final", "prod_max_min_ratio", "turns_until_equalized", "vp_gini",
        ])

    agent_summary  = _safe_load(analysis_dir / "agent_summary.csv")
    seat_summary   = _safe_load(analysis_dir / "seat_summary.csv")
    pairwise       = _safe_load(analysis_dir / "pairwise_vp_diff.csv", index_col=0)
    fairness_summ  = _safe_load(analysis_dir / "fairness_summary.csv")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams.update({"savefig.dpi": 200, "figure.dpi": 120})

    def _agent_palette(agents):
        return {a: PALETTE.get(a, "#888") for a in agents}

    # ================================================================== #
    # PLOT 1 (sanity check): Win rate
    # NOTE: CONVEX is expected to have ~0% win rate. This plot contextualises
    # the fairness plots — CONVEX trades win rate for opponent equity.
    # ================================================================== #
    if agent_summary is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        df = agent_summary.sort_values("win_rate", ascending=False)
        x = np.arange(len(df))
        y = df["win_rate"].to_numpy()
        lo = y - df["win_rate_ci_low"].to_numpy()
        hi = df["win_rate_ci_high"].to_numpy() - y
        colors = [PALETTE.get(a, "#888") for a in df["agent"]]
        ax.bar(x, y, color=colors, alpha=0.85, edgecolor="white")
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#222",
                    capsize=5, linewidth=1.5)
        ax.axhline(1/3, color="gray", ls="--", lw=1,
                   label="Random baseline (1/3)")
        ax.set_xticks(x)
        ax.set_xticklabels(df["agent"])
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate by Agent (95% Bootstrap CI)\n"
                     "CONVEX optimises for fairness, not winning")
        ax.set_ylim(0, min(1.15, float(y.max()) * 1.3 + 0.05))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "01_win_rate_ci.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 2: Production Gini by lineup — PRIMARY fairness result
    # Lower = fairer. CONVEX lineups should be lower than equivalent baselines.
    # ================================================================== #
    if games is not None and "prod_gini_final" in games.columns \
            and games["prod_gini_final"].notna().any():
        fig, ax = plt.subplots(figsize=(13, 5))
        order = _lineup_order_by(games, "prod_gini_final", asc=True)
        colors = [_lineup_color(l) for l in order]
        sns.boxplot(
            data=games, x="lineup", y="prod_gini_final",
            order=order, palette=dict(zip(order, colors)),
            width=0.6, fliersize=2, linewidth=1.1, ax=ax,
        )
        ax.set_title(
            "Production Gini by Lineup  (primary LP fairness metric; lower = fairer)\n"
            "Pink = CONVEX lineup, grey = baseline"
        )
        ax.set_xlabel("Lineup")
        ax.set_ylabel("Production Gini (final turn)")
        plt.xticks(rotation=40, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / "02_prod_gini_by_lineup.png")
        plt.close(fig)
    else:
        print("Skipping production Gini plot — column missing or all NaN.")

    # ================================================================== #
    # PLOT 3: VP Gini by lineup
    # ================================================================== #
    if "vp_gini_game" in players.columns:
        game_vp = players.drop_duplicates("global_game_id")[
            ["lineup", "vp_gini_game"]
        ].copy()
        fig, ax = plt.subplots(figsize=(13, 5))
        order = _lineup_order_by(game_vp, "vp_gini_game", asc=True)
        colors = [_lineup_color(l) for l in order]
        sns.boxplot(
            data=game_vp, x="lineup", y="vp_gini_game",
            order=order, palette=dict(zip(order, colors)),
            width=0.6, fliersize=2, linewidth=1.1, ax=ax,
        )
        ax.set_title("VP Gini by Lineup  (lower = fairer)")
        ax.set_xlabel("Lineup")
        ax.set_ylabel("Game VP Gini")
        plt.xticks(rotation=40, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / "03_vp_gini_by_lineup.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 4: Shannon entropy of resource hand by AGENT
    # CONVEX should have higher entropy (more balanced hand) than others.
    # ================================================================== #
    if "shannon_entropy" in players.columns:
        fig, ax = plt.subplots(figsize=(11, 4))
        order = _agent_order_by(players, "shannon_entropy", asc=False)
        pal = _agent_palette(order)
        sns.boxplot(
            data=players, x="agent", y="shannon_entropy",
            order=order, palette=pal,
            width=0.55, fliersize=2, linewidth=1.1, ax=ax,
        )
        # Overlay mean markers.
        means = players.groupby("agent")["shannon_entropy"].mean()
        for i, ag in enumerate(order):
            if ag in means.index:
                ax.scatter(i, means[ag], marker="D", color="white",
                           edgecolor="#333", s=45, zorder=5,
                           label="Mean" if i == 0 else "")
        ax.set_title(
            "Shannon Entropy of Final Resource Hand by Agent\n"
            "Higher = more balanced resource portfolio  (CONVEX should be highest)"
        )
        ax.set_xlabel("Agent")
        ax.set_ylabel("Shannon Entropy (nats)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "04_shannon_entropy_by_agent.png")
        plt.close(fig)

        # ------------------------------------------------------------------ #
        # PLOT 4b: Mean Shannon entropy by lineup (does CONVEX improve others?)
        # ------------------------------------------------------------------ #
        if "lineup" in players.columns:
            fig, ax = plt.subplots(figsize=(13, 5))
            order_l = _lineup_order_by(players, "shannon_entropy", asc=False)
            colors = [_lineup_color(l) for l in order_l]
            sns.boxplot(
                data=players, x="lineup", y="shannon_entropy",
                order=order_l, palette=dict(zip(order_l, colors)),
                width=0.6, fliersize=2, linewidth=1.1, ax=ax,
            )
            ax.set_title(
                "Shannon Entropy of Resource Hand by Lineup\n"
                "Higher = more balanced hands across all players in that lineup"
            )
            ax.set_xlabel("Lineup")
            ax.set_ylabel("Shannon Entropy (nats)")
            plt.xticks(rotation=40, ha="right")
            fig.tight_layout()
            fig.savefig(outdir / "04b_shannon_entropy_by_lineup.png")
            plt.close(fig)
    else:
        print("Skipping Shannon entropy plots — resource columns not found.")

    # ================================================================== #
    # PLOT 5: Fairness summary bars — WITH vs WITHOUT CONVEX comparison
    # Each paired row shows a baseline lineup and its CONVEX equivalent.
    # ================================================================== #
    if fairness_summ is not None:
        metrics = [
            ("vp_gini_mean",   "vp_gini_std",   "Mean VP Gini"),
            ("prod_gini_mean", "prod_gini_std",  "Mean Production Gini"),
        ]
        avail = [(m, s, t) for m, s, t in metrics if m in fairness_summ.columns]
        if avail:
            n = len(avail)
            fig, axes = plt.subplots(1, n, figsize=(8 * n, max(5, len(fairness_summ) * 0.45 + 1)))
            if n == 1:
                axes = [axes]
            for ax, (col, std_col, title) in zip(axes, avail):
                df_f = fairness_summ.sort_values(col, ascending=True).copy()
                bar_colors = [_lineup_color(l) for l in df_f["lineup"]]
                bars = ax.barh(df_f["lineup"], df_f[col],
                               color=bar_colors, alpha=0.85, edgecolor="white")
                if std_col in df_f.columns:
                    ax.errorbar(
                        df_f[col], range(len(df_f)),
                        xerr=df_f[std_col].fillna(0),
                        fmt="none", ecolor="#333", capsize=3, linewidth=1,
                    )
                ax.set_xlabel(f"{title} (lower = fairer)")
                ax.set_title(title)
                ax.axvline(0, color="black", linewidth=0.5)
            fig.suptitle(
                "Fairness Metrics by Lineup  (pink = CONVEX lineup, grey = baseline)\n"
                "CONVEX lineups should have lower Gini than their baseline equivalents",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(outdir / "05_fairness_summary_bars.png")
            plt.close(fig)

    # ================================================================== #
    # PLOT 6: Pairwise VP differential heatmap
    # CONVEX row should show large negative values (loses VP to everyone) —
    # this is EXPECTED and shows it's not hoarding resources.
    # ================================================================== #
    if pairwise is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pairwise, annot=True, fmt=".1f",
            cmap="RdBu_r", center=0.0,
            linewidths=0.5, linecolor="#ddd",
            annot_kws={"size": 10}, ax=ax,
        )
        ax.set_title(
            "Pairwise VP Differential: E[VP(row) − VP(col)]\n"
            "CONVEX row should be negative (it cedes VP to regulate fairness)"
        )
        ax.set_xlabel("Opponent Agent")
        ax.set_ylabel("Row Agent")
        fig.tight_layout()
        fig.savefig(outdir / "06_pairwise_vp_diff_heatmap.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 7: Production Gini vs VP Gini scatter per lineup
    # Shows whether CONVEX improves BOTH axes simultaneously.
    # ================================================================== #
    if fairness_summ is not None \
            and "prod_gini_mean" in fairness_summ.columns \
            and "vp_gini_mean" in fairness_summ.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, row in fairness_summ.iterrows():
            color = _lineup_color(row["lineup"])
            ax.scatter(
                row["prod_gini_mean"], row["vp_gini_mean"],
                color=color, s=120, alpha=0.85, edgecolor="white", linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                row["lineup"],
                (row["prod_gini_mean"], row["vp_gini_mean"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7.5,
            )
        # Draw lines toward origin = better fairness on both axes.
        ax.set_xlabel("Mean Production Gini (lower = fairer)")
        ax.set_ylabel("Mean VP Gini (lower = fairer)")
        ax.set_title(
            "Production Gini vs VP Gini per Lineup\n"
            "Bottom-left = best fairness on both axes\n"
            "Pink = CONVEX lineup, grey = baseline"
        )
        # Reference: a perfectly fair game would be (0, 0).
        ax.annotate("← fairer", xy=(ax.get_xlim()[0], ax.get_ylim()[0] + 0.01),
                    fontsize=8, color="green")
        fig.tight_layout()
        fig.savefig(outdir / "07_prod_gini_vs_vp_gini_scatter.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 8: Max/min production ratio by lineup
    # Ratio = 1 is perfectly fair. CONVEX lineups should be closer to 1.
    # ================================================================== #
    if games is not None and "prod_max_min_ratio" in games.columns \
            and games["prod_max_min_ratio"].notna().any():
        fig, ax = plt.subplots(figsize=(13, 5))
        order = _lineup_order_by(games, "prod_max_min_ratio", asc=True)
        colors = [_lineup_color(l) for l in order]
        sns.boxplot(
            data=games, x="lineup", y="prod_max_min_ratio",
            order=order, palette=dict(zip(order, colors)),
            width=0.6, fliersize=2, linewidth=1.1, ax=ax,
        )
        ax.axhline(1.0, color="green", ls="--", lw=1, label="Perfect equality (ratio=1)")
        ax.set_title("Production Max/Min Ratio by Lineup  (lower = fairer, 1 = perfect equality)")
        ax.set_xlabel("Lineup")
        ax.set_ylabel("Max/Min Production Ratio")
        ax.legend(fontsize=9)
        plt.xticks(rotation=40, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / "08_prod_max_min_ratio_by_lineup.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 9: Final VP distribution (box + strip) — agent level
    # CONVEX expected to be lower. Shows how much VP it sacrifices.
    # ================================================================== #
    fig, ax = plt.subplots(figsize=(11, 4))
    order = _agent_order_by(players, "final_vp", asc=False)
    pal = _agent_palette(order)
    sns.boxplot(
        data=players, x="agent", y="final_vp", order=order,
        palette=pal, width=0.5, fliersize=2, linewidth=1.1, ax=ax,
    )
    sns.stripplot(
        data=players, x="agent", y="final_vp", order=order,
        palette=pal, alpha=0.1, jitter=True, size=2.5, ax=ax,
    )
    ax.axhline(10, color="black", ls="--", lw=0.9, label="Win threshold (10 VP)")
    ax.set_title("Final VP by Agent\n(CONVEX is expected to be lower — it optimises for opponent fairness)")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Final VP")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "09_final_vp_by_agent.png")
    plt.close(fig)

    # ================================================================== #
    # PLOT 10: Shannon entropy of CONVEX vs others — direct comparison
    # The key claim: CONVEX acquires a more balanced resource portfolio,
    # which in turn influences board placement more evenly.
    # ================================================================== #
    if "shannon_entropy" in players.columns:
        convex_entropy = players[players["agent"] == "CONVEX"]["shannon_entropy"].dropna()
        others_entropy = players[players["agent"] != "CONVEX"]["shannon_entropy"].dropna()

        if len(convex_entropy) > 0 and len(others_entropy) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Left: KDE comparison
            ax = axes[0]
            ax.hist(others_entropy, bins=20, density=True, alpha=0.5,
                    color="#607D8B", label="All other agents")
            ax.hist(convex_entropy, bins=20, density=True, alpha=0.6,
                    color="#E91E63", label="CONVEX")
            ax.set_xlabel("Shannon Entropy (nats)")
            ax.set_ylabel("Density")
            ax.set_title("Resource Hand Entropy: CONVEX vs Others")
            ax.legend()

            # Right: Mean entropy by agent as horizontal bars
            ax = axes[1]
            means = players.groupby("agent")["shannon_entropy"].mean().sort_values()
            bar_colors = [PALETTE.get(a, "#888") for a in means.index]
            ax.barh(means.index, means.values, color=bar_colors,
                    alpha=0.85, edgecolor="white")
            ax.set_xlabel("Mean Shannon Entropy (nats)")
            ax.set_title("Mean Resource Entropy by Agent\n(higher = more balanced hand)")
            ax.axvline(means.get("CONVEX", 0), color="#E91E63", ls="--",
                       lw=1.2, label="CONVEX mean")
            ax.legend(fontsize=9)

            fig.suptitle("Shannon Entropy of Final Resource Hand", fontsize=12)
            fig.tight_layout()
            fig.savefig(outdir / "10_shannon_entropy_comparison.png")
            plt.close(fig)

    # ================================================================== #
    # PLOT 11: Turn-order sensitivity (sanity check)
    # ================================================================== #
    if seat_summary is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        agents_present = seat_summary["agent"].unique().tolist()
        pal = _agent_palette(agents_present)
        sns.lineplot(
            data=seat_summary, x="turn_order", y="win_rate",
            hue="agent", palette=pal, marker="o", linewidth=1.8, ax=ax,
        )
        turn_orders = sorted(seat_summary["turn_order"].dropna().unique().astype(int))
        ax.set_xticks(turn_orders)
        ax.set_xlabel("Turn Order (0 = first)")
        ax.set_ylabel("Win Rate")
        ax.set_title("Turn-Order Effect on Win Rate")
        fig.tight_layout()
        fig.savefig(outdir / "11_turn_order_effect.png")
        plt.close(fig)

    # ================================================================== #
    # PLOT 12: Turns until equalization (if tracked)
    # ================================================================== #
    if games is not None and "turns_until_equalized" in games.columns:
        eq = games[games["turns_until_equalized"] >= 0].copy()
        if not eq.empty:
            fig, ax = plt.subplots(figsize=(13, 4))
            order = (
                eq.groupby("lineup")["turns_until_equalized"]
                .median().sort_values().index.tolist()
            )
            colors = [_lineup_color(l) for l in order]
            sns.boxplot(
                data=eq, x="lineup", y="turns_until_equalized",
                order=order, palette=dict(zip(order, colors)),
                width=0.6, fliersize=2, ax=ax,
            )
            ax.set_title(
                "Turns Until Production Equalized (Gini < 0.10)\n"
                "Lower = CONVEX equalises the board faster"
            )
            ax.set_xlabel("Lineup")
            ax.set_ylabel("Turn")
            plt.xticks(rotation=40, ha="right")
            fig.tight_layout()
            fig.savefig(outdir / "12_turns_until_equalized.png")
            plt.close(fig)
        else:
            print("Skipping equalization plot — no games reached threshold.")

    # ================================================================== #
    # PLOT 13: Composite fairness dashboard (paper figure)
    # 2x2 summary: prod Gini / VP Gini / entropy / max-min ratio
    # One bar per lineup, CONVEX highlighted.
    # ================================================================== #
    if fairness_summ is not None:
        needed = ["prod_gini_mean", "vp_gini_mean"]
        if all(c in fairness_summ.columns for c in needed):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            plots = [
                ("prod_gini_mean",        "prod_gini_std",        "Mean Production Gini",       True),
                ("vp_gini_mean",          "vp_gini_std",          "Mean VP Gini",               True),
                ("prod_max_min_mean",     None,                   "Mean Prod Max/Min Ratio",    True),
                ("turns_equalized_mean",  None,                   "Mean Turns Until Equalized", True),
            ]
            for ax, (col, std_col, title, lower_better) in zip(axes.flat, plots):
                if col not in fairness_summ.columns:
                    ax.set_visible(False)
                    continue
                df_f = fairness_summ.dropna(subset=[col]).sort_values(col, ascending=True)
                bar_colors = [_lineup_color(l) for l in df_f["lineup"]]
                ax.barh(df_f["lineup"], df_f[col],
                        color=bar_colors, alpha=0.85, edgecolor="white")
                if std_col and std_col in df_f.columns:
                    ax.errorbar(
                        df_f[col], range(len(df_f)),
                        xerr=df_f[std_col].fillna(0),
                        fmt="none", ecolor="#333", capsize=3,
                    )
                note = " (↓ fairer)" if lower_better else ""
                ax.set_xlabel(title + note)
                ax.set_title(title)
            fig.suptitle(
                "Composite Fairness Dashboard\n"
                "Pink bars = CONVEX lineup  |  Grey bars = baseline",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(outdir / "13_composite_fairness_dashboard.png")
            plt.close(fig)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
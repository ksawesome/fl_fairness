from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def category_from_condition(condition: str) -> str:
    if condition == "vanilla":
        return "baseline"
    if condition.endswith("_only"):
        return "attack_only"
    return "attack_defense"


def plot_baseline_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    split_order = ["iid", "dirichlet_0_5", "strict_non_iid"]
    opt_order = ["sgd", "adamw", "fedprox"]

    acc = (
        df.pivot(index="split_name", columns="optimizer_name", values="final_val_accuracy")
        .reindex(index=split_order, columns=opt_order)
    )
    eod = (
        df.pivot(index="split_name", columns="optimizer_name", values="final_val_equalized_odds_gap")
        .reindex(index=split_order, columns=opt_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    sns.heatmap(acc, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=axes[0])
    sns.heatmap(eod, annot=True, fmt=".3f", cmap="YlOrRd", cbar=False, ax=axes[1])

    axes[0].set_title("Baseline Accuracy")
    axes[1].set_title("Baseline EO Gap")
    for ax in axes:
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Split")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_attack_only_summary(df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        df.groupby("attack", as_index=False)[
            ["final_val_accuracy", "final_val_equalized_odds_gap", "final_val_demographic_parity_gap"]
        ]
        .mean()
        .sort_values("attack")
    )

    x = np.arange(len(grouped))
    w = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(x - w, grouped["final_val_accuracy"], width=w, label="Accuracy")
    ax.bar(x, grouped["final_val_equalized_odds_gap"], width=w, label="EO gap")
    ax.bar(x + w, grouped["final_val_demographic_parity_gap"], width=w, label="DP gap")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["attack"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Attack-only mean metrics")
    ax.legend(ncol=3, fontsize=8)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_attack_defense_matrix(df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        df.groupby(["attack", "defense"], as_index=False)[
            ["final_val_accuracy", "final_val_equalized_odds_gap"]
        ]
        .mean()
    )

    attack_order = ["A1", "A2", "A3", "A4", "A5"]
    defense_order = ["D1", "D2", "D3", "D4", "D5"]

    acc = (
        agg.pivot(index="attack", columns="defense", values="final_val_accuracy")
        .reindex(index=attack_order, columns=defense_order)
    )
    eod = (
        agg.pivot(index="attack", columns="defense", values="final_val_equalized_odds_gap")
        .reindex(index=attack_order, columns=defense_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    sns.heatmap(acc, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=axes[0])
    sns.heatmap(eod, annot=True, fmt=".3f", cmap="YlOrRd", cbar=False, ax=axes[1])

    axes[0].set_title("Attack-defense Accuracy")
    axes[1].set_title("Attack-defense EO gap")
    for ax in axes:
        ax.set_xlabel("Defense")
        ax.set_ylabel("Attack")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_data_distribution(df: pd.DataFrame, out_path: Path) -> None:
    split_order = ["iid", "dirichlet_0_5", "strict_non_iid"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)

    legend_handles = None
    legend_labels = None

    for ax, split in zip(axes, split_order):
        d = df[df["split_name"] == split].sort_values("client_id")
        client = d["client_id"].to_numpy()
        ax2 = ax.twinx()

        ax.bar(client, d["samples"], alpha=0.35, color="#4C78A8", label="samples")
        ax2.plot(client, d["label_positive_rate"], marker="o", color="#F58518", linewidth=1.5, label="label+")
        ax2.plot(client, d["protected_positive_rate"], marker="s", color="#54A24B", linewidth=1.5, label="protected+")

        ax.set_title(split)
        ax.set_xlabel("Client")
        ax.set_ylabel("Samples")
        ax2.set_ylabel("Rate")
        ax2.set_ylim(0, 1)

        if legend_handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            legend_handles = h1 + h2
            legend_labels = l1 + l2

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, ncol=3, loc="upper center", fontsize=8)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return nondominated points for max-accuracy / min-EO objectives."""
    pts = df[["final_val_accuracy", "final_val_equalized_odds_gap"]].to_numpy()
    keep = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        if not keep[i]:
            continue
        ai, ei = pts[i]
        dominated = (
            (pts[:, 0] >= ai)
            & (pts[:, 1] <= ei)
            & ((pts[:, 0] > ai) | (pts[:, 1] < ei))
        )
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False

    front = df.loc[keep].copy()
    return front.sort_values("final_val_accuracy")


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    d["category"] = d["condition_name"].apply(category_from_condition)

    palette = {
        "baseline": "#4C78A8",
        "attack_only": "#F58518",
        "attack_defense": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for cat, sub in d.groupby("category"):
        ax.scatter(
            sub["final_val_accuracy"],
            sub["final_val_equalized_odds_gap"],
            s=26,
            alpha=0.82,
            label=cat,
            color=palette[cat],
            edgecolor="white",
            linewidth=0.3,
        )

    # Annotate best-accuracy point and worst-fairness point.
    best_acc = d.loc[d["final_val_accuracy"].idxmax()]
    worst_eod = d.loc[d["final_val_equalized_odds_gap"].idxmax()]

    ax.annotate(
        best_acc["condition_name"],
        (best_acc["final_val_accuracy"], best_acc["final_val_equalized_odds_gap"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=7,
    )
    ax.annotate(
        worst_eod["condition_name"],
        (worst_eod["final_val_accuracy"], worst_eod["final_val_equalized_odds_gap"]),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=7,
    )

    ax.set_xlabel("Final validation accuracy")
    ax.set_ylabel("Final EO gap")
    ax.set_title("Accuracy-fairness trade-off (all 189 runs)")
    ax.legend(title="Category", fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_optimizer_pareto(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    d["category"] = d["condition_name"].apply(category_from_condition)
    d = d[d["final_val_accuracy"] > 0.55].copy()

    opt_order = ["sgd", "adamw", "fedprox"]
    palette = {
        "baseline": "#4C78A8",
        "attack_only": "#F58518",
        "attack_defense": "#54A24B",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True, sharey=True)

    for ax, opt in zip(axes, opt_order):
        sub = d[d["optimizer_name"] == opt].copy()
        if sub.empty:
            ax.set_title(opt)
            ax.set_xlabel("Accuracy")
            continue

        for cat, cat_df in sub.groupby("category"):
            ax.scatter(
                cat_df["final_val_accuracy"],
                cat_df["final_val_equalized_odds_gap"],
                s=22,
                alpha=0.75,
                color=palette[cat],
                label=cat,
                edgecolor="white",
                linewidth=0.3,
            )

        front = pareto_front(sub)
        ax.plot(
            front["final_val_accuracy"],
            front["final_val_equalized_odds_gap"],
            color="black",
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            label="Pareto front",
        )

        ax.set_title(opt.upper())
        ax.set_xlabel("Accuracy")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("EO gap")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_a4_dynamics(df_rounds: pd.DataFrame, out_path: Path) -> None:
    d = df_rounds[df_rounds["condition_name"] == "A4_only"].copy()
    d["round_idx"] = d["round_idx"].astype(int)
    d["val_accuracy"] = d["val_accuracy"].astype(float)

    split_order = ["iid", "dirichlet_0_5", "strict_non_iid"]
    opt_order = ["sgd", "adamw", "fedprox"]
    colors = {"sgd": "#4C78A8", "adamw": "#F58518", "fedprox": "#54A24B"}

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True, sharey=True)

    for ax, split in zip(axes, split_order):
        split_df = d[d["split_name"] == split]
        for opt in opt_order:
            sub = split_df[split_df["optimizer_name"] == opt].sort_values("round_idx")
            if sub.empty:
                continue
            ax.plot(
                sub["round_idx"],
                sub["val_accuracy"],
                linewidth=1.7,
                color=colors[opt],
                label=opt,
            )
        ax.set_title(split)
        ax.set_xlabel("Round")
        ax.set_xlim(1, 30)
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Validation accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    results = root.parent / "results"
    fig_dir = root / "figures"
    ensure_dir(fig_dir)

    baseline = pd.read_csv(results / "full_run_20260411_150759_table2_baseline.csv")
    attack_only = pd.read_csv(results / "full_run_20260411_150759_fig1_fig3_fig5_attack_only.csv")
    attack_def = pd.read_csv(results / "full_run_20260411_150759_table1_fig4_attack_defense.csv")
    dist = pd.read_csv(results / "full_run_20260411_150759_fig2_data_distribution.csv")
    scatter = pd.read_csv(results / "full_run_20260411_150759_fig6_scatter_source.csv")
    all_rounds = pd.read_csv(results / "full_run_20260411_150759_all_rounds.csv")

    plot_baseline_heatmaps(baseline, fig_dir / "fig_baseline_heatmaps.png")
    plot_attack_only_summary(attack_only, fig_dir / "fig_attack_only_summary.png")
    plot_attack_defense_matrix(attack_def, fig_dir / "fig_attack_defense_matrix.png")
    plot_data_distribution(dist, fig_dir / "fig_data_distribution.png")
    plot_scatter(scatter, fig_dir / "fig_scatter.png")
    plot_optimizer_pareto(scatter, fig_dir / "fig_optimizer_pareto.png")
    plot_a4_dynamics(all_rounds, fig_dir / "fig_a4_dynamics.png")

    print("Generated figures in:", fig_dir)


if __name__ == "__main__":
    main()

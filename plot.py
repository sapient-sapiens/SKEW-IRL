from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval import compute_log_likelihood_ratio, counterfactual_policy
from formalize import (
    FEATURE_NAMES,
    STATE_FEATURE_INDEX,
    cluster_states,
    compute_action_midpoints,
    discretize_actions,
)

DATA_PATH = Path("./skew_project_data.csv")
RESULT_PATH = Path("./irl_result.npz")
DEFAULT_OUTPUT = Path("./irl_summary.png")


def _ensure_inputs() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Run data.py before plotting."
        )
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Missing IRL artifact at {RESULT_PATH}. Run irl.py before plotting."
        )


def load_context() -> Dict[str, Any]:
    _ensure_inputs()
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])
    data.index.name = "Date"
    states, centroids = cluster_states(data)
    actions = discretize_actions(data)
    results = np.load(RESULT_PATH, allow_pickle=True)
    context = {
        "data": data,
        "states": states,
        "centroids": centroids,
        "actions": actions,
        "theta": results["theta"],
        "policy": results["policy"],
        "mu_E": results["mu_E"],
        "mu_theta": results["mu_theta"],
        "feature_scale": results["feature_scale"],
        "action_midpoints": results["action_midpoints"],
        "feature_names": results["feature_names"].tolist(),
    }
    return context


def action_distributions(
    actions: np.ndarray, states: np.ndarray, policy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_actions = policy.shape[1]
    empirical = np.bincount(actions, minlength=n_actions)
    empirical = empirical / empirical.sum()
    avg_policy = policy[states].mean(axis=0)
    avg_policy = avg_policy / avg_policy.sum()
    return empirical, avg_policy


def build_counterfactual(
    theta: np.ndarray,
    action_midpoints: np.ndarray,
    centroids: np.ndarray,
    feature_scale: np.ndarray,
) -> Dict[str, Any]:
    credit_max = centroids[:, STATE_FEATURE_INDEX["Credit_Spread"]].max()
    base_centroid = centroids.mean(axis=0)
    overrides = {"Credit_Spread": credit_max, "VIX": 12.0}
    scenario = counterfactual_policy(
        theta,
        action_midpoints,
        base_centroid,
        overrides,
        feature_scale,
    )
    scenario["overrides"] = overrides
    return scenario


def plot_results(context: Dict[str, Any], output_path: Path = DEFAULT_OUTPUT) -> None:
    feature_names = context["feature_names"] or FEATURE_NAMES
    theta = context["theta"]
    mu_E = context["mu_E"]
    mu_theta = context["mu_theta"]
    policy = context["policy"]
    states = context["states"]
    actions = context["actions"]
    action_midpoints = context["action_midpoints"]

    emp_actions, policy_actions = action_distributions(actions, states, policy)
    llr = compute_log_likelihood_ratio(states, actions, policy)
    scenario = build_counterfactual(
        theta,
        action_midpoints,
        context["centroids"],
        context["feature_scale"],
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SKEW IRL Diagnostics", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(feature_names, theta, color="#2b8cbe")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Learned Reward Weights θ")
    ax.set_ylabel("weight")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    ax = axes[0, 1]
    indices = np.arange(len(feature_names))
    width = 0.35
    ax.bar(indices - width / 2, mu_E, width, label="Expert", color="#238b45")
    ax.bar(indices + width / 2, mu_theta, width, label="Model", color="#a50f15")
    ax.set_xticks(indices)
    ax.set_xticklabels(feature_names, rotation=20)
    ax.set_title("Feature Expectation Match")
    ax.set_ylabel("normalized expectation")
    ax.legend()

    ax = axes[1, 0]
    action_labels = [f"Bin {i}\n≈{m:.0f}" for i, m in enumerate(action_midpoints)]
    ax.bar(
        np.arange(len(emp_actions)) - width / 2,
        emp_actions,
        width,
        label="Empirical",
        color="#636363",
    )
    ax.bar(
        np.arange(len(policy_actions)) + width / 2,
        policy_actions,
        width,
        label="Policy-Implied",
        color="#08519c",
    )
    ax.set_xticks(np.arange(len(action_labels)))
    ax.set_xticklabels(action_labels)
    ax.set_ylabel("probability")
    ax.set_title("Action Distribution Comparison")
    ax.legend()
    ax.text(
        0.02,
        0.92,
        f"LLR vs uniform: {llr:,.0f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 3},
    )

    ax = axes[1, 1]
    scenario_probs = scenario["probs"]
    ax.bar(np.arange(len(scenario_probs)), scenario_probs, color="#d94801")
    ax.set_xticks(np.arange(len(action_labels)))
    ax.set_xticklabels(action_labels)
    overrides_str = ", ".join(f"{k}={v:.1f}" for k, v in scenario["overrides"].items())
    ax.set_title(f"Counterfactual Policy ({overrides_str})")
    ax.set_ylabel("probability")
    best_mid = scenario["best_action_midpoint"]
    ax.text(
        0.02,
        0.92,
        f"Best action midpoint ≈ {best_mid:.1f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 3},
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=300)
    print(f"Wrote plot to {output_path.resolve()}")


def main(output_path: Path = DEFAULT_OUTPUT) -> None:
    context = load_context()
    plot_results(context, output_path)


if __name__ == "__main__":
    main()


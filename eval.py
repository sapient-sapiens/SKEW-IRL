import numpy as np
import pandas as pd
from scipy.special import logsumexp

from formalize import (
    cluster_states,
    discretize_actions,
    compute_action_midpoints,
    build_state_action_features,
    FEATURE_NAMES,
    STATE_FEATURE_INDEX,
)


def load_results(path: str = "irl_result.npz"):
    return np.load(path, allow_pickle=True)


def compute_log_likelihood_ratio(states, actions, policy):
    n_actions = policy.shape[1]
    eps = 1e-12
    log_pi = np.log(policy[states, actions] + eps).sum()
    log_baseline = len(actions) * np.log(1.0 / n_actions)
    return log_pi - log_baseline


def interpret_weights(theta):
    dominant_idx = np.argmax(np.abs(theta))
    summary = {
        "theta": theta,
        "dominant_feature": FEATURE_NAMES[dominant_idx],
        "dominant_weight": theta[dominant_idx],
        "credit_spread_weight": theta[FEATURE_NAMES.index("crash_phobia")],
    }
    return summary


def counterfactual_policy(theta, action_midpoints, base_centroid, overrides, feature_scale):
    scenario = base_centroid.copy()
    for key, value in overrides.items():
        idx = STATE_FEATURE_INDEX[key]
        scenario[idx] = value
    scenario_features = build_state_action_features(
        np.expand_dims(scenario, axis=0),
        action_midpoints,
    )[0]
    scenario_features = scenario_features / feature_scale
    rewards = scenario_features @ theta
    probs = np.exp(rewards - logsumexp(rewards))
    best_action = np.argmax(probs)
    return {
        "probs": probs,
        "best_action": best_action,
        "best_action_midpoint": action_midpoints[best_action],
    }


def evaluate():
    data = pd.read_csv("./skew_project_data.csv")
    states, centroids = cluster_states(data)
    actions = discretize_actions(data)
    action_midpoints = compute_action_midpoints(data)
    results = load_results()
    theta = results["theta"]
    policy = results["policy"]
    feature_scale = results["feature_scale"]

    llr = compute_log_likelihood_ratio(states, actions, policy)
    weight_summary = interpret_weights(theta)

    credit_max = data["Credit_Spread"].max()
    base_centroid = centroids.mean(axis=0)
    scenario = counterfactual_policy(
        theta,
        action_midpoints,
        base_centroid,
        {
            "Credit_Spread": credit_max,
            "VIX": 12.0,
        },
        feature_scale,
    )

    print("Log-likelihood ratio vs uniform:", llr)
    print("Dominant feature:", weight_summary["dominant_feature"])
    print("Credit spread weight:", weight_summary["credit_spread_weight"])
    print("Counterfactual best action midpoint:", scenario["best_action_midpoint"])
    print("Counterfactual action probabilities:", scenario["probs"])


if __name__ == "__main__":
    evaluate()


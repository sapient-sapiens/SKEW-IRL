import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from scipy.special import logsumexp

from formalize import (
    cluster_states,
    discretize_actions,
    transition,
    action_conditioned_transition,
    compute_action_midpoints,
    build_state_action_features,
    compute_feature_scale,
    normalize_feature_tensor,
    compute_expert_feature_expectations,
    FEATURE_NAMES,
    n_clusters,
)


@dataclass
class MaxEntIRLConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    n_iters: int = 1000
    soft_vi_tol: float = 1e-6
    max_soft_vi_iters: int = 500
    grad_clip: float = 1e4


@dataclass
class MaxEntIRLResult:
    theta: np.ndarray
    policy: np.ndarray
    mu_E: np.ndarray
    mu_theta: np.ndarray
    history: Dict[str, list]


def prepare_dataset(data_path: str) -> Dict[str, Any]:
    data = pd.read_csv(data_path)
    states, centroids = cluster_states(data)
    actions = discretize_actions(data)
    assert len(states) == len(actions)
    trans_matrix = transition(states)
    trans_tensor = action_conditioned_transition(states, actions)
    action_midpoints = compute_action_midpoints(data)
    feature_tensor = build_state_action_features(centroids, action_midpoints)
    feature_scale = compute_feature_scale(feature_tensor)
    feature_tensor = normalize_feature_tensor(feature_tensor, feature_scale)
    mu_E = compute_expert_feature_expectations(states, actions, feature_tensor)
    start_dist = np.zeros(n_clusters)
    start_dist[states[0]] = 1.0
    return {
        "data": data,
        "states": states,
        "actions": actions,
        "centroids": centroids,
        "transition_matrix": trans_matrix,
        "transition_tensor": trans_tensor,
        "action_midpoints": action_midpoints,
        "feature_tensor": feature_tensor,
        "feature_scale": feature_scale,
        "mu_E": mu_E,
        "start_dist": start_dist,
        "horizon": len(states),
    }


def compute_rewards(theta: np.ndarray, feature_tensor: np.ndarray) -> np.ndarray:
    return np.tensordot(feature_tensor, theta, axes=[2, 0])


def soft_value_iteration(
    transition_tensor: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    tol: float,
    max_iters: int,
):
    n_states = transition_tensor.shape[0]
    V = np.zeros(n_states)
    for _ in range(max_iters):
        continuation = np.tensordot(transition_tensor, V, axes=[2, 0])
        Q = rewards + gamma * continuation
        new_V = logsumexp(Q, axis=1)
        if np.max(np.abs(new_V - V)) < tol:
            V = new_V
            break
        V = new_V
    continuation = np.tensordot(transition_tensor, V, axes=[2, 0])
    Q = rewards + gamma * continuation
    policy = np.exp(Q - V[:, None])
    return V, Q, policy


def compute_state_visitation(
    transition_tensor: np.ndarray,
    policy: np.ndarray,
    start_dist: np.ndarray,
    horizon: int,
) -> np.ndarray:
    n_states = transition_tensor.shape[0]
    visitation = np.zeros((horizon, n_states))
    visitation[0] = start_dist
    for t in range(1, horizon):
        flows = visitation[t - 1][:, None, None] * policy[:, :, None] * transition_tensor
        visitation[t] = flows.sum(axis=(0, 1))
    return visitation


def compute_model_feature_expectations(
    policy: np.ndarray,
    visitation: np.ndarray,
    feature_tensor: np.ndarray,
    horizon: int,
) -> np.ndarray:
    state_visits = visitation.sum(axis=0)
    state_action_weights = state_visits[:, None] * policy
    expectations = (state_action_weights[:, :, None] * feature_tensor).sum(axis=(0, 1))
    return expectations / horizon


def maxent_irl(
    feature_tensor: np.ndarray,
    transition_tensor: np.ndarray,
    mu_E: np.ndarray,
    start_dist: np.ndarray,
    horizon: int,
    config: MaxEntIRLConfig,
) -> MaxEntIRLResult:
    theta = np.zeros(feature_tensor.shape[2])
    history = {"grad_norm": [], "mu_theta": []}
    policy = np.full((feature_tensor.shape[0], feature_tensor.shape[1]), 1.0 / feature_tensor.shape[1])
    mu_theta = np.zeros_like(mu_E)

    for _ in range(config.n_iters):
        rewards = compute_rewards(theta, feature_tensor)
        _, _, policy = soft_value_iteration(
            transition_tensor,
            rewards,
            config.gamma,
            config.soft_vi_tol,
            config.max_soft_vi_iters,
        )
        visitation = compute_state_visitation(transition_tensor, policy, start_dist, horizon)
        mu_theta = compute_model_feature_expectations(policy, visitation, feature_tensor, horizon)
        gradient = mu_E - mu_theta
        grad_norm = np.linalg.norm(gradient)
        history["grad_norm"].append(grad_norm)
        history["mu_theta"].append(mu_theta.copy())
        if config.grad_clip is not None and grad_norm > config.grad_clip:
            gradient = gradient * (config.grad_clip / grad_norm)
        theta += config.lr * gradient
        if grad_norm < config.soft_vi_tol:
            break

    return MaxEntIRLResult(theta=theta, policy=policy, mu_E=mu_E, mu_theta=mu_theta, history=history)


def main():
    dataset = prepare_dataset("./skew_project_data.csv")
    config = MaxEntIRLConfig()
    result = maxent_irl(
        dataset["feature_tensor"],
        dataset["transition_tensor"],
        dataset["mu_E"],
        dataset["start_dist"],
        dataset["horizon"],
        config,
    )
    np.savez(
        "irl_result.npz",
        theta=result.theta,
        policy=result.policy,
        mu_E=result.mu_E,
        mu_theta=result.mu_theta,
        feature_names=np.array(FEATURE_NAMES),
        action_midpoints=dataset["action_midpoints"],
        feature_scale=dataset["feature_scale"],
    )
    print("Trained theta:", result.theta)
    print("Final grad norm:", result.history["grad_norm"][-1])


if __name__ == "__main__":
    main()


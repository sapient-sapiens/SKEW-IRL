import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
import numpy as np 

n_clusters = 30 
n_actions = 10 
STATE_FEATURES = ["VIX","SPX","Credit_Spread","SPX_Return_1m","SKEW_Lag1"]
STATE_FEATURE_INDEX = {feature: idx for idx, feature in enumerate(STATE_FEATURES)}
FEATURE_NAMES = [
    "mean_reversion",
    "vix_substitution",
    "momentum",
    "crash_phobia",
    "action_cost",
]
SKEW_MEAN_TARGET = 125.0

def cluster_states(data, n_clusters = n_clusters):
    state_space = data[STATE_FEATURES].to_numpy()
    scaler = StandardScaler() 
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
    state_space = scaler.fit_transform(state_space)
    kmeans.fit(state_space)
    labels = kmeans.labels_
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    print("Cluster id of the first 20 states:", labels[:20])
    return labels, centroids 

def discretize_actions(data, n_actions=n_actions, window = 252):
    rolling_rank = data["SKEW"].rolling(window = window).rank(pct = True) 
    rolling_rank = rolling_rank.fillna(0.5) 
    bins = np.floor(rolling_rank* n_actions) 
    bins = np.clip(bins, 0, n_actions - 1).astype(int).values
    print(bins.shape)
    print("Actions in the first twenty states:", bins[:20])
    return bins 
def transition(sequence, n_states=n_clusters): 
    matrix = np.ones((n_states, n_states))
    for prev_state, next_state in zip(sequence[:-1], sequence[1:]): 
        matrix[prev_state, next_state] += 1 
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums 

def action_conditioned_transition(states, actions, n_states=n_clusters, n_actions=n_actions):
    tensor = np.ones((n_states, n_actions, n_states))
    for curr_state, action, next_state in zip(states[:-1], actions[:-1], states[1:]):
        tensor[curr_state, action, next_state] += 1 
    row_sums = tensor.sum(axis=2, keepdims=True)
    return tensor / row_sums

def compute_action_midpoints(data, n_actions=n_actions):
    skew_values = data["SKEW"].dropna().to_numpy()
    skew_min, skew_max = skew_values.min(), skew_values.max()
    edges = np.linspace(skew_min, skew_max, n_actions + 1)
    return (edges[:-1] + edges[1:]) / 2

def build_state_action_features(centroids, action_midpoints, skew_target=SKEW_MEAN_TARGET):
    n_states = centroids.shape[0]
    n_actions_local = action_midpoints.shape[0]
    features = np.zeros((n_states, n_actions_local, len(FEATURE_NAMES)))
    vix_idx = STATE_FEATURE_INDEX["VIX"]
    credit_idx = STATE_FEATURE_INDEX["Credit_Spread"]
    return_idx = STATE_FEATURE_INDEX["SPX_Return_1m"]
    action_mean = action_midpoints.mean()
    for state_idx, centroid in enumerate(centroids):
        vix = centroid[vix_idx]
        ret = centroid[return_idx]
        credit = centroid[credit_idx]
        for action_idx, action_mid in enumerate(action_midpoints):
            features[state_idx, action_idx, 0] = -((action_mid - skew_target) ** 2)
            features[state_idx, action_idx, 1] = action_mid * vix
            features[state_idx, action_idx, 2] = action_mid * ret
            features[state_idx, action_idx, 3] = action_mid * credit
            features[state_idx, action_idx, 4] = -abs(action_mid - action_mean)
    return features

def compute_feature_scale(feature_tensor, eps=1e-6):
    flat = feature_tensor.reshape(-1, feature_tensor.shape[-1])
    scale = flat.std(axis=0)
    scale[scale < eps] = eps
    return scale

def normalize_feature_tensor(feature_tensor, scale):
    return feature_tensor / scale

def sample_feature_trajectory(states, actions, feature_tensor):
    return feature_tensor[states, actions]

def compute_expert_feature_expectations(states, actions, feature_tensor):
    trajectory_features = sample_feature_trajectory(states, actions, feature_tensor)
    return trajectory_features.mean(axis=0)

if __name__ == "__main__": 
    data = pd.read_csv("./skew_project_data.csv")
    labels, centroids = cluster_states(data)
    actions = discretize_actions(data) 
    assert labels.shape[0] == actions.shape[0] == data.shape[0]
    transition_matrix = transition(labels) 
    transition_tensor = action_conditioned_transition(labels, actions)
    action_midpoints = compute_action_midpoints(data)
    feature_tensor = build_state_action_features(centroids, action_midpoints)
    feature_scale = compute_feature_scale(feature_tensor)
    normalized_tensor = normalize_feature_tensor(feature_tensor, feature_scale)
    feature_matrix = sample_feature_trajectory(labels, actions, normalized_tensor)
    expert_expectations = compute_expert_feature_expectations(labels, actions, normalized_tensor)
    print("Expert feature expectations:", expert_expectations)


"""Backbone-agnostic candidate selection for Selected Diffusion Noise (SDN).

All functions take action chunks sampled from different initial noises as tensors of
shape [N, T, D] (N candidates, T timesteps, D action dims) and return candidate
indices, so they can be shared by every VLA backbone.

Stage 1 (grounding filter): rank positive candidates G (sampled from the original
observation) by a k-NN density-ratio score against the negative set B (sampled from
the object-masked observation) and keep the top-M.
Stage 2 (kinematic refinement): among the top-M, pick the smoothest chunk. Two smoothness
scores are available: 'acceleration' (second-order difference + energy, used for pi0) and
'jerk' (JerkRMS, third-order difference).
"""

import torch


def knn_grounding_scores(actions, neg_actions, knn_k, eps=1e-8):
    """k-NN contrastive grounding score (higher = more grounded).

    For every positive candidate A_i, R_G / R_B is the sum of squared L2 distances to its
    k nearest neighbours in G \\ {A_i} / B. The score is log R_B - log R_G, i.e. candidates
    that lie in dense regions of G and sparse regions of B are preferred.

    Args:
        actions: [N, T, D] positive candidates G.
        neg_actions: [C, T, D] negative candidates B.
        knn_k: number of nearest neighbours, must satisfy knn_k < N and knn_k <= C.
        eps: numerical stabiliser inside the log.

    Returns:
        scores: [N]
    """
    num_pos, num_neg = actions.shape[0], neg_actions.shape[0]
    if not (0 < knn_k < num_pos and knn_k <= num_neg):
        raise ValueError(f"knn_k={knn_k} must be in [1, {min(num_pos - 1, num_neg)}] "
                         f"for {num_pos} positive and {num_neg} negative candidates")

    pos = actions.reshape(num_pos, -1).float()
    neg = neg_actions.reshape(num_neg, -1).float()

    # distance to the negative set
    dist_pos_neg = torch.cdist(pos, neg, p=2) ** 2
    r_neg = torch.topk(dist_pos_neg, k=knn_k, largest=False, dim=1).values.sum(dim=1)

    # distance to the positive set, excluding the candidate itself
    dist_pos_pos = torch.cdist(pos, pos, p=2) ** 2
    dist_pos_pos.fill_diagonal_(float("inf"))
    r_pos = torch.topk(dist_pos_pos, k=knn_k, largest=False, dim=1).values.sum(dim=1)

    return torch.log(r_neg + eps) - torch.log(r_pos + eps)


def smoothness_scores(actions, lambda_energy=0.05, ignore_gripper=True, eps=1e-6):
    """Kinematic smoothness score of each candidate (lower = smoother).

    Actions are normalised per dimension by their std over all candidates and timesteps.
    The score is the mean squared second-order finite difference plus
    `lambda_energy` times the mean squared magnitude, which prevents trivially
    selecting near-zero (idle) chunks.

    Args:
        actions: [N, T, D] candidates, T >= 3.
        lambda_energy: weight of the energy term.
        ignore_gripper: drop the last action dimension (binary gripper command).
        eps: numerical stabiliser for the std normalisation.

    Returns:
        scores: [N]
    """
    if actions.ndim != 3 or actions.shape[1] < 3:
        raise ValueError(f"expected [N, T>=3, D] actions, got {tuple(actions.shape)}")

    act = actions[..., :-1] if ignore_gripper else actions
    act = act / (act.std(dim=(0, 1), keepdim=True) + eps)

    acc = act[:, 2:] - 2 * act[:, 1:-1] + act[:, :-2]
    smoothness = (acc ** 2).sum(dim=-1).mean(dim=1)
    energy = (act ** 2).sum(dim=-1).mean(dim=1)
    return smoothness + lambda_energy * energy


def jerk_scores(actions, ignore_gripper=True, eps=1e-8, std_eps=None, delta=False):
    """JerkRMS of each candidate (lower = smoother), Eq. 9 of the paper.

    Actions are normalised per dimension by their std over all candidates and timesteps.
    The score is the RMS over time of the L2 norm of the third-order finite difference of the
    trajectory: of the actions themselves if they are absolute (positions), or the second-order
    difference if they are per-step deltas (`delta=True`), since then the trajectory is their running sum.

    Args:
        actions: [N, T, D] candidates, T >= 4.
        ignore_gripper: drop the last action dimension (binary gripper command).
        eps: numerical stabiliser for the square root, and for the std normalisation unless
            `std_eps` is given.
        std_eps: numerical stabiliser for the std normalisation (default: eps).
        delta: the actions are per-step deltas instead of absolute.

    Returns:
        scores: [N]
    """
    min_len = 3 if delta else 4
    if actions.ndim != 3 or actions.shape[1] < min_len:
        raise ValueError(f"expected [N, T>={min_len}, D] actions, got {tuple(actions.shape)}")

    std_eps = eps if std_eps is None else std_eps
    act = actions / (actions.std(dim=(0, 1), keepdim=True) + std_eps)
    if ignore_gripper:
        act = act[..., :-1]

    if delta:
        jerk = act[:, 2:] - 2 * act[:, 1:-1] + act[:, :-2]
    else:
        jerk = act[:, 3:] - 3 * act[:, 2:-1] + 3 * act[:, 1:-2] - act[:, :-3]
    return torch.sqrt((jerk ** 2).sum(dim=-1).mean(dim=1) + eps)


SMOOTHNESS_SCORES = {
    'acceleration': smoothness_scores,
    'jerk': jerk_scores,
}


def select_by_grounding(actions, neg_actions, knn_k, eps=1e-8):
    """Stage 1 only: index of the most grounded candidate."""
    return int(torch.argmax(knn_grounding_scores(actions, neg_actions, knn_k, eps)))


def select_by_smoothness(actions, smoothness='acceleration', **smoothness_kwargs):
    """Stage 2 only: index of the smoothest candidate.

    `smoothness` picks the score from SMOOTHNESS_SCORES; `smoothness_kwargs` are forwarded to it.
    """
    if smoothness not in SMOOTHNESS_SCORES:
        raise ValueError(f"smoothness must be one of {list(SMOOTHNESS_SCORES)}, got {smoothness}")
    return int(torch.argmin(SMOOTHNESS_SCORES[smoothness](actions, **smoothness_kwargs)))


def sdn_select(actions, neg_actions, knn_k, top_m, grounding_horizon=None, smoothness='acceleration',
               smoothness_horizon=None, **smoothness_kwargs):
    """Full SDN: grounding filter to the top-M candidates, then the smoothest of those.

    Args:
        actions: [N, T, D] positive candidates. T may be longer than the executed horizon
            so that stage 2 can see delayed oscillations.
        neg_actions: [C, T, D] negative candidates.
        knn_k: k of the k-NN grounding score.
        top_m: number of candidates kept by stage 1.
        grounding_horizon: if set, stage 1 compares only the first `grounding_horizon`
            steps (the part that will actually be executed).
        smoothness: 'acceleration' (`smoothness_scores`) or 'jerk' (`jerk_scores`).
        smoothness_horizon: if set, stage 2 scores only the first `smoothness_horizon` steps;
            by default it uses all T.
        **smoothness_kwargs: forwarded to the smoothness score.

    Returns:
        index into `actions` of the selected candidate.
    """
    if not 0 < top_m <= actions.shape[0]:
        raise ValueError(f"top_m={top_m} must be in [1, {actions.shape[0]}]")

    horizon = grounding_horizon or actions.shape[1]
    pos, neg = actions[:, :horizon], neg_actions[:, :horizon]
    scores = knn_grounding_scores(pos, neg, knn_k)
    top_indices = torch.topk(scores, k=top_m, largest=True).indices

    best_in_top = select_by_smoothness(actions[top_indices, :smoothness_horizon], smoothness, **smoothness_kwargs)
    return int(top_indices[best_in_top])

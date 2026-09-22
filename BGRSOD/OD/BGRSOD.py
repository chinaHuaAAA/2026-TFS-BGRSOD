import numpy as np
from scipy.stats import gaussian_kde

from GB import getGranularBall
from GBFS import get_feature_ranking


def assign_point_densities(X, balls, radii_mean, eps=1e-8):
    """Assign each sample the density of its original granular ball."""
    n_samples = len(X)
    point_density = np.full(n_samples, eps, dtype=float)
    assignment_count = np.zeros(n_samples, dtype=np.int32)

    if len(balls) != len(radii_mean):
        raise ValueError("balls and radii_mean must have the same length")

    for ball, radius in zip(balls, radii_mean):
        ball = np.asarray(ball)
        if ball.ndim != 2 or ball.shape[1] < 2:
            raise ValueError("each granular ball must be a non-empty 2-D array")

        # A row is [feature_1, ..., feature_d, original_sample_index].
        sample_indices = ball[:, -1].astype(np.intp)
        if np.any(sample_indices < 0) or np.any(sample_indices >= n_samples):
            raise ValueError("granular ball contains an invalid sample index")

        assignment_count[sample_indices] += 1
        if radius > 0:
            point_density[sample_indices] = len(ball) / radius

    if np.any(assignment_count != 1):
        missing = int(np.sum(assignment_count == 0))
        duplicated = int(np.sum(assignment_count > 1))
        raise ValueError(
            "granular balls must partition the samples exactly once; "
            f"missing={missing}, duplicated={duplicated}"
        )

    return point_density


def density_diff_pairing(subset_densities, n_pairs=None):
    """Greedily pair current minimum and maximum positive densities.

    For positive a <= b, (b-a)/(a+b) increases when a decreases or b
    increases.  Therefore sorting once gives the same greedy maximum-difference
    pairing as repeatedly searching all remaining pairs, except for irrelevant
    tie ordering.
    """
    subset_densities = np.asarray(subset_densities, dtype=float)
    omega = len(subset_densities)
    if n_pairs is None:
        n_pairs = omega // 2
    n_pairs = min(int(n_pairs), omega // 2)

    order = np.argsort(subset_densities, kind="stable")
    return [
        (int(order[i]), int(order[-1 - i]))
        for i in range(n_pairs)
    ]


def make_feature_sampling_weights(n_features):
    """Build rank-based feature probabilities once for all subspaces."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    ranks = np.arange(1, n_features + 1)
    weights = np.log(ranks[::-1] + 1.0)
    return weights / weights.sum()


def select_features_with_gentle_weights(
    sorted_indices,
    n_features_to_select,
    rng,
    feature_weights,
):
    """Sample feature indices without rebuilding probabilities or reseeding."""
    selected_positions = rng.choice(
        len(sorted_indices),
        size=n_features_to_select,
        replace=False,
        p=feature_weights,
    )
    return sorted_indices[selected_positions]


def compute_three_way_thresholds(scores, grid_size=512):
    """Estimate lower/upper three-way thresholds from KDE valleys."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("scores must contain at least one finite value")

    median = float(np.median(scores))
    if scores.size < 2 or np.ptp(scores) <= np.finfo(float).eps:
        return median, median

    grid_size = max(32, int(grid_size))
    u = np.linspace(scores.min(), scores.max(), grid_size)

    try:
        f = gaussian_kde(scores).evaluate(u)
    except (ValueError, np.linalg.LinAlgError):
        return median, median

    valley_mask = (f[:-2] > f[1:-1]) & (f[1:-1] < f[2:])
    valleys = u[1:-1][valley_mask]
    if valleys.size == 0:
        return median, median
    if valleys.size == 1:
        threshold = float(valleys[0])
        return threshold, threshold

    return float(valleys[0]), float(valleys[-1])


def _euclidean_distances(X, X_squared_norms, centers):
    """Compute distances to centers while reusing squared sample norms."""
    center_squared_norms = np.einsum("ij,ij->i", centers, centers)
    dist_sq = (
        X_squared_norms[:, None]
        + center_squared_norms[None, :]
        - 2.0 * np.dot(X, centers.T)
    )
    np.maximum(dist_sq, 0.0, out=dist_sq)
    np.sqrt(dist_sq, out=dist_sq)
    return dist_sq


def _apply_permeation_in_place(distances, ball_radii, alpha):
    """Convert distances to permeation deviations without 0*inf NaNs."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(distances, ball_radii[None, :], out=distances)
    outside = distances > 1.0

    # With alpha=0, the theoretical outside-ball value is exactly one.
    # Assign it directly because evaluating 0 * inf would produce NaN.
    if alpha == 0:
        distances[outside] = 1.0
    else:
        distances[outside] = 1.0 + alpha * (distances[outside] - 1.0)

    np.clip(distances, 0.0, 2.0, out=distances)
    if not np.isfinite(distances).all():
        raise FloatingPointError(
            "permeation deviations contain NaN or Inf; check regenerated-ball radii"
        )
    return distances


def _scores_from_regenerated_balls(
    X_sub,
    X_squared_norms,
    ball_centers,
    ball_radii,
    alpha,
    batch_size=None,
):
    """Score all samples, optionally using bounded-memory batches."""
    n_samples = len(X_sub)
    n_balls = len(ball_centers)

    if batch_size is None or batch_size >= n_samples:
        distances = _euclidean_distances(
            X_sub, X_squared_norms, ball_centers
        )
        ball_counts = np.count_nonzero(
            distances <= ball_radii[None, :], axis=0
        )
        densities = ball_counts / np.maximum(ball_radii, 1e-8)
        max_density = float(densities.max()) if n_balls else 0.0
        normalized_density = (
            densities / max_density if max_density > 0 else np.zeros(n_balls)
        )

        # Reuse the distance matrix as ratio, pitch, and weighted pitch.
        _apply_permeation_in_place(distances, ball_radii, alpha)
        distances *= (1.0 - normalized_density)[None, :]
        scores = distances.mean(axis=1)
        if not np.isfinite(scores).all():
            raise FloatingPointError("subspace scores contain NaN or Inf")
        return scores

    batch_size = max(1, int(batch_size))
    ball_counts = np.zeros(n_balls, dtype=np.int64)

    # First pass obtains global regenerated-ball densities.
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        distances = _euclidean_distances(
            X_sub[start:end],
            X_squared_norms[start:end],
            ball_centers,
        )
        ball_counts += np.count_nonzero(
            distances <= ball_radii[None, :], axis=0
        )

    densities = ball_counts / np.maximum(ball_radii, 1e-8)
    max_density = float(densities.max()) if n_balls else 0.0
    normalized_density = (
        densities / max_density if max_density > 0 else np.zeros(n_balls)
    )

    # Second pass computes scores with the now-known global densities.
    scores = np.empty(n_samples, dtype=float)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        distances = _euclidean_distances(
            X_sub[start:end],
            X_squared_norms[start:end],
            ball_centers,
        )
        _apply_permeation_in_place(distances, ball_radii, alpha)
        distances *= (1.0 - normalized_density)[None, :]
        scores[start:end] = distances.mean(axis=1)

    if not np.isfinite(scores).all():
        raise FloatingPointError("subspace scores contain NaN or Inf")
    return scores


def run_one_round(
    X,
    n_subsets,
    omega,
    alpha,
    rng,
    point_density,
    sorted_indices,
    n_sub_features,
    feature_weights=None,
    sample_pool=None,
    score_batch_size=None,
    radius_eps=1e-12,
):
    """Run subspaces and return an online score sum and valid count."""
    n_samples, _ = X.shape
    full_squared_norms = None
    if sorted_indices is None:
        full_squared_norms = np.einsum("ij,ij->i", X, X)

    score_sum = np.zeros(n_samples, dtype=float)
    valid_subsets = 0

    for _ in range(n_subsets):
        if sorted_indices is not None:
            feature_indices = select_features_with_gentle_weights(
                sorted_indices,
                n_sub_features,
                rng,
                feature_weights,
            )
            X_sub = X[:, feature_indices]
            X_squared_norms = np.einsum("ij,ij->i", X_sub, X_sub)
        else:
            X_sub = X
            X_squared_norms = full_squared_norms

        if sample_pool is None:
            indices = rng.choice(n_samples, omega, replace=False)
        else:
            indices = rng.choice(sample_pool, omega, replace=False)

        subset = X_sub[indices]
        subset_densities = point_density[indices]
        pairs = density_diff_pairing(subset_densities, omega // 2)

        ball_centers = []
        ball_radii = []
        for i_idx, j_idx in pairs:
            c1, c2 = subset[i_idx], subset[j_idx]
            distance = np.linalg.norm(c1 - c2)
            if distance <= 0:
                continue

            rho1 = subset_densities[i_idx]
            rho2 = subset_densities[j_idx]
            density_sum = rho1 + rho2
            ratio = rho2 / density_sum if density_sum > 0 else 0.5

            ball_centers.extend((c1, c2))
            ball_radii.extend((ratio * distance, (1.0 - ratio) * distance))

        if not ball_centers:
            continue

        ball_centers = np.asarray(ball_centers, dtype=float)
        ball_radii = np.asarray(ball_radii, dtype=float)
        valid_radius = np.isfinite(ball_radii) & (ball_radii > radius_eps)
        if not np.all(valid_radius):
            ball_centers = ball_centers[valid_radius]
            ball_radii = ball_radii[valid_radius]
        if ball_radii.size == 0:
            continue

        scores = _scores_from_regenerated_balls(
            X_sub,
            X_squared_norms,
            ball_centers,
            ball_radii,
            alpha,
            batch_size=score_batch_size,
        )
        score_sum += scores
        valid_subsets += 1

    return score_sum, valid_subsets


def BGRSOD(
    X,
    n_subsets=512,
    omega=18,
    alpha=0.1,
    random_state=None,
    subsample_features=True,
    kde_grid_size=512,
    score_batch_size=None,
    radius_eps=1e-12,
):
    """Optimized two-round BGRSOD anomaly scoring.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_subsets : int
        Total target number of subspaces across the two rounds.
    omega : int
        Even sample-subspace size.
    alpha : float
        Out-of-ball permeation attenuation in [0, 1].
    random_state : int or None
        Seed for a local NumPy random generator.
    subsample_features : bool
        Whether to use GBFS ranking and weighted feature subsampling.
    kde_grid_size : int
        Fixed one-dimensional grid size used to find KDE valleys.
    score_batch_size : int or None
        None uses the fastest full vectorized scoring.  Set, for example, to
        8192 on large datasets to bound temporary n-by-omega memory.
    radius_eps : float
        Regenerated radii at or below this value are treated as degenerate and
        removed before scoring.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")

    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 1:
        raise ValueError("X must contain at least two samples and one feature")
    if not isinstance(n_subsets, (int, np.integer)) or n_subsets < 2:
        raise ValueError("n_subsets must be an integer of at least 2")
    if not isinstance(omega, (int, np.integer)) or omega < 2 or omega % 2:
        raise ValueError("omega must be a positive even integer")
    if omega > n_samples:
        raise ValueError("omega cannot exceed the number of samples")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if score_batch_size is not None and score_batch_size < 1:
        raise ValueError("score_batch_size must be positive or None")
    if not np.isfinite(radius_eps) or radius_eps <= 0:
        raise ValueError("radius_eps must be a positive finite number")

    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(random_state)

    _, balls, _, radii_mean, _ = getGranularBall(X)
    point_density = assign_point_densities(X, balls, radii_mean)

    sorted_indices = None
    feature_weights = None
    if subsample_features:
        sorted_indices, _ = get_feature_ranking(X)
        n_sub_features = min(n_features, max(1, int(np.sqrt(n_features))))
        feature_weights = make_feature_sampling_weights(n_features)
    else:
        n_sub_features = n_features

    n_round1 = n_subsets // 2
    n_round2 = n_subsets - n_round1

    round1_sum, count1 = run_one_round(
        X,
        n_round1,
        omega,
        alpha,
        rng,
        point_density,
        sorted_indices,
        n_sub_features,
        feature_weights=feature_weights,
        sample_pool=None,
        score_batch_size=score_batch_size,
        radius_eps=radius_eps,
    )
    if count1 == 0:
        return np.zeros(n_samples, dtype=float)

    preliminary_scores = round1_sum / count1
    beta, alpha_threshold = compute_three_way_thresholds(
        preliminary_scores,
        grid_size=kde_grid_size,
    )
    boundary_indices = np.flatnonzero(
        (preliminary_scores >= beta)
        & (preliminary_scores <= alpha_threshold)
    )

    total_sum = round1_sum
    total_count = count1
    if boundary_indices.size >= omega:
        round2_sum, count2 = run_one_round(
            X,
            n_round2,
            omega,
            alpha,
            rng,
            point_density,
            sorted_indices,
            n_sub_features,
            feature_weights=feature_weights,
            sample_pool=boundary_indices,
            score_batch_size=score_batch_size,
            radius_eps=radius_eps,
        )
        total_sum += round2_sum
        total_count += count2

    return total_sum / total_count if total_count else np.zeros(n_samples)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler

    X_normal, _ = make_blobs(
        n_samples=500,
        centers=3,
        n_features=20,
        cluster_std=0.5,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    X_outlier = rng.uniform(low=-5, high=5, size=(50, 20))
    X = np.vstack((X_normal, X_outlier))
    y = np.r_[np.zeros(len(X_normal)), np.ones(len(X_outlier))]
    X = MinMaxScaler().fit_transform(X)

    scores = BGRSOD(
        X,
        n_subsets=64,
        omega=18,
        alpha=0.1,
        random_state=42,
    )
    print(f"AUC: {roc_auc_score(y, scores):.6f}")

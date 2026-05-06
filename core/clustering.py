"""
FCM++ Clustering with Dual-Threshold Filtering.
Implements entropy-driven pseudo-label generation with cross-modal consensus.
"""
import collections
import random
import numpy as np
import faiss


def _init_kmeans_centers(X, ncentroids, niter=50):
    d = X.shape[1]
    km = faiss.Kmeans(
        d, ncentroids, niter=niter, verbose=True,
        spherical=True, min_points_per_centroid=1,
        max_points_per_centroid=X.shape[0], gpu=False, nredo=10
    )
    km.train(X.astype(np.float32))
    return km.centroids.astype(np.float32)


def do_cluster_v2_fcm_plus(
    all_keys, all_emb, all_emb_v, all_emb_f,
    ncentroids, input_emb_type="all",
    m_start=2.0, m_end=1.25,
    gamma_start=1.0, gamma_end=2.0,
    max_iter=30, eps=1e-4,
    top_p=0.6, min_keep_ratio=0.45,
    consensus_alpha=0.25, use_consensus=True,
    prev_centers=None, ema_momentum=0.7,
    epoch=0, epoch_max=10,
    verbose=True
):
    """
    Fuzzy C-Means clustering with:
      - Temperature annealing (m, gamma)
      - EMA center update
      - Cross-modal consensus regularization
      - Dual-threshold sample selection (safe set + hard mining)
    Returns:
        movie2label: dict, pseudo-label mapping
        C: cluster centers
        kept_ratio: float
        movie2weight: dict, adaptive confidence weights
    """
    # 1) Select clustering space
    if input_emb_type == "v":
        X = np.array(all_emb_v, dtype=np.float32)
    elif input_emb_type == "f":
        X = np.array(all_emb_f, dtype=np.float32)
    elif input_emb_type == "all":
        X = np.array(all_emb, dtype=np.float32)
    else:
        raise ValueError("input_emb_type must be 'v', 'f', or 'all'")

    # 2) Normalize
    faiss.normalize_L2(X)
    N, D = X.shape

    # 3) Annealing coefficients
    t = min(1.0, float(epoch) / max(1, int(epoch_max)))
    m_eff = m_start + (m_end - m_start) * t
    gamma_eff = gamma_start + (gamma_end - gamma_start) * t

    # 4) Initialize centers
    C = _init_kmeans_centers(X, ncentroids)
    faiss.normalize_L2(C)
    if prev_centers is not None and prev_centers.shape == C.shape:
        C = (1.0 - ema_momentum) * C + ema_momentum * prev_centers
        faiss.normalize_L2(C)

    # 5) FCM iteration
    U = None
    for _ in range(max_iter):
        # (a) Distance: 1 - cosine similarity
        sims = X @ C.T
        dist = 1.0 - np.clip(sims, -1.0, 1.0)
        dist = np.maximum(dist, 1e-7)

        # (b) Update membership U
        power = 2.0 / max(1e-6, (m_eff - 1.0))
        inv = dist ** (-power)
        inv = np.clip(inv, 0.0, 1e30)
        denom = np.sum(inv, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        U = inv / denom

        # NaN repair
        if np.isnan(U).any():
            nan_rows = np.isnan(U).any(axis=1)
            U[nan_rows] = 1.0 / ncentroids
            if verbose:
                print(f"  [Warning] Fixed {np.sum(nan_rows)} NaN rows.")

        # (c) Sharpen membership
        if gamma_eff != 1.0:
            U = U ** gamma_eff
            U = np.clip(U, 0.0, 1e30)
            U = U / (U.sum(axis=1, keepdims=True) + 1e-12)

        # (d) Cross-modal consensus
        if use_consensus and input_emb_type == "all":
            if (all_emb_v is not None) and (all_emb_f is not None):
                V = np.array(all_emb_v, dtype=np.float32)
                faiss.normalize_L2(V)
                F = np.array(all_emb_f, dtype=np.float32)
                faiss.normalize_L2(F)

                denom_c = np.sum(U, axis=0, keepdims=True).T
                denom_c = np.maximum(denom_c, 1e-12)

                C_v = (U.T @ V) / denom_c
                C_f = (U.T @ F) / denom_c
                faiss.normalize_L2(C_v)
                faiss.normalize_L2(C_f)

                sims_v = V @ C_v.T
                sims_f = F @ C_f.T

                Uv = np.exp(sims_v)
                Uv /= (Uv.sum(axis=1, keepdims=True) + 1e-12)
                Uf = np.exp(sims_f)
                Uf /= (Uf.sum(axis=1, keepdims=True) + 1e-12)

                U = U * (np.power(Uv, consensus_alpha)) * (np.power(Uf, consensus_alpha))
                U = U / (U.sum(axis=1, keepdims=True) + 1e-12)

        # (e) Update centers
        Um = U ** m_eff
        denom_new = np.sum(Um, axis=0, keepdims=True).T + 1e-12
        C_new = (Um.T @ X) / denom_new
        faiss.normalize_L2(C_new)

        # (f) Convergence check
        delta = np.linalg.norm(C_new - C) / np.sqrt(np.prod(C.shape))
        C = C_new
        if delta < eps:
            break

    # 6) Hard labeling + dual-threshold filtering
    hard = np.argmax(U, axis=1)
    idxs = np.arange(N)

    # (1) Safe set: top-p confidence per cluster
    keep = np.zeros(N, dtype=bool)
    for k in range(ncentroids):
        members = idxs[hard == k]
        if len(members) == 0:
            continue
        scores = U[members, k]
        K_num = max(1, int(np.ceil(len(members) * top_p)))
        top_idx = members[np.argpartition(-scores, K_num - 1)[:K_num]]
        keep[top_idx] = True

    # (2) Hard-sample recovery band
    hard_band_ql, hard_band_qh = 0.20, 0.40
    hard_ratio_pc, hard_cap_glb = 0.10, 0.15

    s_max = U.max(axis=1)
    thr_l = np.quantile(s_max, hard_band_ql)
    thr_h = np.quantile(s_max, hard_band_qh)
    entropy = -np.sum(U * np.log(U + 1e-12), axis=1)

    band_mask = (~keep) & (s_max >= thr_l) & (s_max <= thr_h)
    num_cap_glb = int(np.floor(N * hard_cap_glb))
    picked_glb = 0

    for k in range(ncentroids):
        members = idxs[(hard == k) & band_mask]
        if len(members) == 0:
            continue
        K_hard = max(0, int(np.floor(len(idxs[hard == k]) * hard_ratio_pc)))
        if K_hard == 0:
            continue

        scores = entropy[members]
        if K_hard < len(members):
            pick = members[np.argpartition(-scores, K_hard - 1)[:K_hard]]
        else:
            pick = members

        room = num_cap_glb - picked_glb
        if room <= 0:
            break
        if len(pick) > room:
            pick = pick[:room]
        keep[pick] = True
        picked_glb += len(pick)

    # Fallback if kept ratio too low
    kept_ratio = keep.mean()
    if kept_ratio < float(min_keep_ratio):
        q = 1.0 - float(min_keep_ratio)
        thr = np.quantile(U.max(axis=1), q)
        keep = (U.max(axis=1) >= thr)
        kept_ratio = keep.mean()

    # Build outputs
    s_max = U.max(axis=1)
    s_max = np.nan_to_num(s_max, nan=0.0)

    movie2label = {}
    movie2weight = {}
    for key, label, weight, kk in zip(all_keys, hard, s_max, keep):
        if kk:
            movie2label[key] = int(label)
            movie2weight[key] = float(weight)

    if verbose:
        print(f"[FCM+] kept={keep.sum()}/{N} ({kept_ratio:.1%}), "
              f"safe_p={top_p}, band=[{hard_band_ql:.2f},{hard_band_qh:.2f}], "
              f"hard_pc={hard_ratio_pc}, hard_cap={hard_cap_glb}, C={C.shape}")

    if not isinstance(C, np.ndarray):
        C = np.zeros((ncentroids, X.shape[1]), dtype=np.float32)
    C = C.astype(np.float32)

    return movie2label, C, float(kept_ratio), movie2weight

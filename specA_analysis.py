"""
specA_analysis.py

Geometry analysis module: Parts A–H of the PDSF experiment.

Part of: PDSF — Prediction-Anchored Decomposition into Functional Subspaces

Paper:   "Geometric and Behavioral Stratification in Transformer Residual Streams"
         Nelson Guda, 2026
Repo:    https://github.com/nelsonguda/pdsf-residual-geometry

License: MIT (code), CC-BY 4.0 (data)


Purpose:
    Implements the geometry analysis pipeline that tests how surface-form
    variations in prompts affect residual stream geometry. Given prompts that
    vary in surface form but share a semantic answer, decomposes each prompt's
    residual stream into P + D + S + F and analyzes the geometric properties
    of each subspace.

Analysis parts (corresponds to paper sections):
    Part A — Participation Ratio (PR):     §4.1, Table 1
    Part N — Scaling rank (6 estimators):  §4.1, Table 1, Figure 2
    Part B — Family Separation / angles:   §4.2
    Part C — Factor Effects:               §4.5, Appendix C.3
    Part D — PR(D) Trajectory:             §4.1
    Part E — μ Landscape:                  diagnostic (not reported)
    Part F — Rotation Analysis:            diagnostic here; developed in the companion paper
    Part G — Interventions:                §5.1–5.2 (implemented in pds_geometry.py)
    Part H — Injectivity:                  diagnostic (not reported)

Mathematical conventions:
    H ∈ ℝ^(n×d): Hidden states, n=samples, d=hidden_dim
    Bp ∈ ℝ^(d×r_p): P basis (orthonormal columns), usually r_p=1
    Bd ∈ ℝ^(d×r_d): D basis (top-k PCA of H⊥P), r_d = round(PR(H⊥P))
    Bs ∈ ℝ^(d×r_s): S basis (top-k PCA after projecting out P and D)
    project_onto_basis(X, B) = X @ B @ B^T (orthogonal projection)
    participation_ratio(X) = (Σλ_i)² / Σλ_i² (effective dimensionality)
    principal_angles(U, V) = arccos(SVD(U^T @ V)) (subspace alignment, degrees)

NOTE on pca_basis():
    This module's pca_basis() returns np.ndarray only.
    pds_continuation.py's version returns Tuple[np.ndarray, np.ndarray].
    They are NOT interchangeable.

Inputs:
    ModelExtraction objects (hidden states per layer), P bases, unembedding matrix.

Outputs:
    Per-part JSON result files (one per model per analysis part).

Dependencies:
    numpy, torch (for get_unembed_weight only), pds_geometry (MODEL_REGISTRY)
"""

from __future__ import annotations


import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import warnings
import numpy as np
import torch


def get_unembed_weight(model):
    """Return unembedding/lm_head weight as torch Tensor (vocab, d).
    
    The unembedding matrix maps residual stream vectors to vocabulary logits.
    Its rows are the "token directions" used to construct the P subspace.
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    raise AttributeError("Could not locate unembedding weight.")


# ----------------------------
# Basic linear algebra helpers
# ----------------------------

def _to_np(x) -> np.ndarray:
    """Convert tensor to numpy array with float32 precision."""
    if isinstance(x, np.ndarray):
        return x
    # torch tensor - ensure float32 for consistent computation
    return x.detach().cpu().float().numpy()


def orthonormal_basis_from_vectors(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Construct an orthonormal basis for span(V) via SVD.
    
    Args:
        V: Input vectors, shape (d, m) or (m, d) where d is dimension
        eps: Threshold for numerical rank determination
        
    Returns:
        Orthonormal basis of shape (d, r) where r is numerical rank
    """
    if V.ndim != 2:
        raise ValueError("V must be 2D")
    # Ensure shape (d, m)
    if V.shape[0] < V.shape[1]:
        # likely (m, d) where m < d; transpose to (d, m)
        V = V.T
    # QR with column pivoting is ideal; use SVD for robustness
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    r = int((S > eps * S.max()).sum()) if S.size else 0
    if r == 0:
        return np.zeros((V.shape[0], 0), dtype=np.float32)
    return U[:, :r].astype(np.float32)


def project_onto_basis(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal projection of rows of X onto span(B).
    
    Args:
        X: Data matrix, shape (n, d)
        B: Orthonormal basis, shape (d, r)
        
    Returns:
        Projection of X onto span(B), shape (n, d)
    """
    if B.size == 0:
        return np.zeros_like(X)
    return (X @ B) @ B.T


def participation_ratio_from_cov_eigs(eigs: np.ndarray, eps: float = 1e-12) -> float:
    """Compute PR from covariance eigenvalues.
    
    PR = (Σλ)² / Σλ²
    """
    eigs = np.clip(eigs, 0, None)
    s1 = float(eigs.sum())
    s2 = float((eigs ** 2).sum())
    if s2 < eps:
        return 0.0
    return (s1 ** 2) / s2


def participation_ratio(X: np.ndarray, center: bool = True) -> float:
    """Compute participation ratio (effective dimensionality) of point cloud X.
    
    Uses covariance eigenvalues via SVD for numerical stability.
    
    Args:
        X: Data matrix, shape (n, d)
        center: Whether to center the data first
        
    Returns:
        Participation ratio (1 to min(n-1, d))
    """
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # Cov eigenvalues proportional to singular values squared
    try:
        s = np.linalg.svd(X, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError:
        return float('nan')
    eigs = (s ** 2) / max(1, (X.shape[0] - 1))
    return float(participation_ratio_from_cov_eigs(eigs))


def pca_basis(X: np.ndarray, k: int, center: bool = True) -> np.ndarray:
    """Compute top-k PCA directions as orthonormal basis.
    
    Args:
        X: Data matrix, shape (n, d)
        k: Number of components
        center: Whether to center data
        
    Returns:
        Basis vectors, shape (d, k)
    """
    if center:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X
    # SVD of (n,d): right singular vectors are principal axes
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    B = Vt[:k].T
    return B.astype(np.float32)


def principal_angles(U: np.ndarray, V: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Compute principal angles between subspaces spanned by columns of U and V.
    
    Principal angles are the canonical angles measuring alignment between
    two subspaces. Small angles indicate aligned subspaces.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    Given two subspaces U and V of dimension r1 and r2:
    - The first principal angle θ₁ is the smallest angle between any vector
      in U and any vector in V
    - θ₂ is the smallest angle in the orthogonal complement of the vectors
      achieving θ₁, and so on
    - If U and V share a k-dimensional intersection, the first k angles are 0°
    
    The angles are computed via SVD: if U and V are orthonormal bases, then
    the singular values of U.T @ V are cos(θᵢ).
    
    RELATIONSHIP TO SUBSPACE ALIGNMENT:
    ----------------------------------
    - All angles ≈ 0°: Subspaces are nearly identical
    - All angles ≈ 90°: Subspaces are nearly orthogonal
    - Mixed: Partial overlap
    
    For SpecA, we expect:
    - Within-family D subspaces: small angles (similar variation patterns)
    - Between-family D subspaces: large angles (different semantic content)
    
    Args:
        U: First basis, shape (d, r1) with orthonormal columns
        V: Second basis, shape (d, r2) with orthonormal columns
        k: Max number of angles to return (default: min(r1, r2))
        
    Returns:
        Array of angles in degrees, length = min(r1, r2, k)
        Angles are sorted from smallest to largest.
    """
    if U.size == 0 or V.size == 0:
        return np.array([], dtype=np.float32)
    r = min(U.shape[1], V.shape[1])
    if k is not None:
        r = min(r, k)
    # Compute singular values of U^T V (these are cosines of principal angles)
    M = U.T @ V
    try:
        s = np.linalg.svd(M, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((r,), np.nan, dtype=np.float32)
    # Clip to [-1, 1] to handle numerical error before arccos
    s = np.clip(s[:r], -1.0, 1.0)
    # Convert cosines to angles in degrees
    angles = np.degrees(np.arccos(s))
    return angles.astype(np.float32)


def subspace_overlap_trace(U: np.ndarray, V: np.ndarray) -> float:
    """Compute normalized trace overlap between orthonormal bases U and V.
    
    Returns tr(P_U @ P_V) / dim(U), measuring what fraction of U is captured by V.
    """
    if U.size == 0:
        return 0.0
    M = U.T @ V
    return float((M * M).sum() / U.shape[1])


def entropy_from_logits(logits: np.ndarray) -> float:
    """Compute entropy of softmax distribution from logits."""
    z = logits - logits.max()
    exp = np.exp(z)
    p = exp / exp.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def kl_divergence_from_logits(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Compute KL(p||q) from logits using softmax."""
    pz = p_logits - p_logits.max()
    qz = q_logits - q_logits.max()
    p = np.exp(pz)
    p = p / p.sum()
    q = np.exp(qz)
    q = q / q.sum()
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


# ----------------------------
# Spec A analysis data structures
# ----------------------------

@dataclass
class LayerTensors:
    """Container for extracted tensors at a single layer.
    
    Attributes:
        H: Hidden states at prompt boundary, shape (n_samples, hidden_dim)
        logits: Optional boundary logits, shape (n_samples, vocab_size)
    """
    H: np.ndarray            # (n, d) boundary hidden states
    logits: Optional[np.ndarray] = None  # (n, vocab) boundary logits (optional)


@dataclass
class ModelExtraction:
    """Complete extraction results for one model.
    
    Attributes:
        model_id: HuggingFace model identifier
        layers: Dict mapping layer index → LayerTensors
        group_ids: Group ID for each sample (length n)
        variant_ids: Unique ID for each sample (length n)
        meta_rows: Full metadata dict for each sample
    """
    model_id: str
    layers: Dict[int, LayerTensors]  # layer_idx -> tensors
    group_ids: List[str]             # length n; group for each row in H
    variant_ids: List[str]           # length n; unique id per sample
    meta_rows: List[Dict[str, Any]]  # metadata per sample (factors, expected token)


def build_P_basis_from_expected_tokens(
    token_ids: List[int],
    unembed_weight: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build orthonormal basis for P from unembedding rows corresponding to token_ids.
    
    The P (predictive) subspace is the span of unembedding vectors for expected tokens.
    This represents the directions that directly contribute to predicting the answer.
    
    RELATIONSHIP TO UNEMBEDDING MATRIX:
    ----------------------------------
    The unembedding matrix W (lm_head.weight) has shape (vocab_size, hidden_dim).
    Row W[i] is the "token direction" for vocabulary item i - projecting the residual
    stream onto this row gives the logit for token i.
    
    For a question with expected answer token t, the P subspace is span{W[t]}.
    If multiple answer tokens are acceptable (e.g., "Yes"/"yes"), P spans all of them.
    
    We orthonormalize to get a proper basis, which may have rank < len(token_ids)
    if some token directions are linearly dependent.
    
    Args:
        token_ids: List of token IDs for expected answers
        unembed_weight: Unembedding matrix, shape (vocab, d) or (d, vocab)
        eps: Threshold for numerical rank (singular values < eps*max are dropped)
        
    Returns:
        Orthonormal basis for P, shape (d, r) where r ≤ len(token_ids)
        
    Example:
        >>> # Get P basis for "Yes" and "yes" tokens
        >>> yes_ids = [tokenizer.encode("Yes")[0], tokenizer.encode("yes")[0]]
        >>> P_basis = build_P_basis_from_expected_tokens(yes_ids, unembed_weight)
        >>> P_basis.shape  # (hidden_dim, 1) or (hidden_dim, 2) depending on linear independence
    """
    W = unembed_weight
    if W.ndim != 2:
        raise ValueError("unembed_weight must be 2D")
    # Expect HF lm_head weight usually (vocab, d)
    if W.shape[0] < W.shape[1]:
        # could be (d, vocab) - transpose to standard form
        W = W.T
    # Extract rows for expected tokens: (m, d) where m = len(token_ids)
    vecs = W[token_ids, :]
    # Orthonormalize to get basis: returns (d, r) where r = numerical rank
    B = orthonormal_basis_from_vectors(vecs)
    return B


# ============================================================
# PART A: Participation Ratio Analysis
# ============================================================
#
# HYPOTHESIS TESTED:
# -----------------
# H1: Prediction uses a low-dimensional subspace P, while most variance
#     lies in the orthogonal complement P⊥.
#
# EXPECTED RESULTS:
# ----------------
# - PR(P) should be small (close to 1-3), indicating prediction is concentrated
# - PR(P⊥) should be much larger, showing rich information in complement
# - The ratio PR(P⊥)/PR(P) quantifies how much "extra" information exists
#
# INTERPRETATION GUIDE:
# --------------------
# - PR(P) ≈ 1-2: Model uses essentially one direction for prediction (good)
# - PR(P⊥) >> PR(P): Most geometric structure is orthogonal to prediction
# - If PR(P) is large, the P subspace definition may need revision
#
# Note: PR is computed on centered data by default, measuring variance
# spread rather than absolute position.

# §4.1, Table 1
def compute_part_A_PR(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    center: bool = True,
) -> Dict[str, Any]:
    """Part A: Compute PR(P) and PR(P⊥) across all layers.
    
    This tests the hypothesis that prediction uses a low-dimensional subspace,
    while most variance lies in the orthogonal complement.
    
    The P subspace is defined by unembedding vectors for expected tokens.
    P⊥ is everything else - this is where we expect semantic and divergent
    information to live.
    
    Args:
        extraction: Model extraction containing H matrices per layer
        P_bases: P_bases[layer][group_id] gives orthonormal basis for P
        center: Whether to center data before computing PR (recommended)
        
    Returns:
        Dict with structure:
        {
            "by_layer": {
                "<layer>": {
                    "PR_P": float,        # Effective dim of prediction subspace
                    "PR_P_perp": float,   # Effective dim of orthogonal complement
                    "n": int,             # Number of samples
                    "d": int              # Hidden dimension
                }, ...
            },
            "summary": { ... }  # Stats from last layer
        }
    """
    out = {"by_layer": {}, "summary": {}}
    for layer, lt in extraction.layers.items():
        H = lt.H
        # Build per-sample P projection using group-specific bases
        HP = np.zeros_like(H)
        for gi, g in enumerate(extraction.group_ids):
            Bp = P_bases[layer].get(gi, np.zeros((H.shape[1], 1), dtype=np.float32))
            HP[gi:gi+1, :] = project_onto_basis(H[gi:gi+1, :], Bp)
        Hperp = H - HP
        pr_p = participation_ratio(HP, center=center)
        pr_perp = participation_ratio(Hperp, center=center)
        out["by_layer"][str(layer)] = {
            "PR_P": pr_p,
            "PR_P_perp": pr_perp,
            # Add convenience aliases for backward compatibility
            "PR_P_mean": pr_p,
            "PR_Pperp_mean": pr_perp,
            "n": int(H.shape[0]),
            "d": int(H.shape[1]),
        }
    # Summary: last layer stats
    last = max(extraction.layers.keys())
    out["summary"] = out["by_layer"][str(last)]
    out["summary"]["layer"] = int(last)
    return out


# ============================================================
# PART B: Family Separation Analysis
# ============================================================

def compute_D_basis_per_group(
    Hperp_by_group: Dict[str, np.ndarray],
    k_policy: str = "round_pr",
    k_min: int = 1,
    k_max: int = 64,
) -> Dict[str, np.ndarray]:
    """Compute D basis for each group using top-k PCs of H_perp.
    
    The number of components k is set adaptively based on PR of the group's data.
    """
    bases = {}
    for g, X in Hperp_by_group.items():
        pr = participation_ratio(X, center=True)
        if k_policy == "round_pr":
            k = int(round(pr))
        elif k_policy == "ceil_pr":
            k = int(math.ceil(pr))
        else:
            raise ValueError("unknown k_policy")
        k = max(k_min, min(k, k_max, X.shape[1], X.shape[0]-1 if X.shape[0] > 1 else 1))
        bases[g] = pca_basis(X, k=k, center=True)
        bases[g + "__meta"] = {"k": k, "pr": pr}
    return bases


#   between_family_mean_angle_deg (should be large)
def compute_part_B_family_separation(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    layer: int,
    k_policy: str = "round_pr",
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part B: Compare within-family vs between-family D subspace angles.
    
    This tests whether prompt families (same question, different surface forms)
    have aligned D subspaces, while different families have orthogonal D subspaces.
    
    Args:
        extraction: Model extraction
        P_bases: P bases per layer and group
        layer: Which layer to analyze
        k_policy: How to set dimensionality ("round_pr" or "ceil_pr")
        k_max: Maximum D dimensionality
        
    Returns:
        Dict with:
        - within_family_mean_angle_deg: Mean angle within groups (should be small)
        - between_family_mean_angle_deg: Mean angle between groups (should be large)
        - per_group: Detailed stats per group
    """
    lt = extraction.layers[layer]
    H = lt.H
    # Per-sample P projection -> Hperp
    HP = np.zeros_like(H)
    for i, g in enumerate(extraction.group_ids):
        HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
    Hperp = H - HP

    # Build per-group matrices
    group_to_idx = {}
    for i, g in enumerate(extraction.group_ids):
        group_to_idx.setdefault(g, []).append(i)

    D_group = {}
    within_angles = []
    for g, idxs in group_to_idx.items():
        X = Hperp[idxs, :]
        pr = participation_ratio(X, center=True)
        k = max(1, min(int(round(pr)), k_max, X.shape[1], X.shape[0]-1 if X.shape[0] > 1 else 1))
        Dg = pca_basis(X, k=k, center=True)
        D_group[g] = {"basis": Dg, "k": k, "pr": pr, "n": len(idxs)}
        # Within-family: split into halves if possible
        if len(idxs) >= 4:
            mid = len(idxs) // 2
            X1 = Hperp[idxs[:mid], :]
            X2 = Hperp[idxs[mid:], :]
            k1 = max(1, min(int(round(participation_ratio(X1))), k_max, X1.shape[1], X1.shape[0]-1 if X1.shape[0] > 1 else 1))
            k2 = max(1, min(int(round(participation_ratio(X2))), k_max, X2.shape[1], X2.shape[0]-1 if X2.shape[0] > 1 else 1))
            U1 = pca_basis(X1, k=k1)
            U2 = pca_basis(X2, k=k2)
            ang = principal_angles(U1, U2)
            if ang.size:
                within_angles.append(float(np.mean(ang)))
    
    # Between-family angles
    groups = list(D_group.keys())
    between_angles = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            U = D_group[groups[i]]["basis"]
            V = D_group[groups[j]]["basis"]
            ang = principal_angles(U, V)
            if ang.size:
                between_angles.append(float(np.mean(ang)))
    
    within_mean = float(np.mean(within_angles)) if within_angles else float('nan')
    between_mean = float(np.mean(between_angles)) if between_angles else float('nan')
    
    return {
        "layer": int(layer),
        # Primary keys
        "within_family_mean_angle_deg": within_mean,
        "between_family_mean_angle_deg": between_mean,
        # Convenience aliases for backward compatibility
        "within_mean_deg": within_mean,
        "between_mean_deg": between_mean,
        # Detailed data
        "within_family_angles_deg": within_angles,
        "between_family_angles_deg": between_angles,
        "per_group": {g: {k: v for k, v in info.items() if k != "basis"} for g, info in D_group.items()},
    }


# ============================================================
# PART C: Factor Effects Analysis
# ============================================================

def compute_part_C_factor_effects(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    layer: int,
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part C: Measure how design factors (A,B,C,D) affect D-space energy.
    
    The experiment uses a 2^4 factorial design with factors:
    - A: Paraphrase (0=version1, 1=version2)
    - B: Constraint (0=none, 1=present)
    - C: Clutter (0=none, 1=present)
    - D: Format (0=imperative, 1=declarative)
    
    This analysis measures the geometric impact of each factor.
    
    Args:
        extraction: Model extraction
        P_bases: P bases per layer and group
        layer: Which layer to analyze
        k_max: Maximum D dimensionality
        
    Returns:
        Dict with Cohen's d and mean difference per factor
    """
    lt = extraction.layers[layer]
    H = lt.H
    # Per-sample H_perp
    HP = np.zeros_like(H)
    for i, g in enumerate(extraction.group_ids):
        HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
    Hperp = H - HP
    X = Hperp - Hperp.mean(axis=0, keepdims=True)
    pr = participation_ratio(X, center=False)
    k = max(1, min(int(round(pr)), k_max, X.shape[1], X.shape[0]-1 if X.shape[0] > 1 else 1))
    Bd = pca_basis(X, k=k, center=False)
    XD = project_onto_basis(X, Bd)
    scores = np.sum(XD * XD, axis=1)  # (n,) - D-space energy per sample

    def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Cohen's d effect size between two groups."""
        if a.size < 2 or b.size < 2:
            return float('nan')
        va = a.var(ddof=1)
        vb = b.var(ddof=1)
        sp = math.sqrt(((a.size-1)*va + (b.size-1)*vb) / max(1, (a.size+b.size-2)))
        if sp == 0:
            return float('nan')
        return float((a.mean() - b.mean()) / sp)

    results = {"layer": int(layer), "k_D": int(k), "PR_P_perp": float(pr), "factors": {}}
    for F in ["A", "B", "C", "D"]:
        vals = np.array([int(m.get(F, 0)) for m in extraction.meta_rows], dtype=int)
        s0 = scores[vals == 0]
        s1 = scores[vals == 1]
        results["factors"][F] = {
            "n0": int(s0.size),
            "n1": int(s1.size),
            "mean0": float(s0.mean()) if s0.size else float('nan'),
            "mean1": float(s1.mean()) if s1.size else float('nan'),
            "diff_1_minus_0": float((s1.mean() - s0.mean())) if (s0.size and s1.size) else float('nan'),
            "cohens_d": cohens_d(s1, s0),
        }
    return results


# ============================================================
# PART D: PR(D) Trajectory
# ============================================================

def compute_part_D_trajectory(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    k_policy: str = "round_pr",
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part D: PR(D) trajectory across layers.
    
    Alias for compute_part_D_trajectory_PRD for backward compatibility.
    """
    return compute_part_D_trajectory_PRD(extraction, P_bases, k_policy, k_max)


# §4.1 (depth trajectory)
def compute_part_D_trajectory_PRD(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    k_policy: str = "round_pr",
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part D: Track PR(D) across layers.
    
    Shows how the model allocates effective dimensions to the D subspace
    at different depths. Increasing PR(D) suggests the model is spreading
    divergent information across more dimensions.
    
    Args:
        extraction: Model extraction
        P_bases: P bases per layer and group
        k_policy: How to set k ("round_pr")
        k_max: Maximum D dimensionality
        
    Returns:
        Dict with PR_P_perp and PR_D per layer
    """
    out = {"by_layer": {}}
    for layer, lt in extraction.layers.items():
        H = lt.H
        HP = np.zeros_like(H)
        for i, g in enumerate(extraction.group_ids):
            HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
        Hperp = H - HP
        pr_perp = participation_ratio(Hperp, center=True)
        k = max(1, min(int(round(pr_perp)), k_max, Hperp.shape[1], Hperp.shape[0]-1 if Hperp.shape[0] > 1 else 1))
        D = pca_basis(Hperp, k=k, center=True)
        HD = project_onto_basis(Hperp, D)
        pr_d = participation_ratio(HD, center=True)
        out["by_layer"][str(layer)] = {"PR_P_perp": pr_perp, "k": k, "PR_D": pr_d, "n": int(H.shape[0])}
    return out


# ============================================================
# PART E: μ Landscape Analysis
# ============================================================


#   Also: group centroid pairwise distances in D vs S (D_to_S_ratio)
def compute_part_E_mu_landscape(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    unembed_weight: Optional[np.ndarray],
    s_small_r: int = 16,
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part E: Decompose mean vector μ into P, D, S energy components.
    
    Tests whether μ (representing shared semantic content) lies primarily
    in the S (semantic) subspace rather than D (divergent).
    
    Includes per-group μ analysis to compare centroids across
    answer families (Yes, No, True, A, etc.).
    
    S is the top-k PCA of H after projecting out P and D (sequential decomposition).
    
    Args:
        extraction: Model extraction
        P_bases: P bases per layer and group
        unembed_weight: Unembedding matrix for baseline logit computation
        s_small_r: Maximum rank for S basis
        k_max: Maximum D dimensionality
        
    Returns:
        Dict with:
            - by_layer: Global μ energy decomposition per layer
            - per_group: Per-group μ analysis
            - group_separation: Distance/angle between group centroids
            - drift: μ movement across layers
            - summary: Key metrics from last layer
    """
    layers_sorted = sorted(extraction.layers.keys())
    unique_groups = sorted(set(extraction.group_ids))
    
    out = {
        "by_layer": {},
        "per_group": {
            "groups": unique_groups,
            "by_layer": {},
        },
        "group_separation": {
            "by_layer": {},
        },
        "drift": {
            "mu_angle_deg": {},
            "mu_norm": {},
            "baseline_logits": {},
            "per_group_drift": {},
        },
        "summary": {},
    }

    # Build group index once
    group_to_idx = {}
    for i, g in enumerate(extraction.group_ids):
        group_to_idx.setdefault(g, []).append(i)

    mu_vecs = {}
    mu_vecs_by_group = {g: {} for g in unique_groups}
    b_logits = {}
    
    # Precompute per-layer means (global and per-group)
    for layer in layers_sorted:
        H = extraction.layers[layer].H
        
        # Global mean
        mu = H.mean(axis=0)
        mu_vecs[layer] = mu
        
        # Per-group means
        for g in unique_groups:
            idxs = group_to_idx[g]
            mu_g = H[idxs].mean(axis=0)
            mu_vecs_by_group[g][layer] = mu_g

        # Baseline logits (what model would predict from mean representation)
        if unembed_weight is not None:
            W = unembed_weight
            if W.shape[0] < W.shape[1]:
                W = W.T
            b = W @ mu  # (vocab,)
            b_logits[layer] = b

    # Compute per-layer decompositions
    for layer in layers_sorted:
        H = extraction.layers[layer].H
        mu = mu_vecs[layer]
        mu_norm = float(np.linalg.norm(mu))
        
        # ============================================================
        # GLOBAL μ ANALYSIS (original Part E)
        # ============================================================
        
        # P energy (union of all unique per-prompt P bases)
        P_union_cols = []
        seen_P = set()
        for i in range(H.shape[0]):
            Bp = P_bases[layer].get(i)
            if Bp is not None and Bp.size:
                # Deduplicate by checking if this P vector is already included
                key = tuple(Bp.flatten()[:4].tolist())  # Fast approximate dedup
                if key not in seen_P:
                    P_union_cols.append(Bp)
                    seen_P.add(key)
        Bp_union = np.concatenate(P_union_cols, axis=1) if P_union_cols else np.zeros((H.shape[1], 0), dtype=np.float32)
        Bp_union = orthonormal_basis_from_vectors(Bp_union) if Bp_union.size else Bp_union
        muP = project_onto_basis(mu[None, :], Bp_union)[0]
        EP = float(np.dot(muP, muP))

        # Compute per-sample Hperp for D and S definitions
        HP = np.zeros_like(H)
        for i, g in enumerate(extraction.group_ids):
            HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
        Hperp = H - HP
        
        # D basis: top-k PCA of H after projecting out P (sequential decomposition)
        pr = participation_ratio(Hperp)
        k = max(1, min(int(round(pr)), k_max, Hperp.shape[1], Hperp.shape[0]-1 if Hperp.shape[0] > 1 else 1))
        Bd = pca_basis(Hperp, k=k, center=True)

        # S basis: top-k PCA of H after projecting out P and D (sequential decomposition)
        mu_perp = mu - muP
        Hperp_D = Hperp - project_onto_basis(Hperp, Bd) if Bd.size else Hperp.copy()
        s_pr = participation_ratio(Hperp_D)
        s_k = max(1, min(int(round(s_pr)), s_small_r, Hperp_D.shape[1], Hperp_D.shape[0]-1 if Hperp_D.shape[0] > 1 else 1))
        Bs = pca_basis(Hperp_D, k=s_k, center=True) if s_k > 0 else np.zeros((H.shape[1], 0), dtype=np.float32)

        muD = project_onto_basis(mu_perp[None, :], Bd)[0]
        ED = float(np.dot(muD, muD))
        muS = project_onto_basis(mu_perp[None, :], Bs)[0] if Bs.size else np.zeros_like(mu_perp)
        ES = float(np.dot(muS, muS))
        
        # Compute fraction for convenience plotting
        total_mu_energy = float(np.dot(mu, mu))
        ES_frac = ES / max(1e-12, total_mu_energy)
        EP_frac = EP / max(1e-12, total_mu_energy)
        ED_frac = ED / max(1e-12, total_mu_energy)

        out["by_layer"][str(layer)] = {
            "mu_norm": mu_norm,
            "mu_energy_P": EP,
            "mu_energy_D": ED,
            "mu_energy_S": ES,
            "mu_energy_frac_P": EP_frac,
            "mu_energy_frac_D": ED_frac,
            "mu_energy_frac_S": ES_frac,
            "k_D": int(k),
            "k_S": int(s_k),
        }
        
        # ============================================================
        # PER-GROUP μ ANALYSIS
        # ============================================================
        
        per_group_layer = {}
        
        for g in unique_groups:
            mu_g = mu_vecs_by_group[g][layer]
            mu_g_norm = float(np.linalg.norm(mu_g))
            
            # Project μ_g onto P (union of P bases for group members)
            group_P_cols = []
            for idx in group_to_idx[g]:
                bp = P_bases[layer].get(idx)
                if bp is not None and bp.size:
                    group_P_cols.append(bp)
            if group_P_cols:
                Bp_g = np.concatenate(group_P_cols, axis=1)
                Bp_g = orthonormal_basis_from_vectors(Bp_g) if Bp_g.shape[1] > 1 else Bp_g
            else:
                Bp_g = np.zeros((H.shape[1], 0), dtype=np.float32)
            
            if Bp_g.size:
                mu_g_P = project_onto_basis(mu_g[None, :], Bp_g)[0]
            else:
                mu_g_P = np.zeros_like(mu_g)
            E_g_P = float(np.dot(mu_g_P, mu_g_P))
            
            # μ_g in P⊥
            mu_g_perp = mu_g - mu_g_P
            
            # Project onto D (global D basis)
            mu_g_D = project_onto_basis(mu_g_perp[None, :], Bd)[0]
            E_g_D = float(np.dot(mu_g_D, mu_g_D))
            
            # Project onto S (canonical S basis)
            mu_g_S = project_onto_basis(mu_g_perp[None, :], Bs)[0] if Bs.size else np.zeros_like(mu_g)
            E_g_S = float(np.dot(mu_g_S, mu_g_S))
            
            # Fractions
            total_g_energy = float(np.dot(mu_g, mu_g))
            
            per_group_layer[g] = {
                "mu_norm": mu_g_norm,
                "mu_energy_P": E_g_P,
                "mu_energy_D": E_g_D,
                "mu_energy_S": E_g_S,
                "mu_energy_frac_P": E_g_P / max(1e-12, total_g_energy),
                "mu_energy_frac_D": E_g_D / max(1e-12, total_g_energy),
                "mu_energy_frac_S": E_g_S / max(1e-12, total_g_energy),
                "n_samples": len(group_to_idx[g]),
            }
        
        out["per_group"]["by_layer"][str(layer)] = per_group_layer
        
        # ============================================================
        # GROUP CENTROID SEPARATION
        # ============================================================
        
        separation_layer = {
            "pairwise_distances": {},
            "pairwise_angles_deg": {},
            "pairwise_D_distances": {},
            "pairwise_S_distances": {},
            "centroid_spread": {},
        }
        
        # Compute pairwise distances and angles between group centroids
        group_list = list(unique_groups)
        all_distances = []
        all_D_distances = []
        all_S_distances = []
        all_F_distances = []
        
        for ii, g1 in enumerate(group_list):
            for g2 in group_list[ii+1:]:
                mu_1 = mu_vecs_by_group[g1][layer]
                mu_2 = mu_vecs_by_group[g2][layer]
                
                # Euclidean distance
                dist = float(np.linalg.norm(mu_1 - mu_2))
                separation_layer["pairwise_distances"][f"{g1}_vs_{g2}"] = dist
                all_distances.append(dist)
                
                # Angle between centroids
                norm1 = np.linalg.norm(mu_1)
                norm2 = np.linalg.norm(mu_2)
                if norm1 > 1e-12 and norm2 > 1e-12:
                    cos = float(np.dot(mu_1, mu_2) / (norm1 * norm2))
                    cos = max(-1.0, min(1.0, cos))
                    angle = float(np.degrees(np.arccos(cos)))
                else:
                    angle = 0.0
                separation_layer["pairwise_angles_deg"][f"{g1}_vs_{g2}"] = angle
                
                # Distance in D subspace
                # Build P union for each group
                Bp_g1_cols = [P_bases[layer].get(idx) for idx in group_to_idx.get(g1, []) if P_bases[layer].get(idx) is not None]
                Bp_g1 = orthonormal_basis_from_vectors(np.concatenate(Bp_g1_cols, axis=1)) if Bp_g1_cols else np.zeros((H.shape[1], 0), dtype=np.float32)
                Bp_g2_cols = [P_bases[layer].get(idx) for idx in group_to_idx.get(g2, []) if P_bases[layer].get(idx) is not None]
                Bp_g2 = orthonormal_basis_from_vectors(np.concatenate(Bp_g2_cols, axis=1)) if Bp_g2_cols else np.zeros((H.shape[1], 0), dtype=np.float32)
                mu_1_P = project_onto_basis(mu_1[None, :], Bp_g1)[0] if Bp_g1.size else np.zeros_like(mu_1)
                mu_2_P = project_onto_basis(mu_2[None, :], Bp_g2)[0] if Bp_g2.size else np.zeros_like(mu_2)
                mu_1_perp = mu_1 - mu_1_P
                mu_2_perp = mu_2 - mu_2_P
                mu_1_D = project_onto_basis(mu_1_perp[None, :], Bd)[0]
                mu_2_D = project_onto_basis(mu_2_perp[None, :], Bd)[0]
                D_dist = float(np.linalg.norm(mu_1_D - mu_2_D))
                separation_layer["pairwise_D_distances"][f"{g1}_vs_{g2}"] = D_dist
                all_D_distances.append(D_dist)
                
                # Distance in S subspace (project onto canonical S basis, NOT entire P⊥-D residual)
                mu_1_S = project_onto_basis(mu_1_perp[None, :] - mu_1_D[None, :], Bs)[0] if Bs.size else np.zeros_like(mu_1)
                mu_2_S = project_onto_basis(mu_2_perp[None, :] - mu_2_D[None, :], Bs)[0] if Bs.size else np.zeros_like(mu_2)
                S_dist = float(np.linalg.norm(mu_1_S - mu_2_S))
                separation_layer["pairwise_S_distances"][f"{g1}_vs_{g2}"] = S_dist
                all_S_distances.append(S_dist)
                
                # Distance in F (residual after P, D, S)
                mu_1_F = (mu_1_perp - mu_1_D) - mu_1_S
                mu_2_F = (mu_2_perp - mu_2_D) - mu_2_S
                F_dist = float(np.linalg.norm(mu_1_F - mu_2_F))
                separation_layer.setdefault("pairwise_F_distances", {})[f"{g1}_vs_{g2}"] = F_dist
                all_F_distances.append(F_dist)
        
        # Summary statistics
        if all_distances:
            separation_layer["centroid_spread"] = {
                "mean_distance": float(np.mean(all_distances)),
                "std_distance": float(np.std(all_distances)),
                "max_distance": float(np.max(all_distances)),
                "min_distance": float(np.min(all_distances)),
                "mean_D_distance": float(np.mean(all_D_distances)),
                "mean_S_distance": float(np.mean(all_S_distances)),
                "mean_F_distance": float(np.mean(all_F_distances)) if all_F_distances else 0.0,
                "D_to_S_ratio": float(np.mean(all_D_distances) / max(1e-12, np.mean(all_S_distances))),
            }
        
        out["group_separation"]["by_layer"][str(layer)] = separation_layer

    # ============================================================
    # DRIFT METRICS ACROSS LAYERS
    # ============================================================
    
    # Global μ drift
    for a, b in zip(layers_sorted[:-1], layers_sorted[1:]):
        mua = mu_vecs[a]
        mub = mu_vecs[b]
        denom = (np.linalg.norm(mua) * np.linalg.norm(mub) + 1e-12)
        cos = float(np.dot(mua, mub) / denom)
        cos = max(-1.0, min(1.0, cos))
        out["drift"]["mu_angle_deg"][f"{a}->{b}"] = float(np.degrees(np.arccos(cos)))
        out["drift"]["mu_norm"][str(a)] = float(np.linalg.norm(mua))
    out["drift"]["mu_norm"][str(layers_sorted[-1])] = float(np.linalg.norm(mu_vecs[layers_sorted[-1]]))

    # Per-group μ drift
    for g in unique_groups:
        out["drift"]["per_group_drift"][g] = {"angle_deg": {}, "norm": {}}
        for a, b in zip(layers_sorted[:-1], layers_sorted[1:]):
            mua = mu_vecs_by_group[g][a]
            mub = mu_vecs_by_group[g][b]
            denom = (np.linalg.norm(mua) * np.linalg.norm(mub) + 1e-12)
            cos = float(np.dot(mua, mub) / denom)
            cos = max(-1.0, min(1.0, cos))
            out["drift"]["per_group_drift"][g]["angle_deg"][f"{a}->{b}"] = float(np.degrees(np.arccos(cos)))
            out["drift"]["per_group_drift"][g]["norm"][str(a)] = float(np.linalg.norm(mua))
        out["drift"]["per_group_drift"][g]["norm"][str(layers_sorted[-1])] = float(np.linalg.norm(mu_vecs_by_group[g][layers_sorted[-1]]))

    # Baseline logits drift (unchanged)
    if b_logits:
        for a, b in zip(layers_sorted[:-1], layers_sorted[1:]):
            ba = b_logits[a]
            bb = b_logits[b]
            denom = (np.linalg.norm(ba) * np.linalg.norm(bb) + 1e-12)
            cos = float(np.dot(ba, bb) / denom)
            out["drift"]["baseline_logits"][f"{a}->{b}"] = {
                "cosine": cos,
                "kl_a_to_b": kl_divergence_from_logits(ba, bb),
                "entropy_a": entropy_from_logits(ba),
                "entropy_b": entropy_from_logits(bb),
            }

    # ============================================================
    # SUMMARY
    # ============================================================
    
    last = layers_sorted[-1]
    last_separation = out["group_separation"]["by_layer"][str(last)]
    
    # Per-group summary at last layer
    per_group_summary = {}
    for g in unique_groups:
        pg = out["per_group"]["by_layer"][str(last)][g]
        per_group_summary[g] = {
            "mu_energy_frac_P": pg["mu_energy_frac_P"],
            "mu_energy_frac_D": pg["mu_energy_frac_D"],
            "mu_energy_frac_S": pg["mu_energy_frac_S"],
        }
    
    out["summary"] = {
        "last_layer": int(last),
        # Global μ
        **out["by_layer"][str(last)],
        # Group separation summary
        "group_centroid_mean_distance": last_separation["centroid_spread"].get("mean_distance", 0),
        "group_centroid_D_to_S_ratio": last_separation["centroid_spread"].get("D_to_S_ratio", 0),
        # Per-group at last layer
        "per_group": per_group_summary,
    }
    
    return out

def compute_part_F_rotation(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    k_max: int = 64,
) -> Dict[str, Any]:
    """Part F: Analyze rotation of D subspace between consecutive layers.
    
    Measures how the D subspace rotates as representations propagate through
    the model. This tests whether D "tracks" the evolving representation by
    rotating to maintain variance capture, rather than being a fixed subspace.
    
    Key insight: If D at layer L+1 is rotated relative to D at layer L, but
    both capture similar variance fractions, the model is adapting its
    "steering dimensions" to the changing representation geometry.
    
    Note: This function returns AGGREGATE statistics only. For per-prompt
    rotation trajectories (needed for family clustering analysis), use
    compute_part_F_rotation_with_per_prompt() instead.
    
    Args:
        extraction: Model extraction
        P_bases: P bases per layer and group
        k_max: Maximum D dimensionality
        
    Returns:
        Dict with:
        - transitions: List of layer→layer transition data
        - by_layer: Dict for easy per-layer plotting
        - rotation_mean_deg: Overall mean rotation
    """
    layers_sorted = sorted(extraction.layers.keys())
    # Precompute D bases per layer
    D_bases = {}
    Hperp_by_layer = {}
    for layer in layers_sorted:
        H = extraction.layers[layer].H
        HP = np.zeros_like(H)
        for i, g in enumerate(extraction.group_ids):
            HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
        Hperp = H - HP
        Hperp_by_layer[layer] = Hperp
        pr = participation_ratio(Hperp)
        k = max(1, min(int(round(pr)), k_max, Hperp.shape[1], Hperp.shape[0]-1 if Hperp.shape[0] > 1 else 1))
        D_bases[layer] = pca_basis(Hperp, k=k, center=True)

    transitions = []
    by_layer = {}  # For easier plotting
    
    for a, b in zip(layers_sorted[:-1], layers_sorted[1:]):
        U = D_bases[a]
        V = D_bases[b]
        ang = principal_angles(U, V)
        mean_ang = float(np.mean(ang)) if ang.size else float('nan')
        
        # Variance captured in layer b using rotated vs frozen basis
        Xb = Hperp_by_layer[b]
        Xb_c = Xb - Xb.mean(axis=0, keepdims=True)
        total_var = float((Xb_c ** 2).sum())
        
        # Rotated capture (use V, the basis fit to layer b)
        rot = project_onto_basis(Xb_c, V)
        rot_var = float((rot ** 2).sum())
        
        # Frozen capture (use U, the basis from layer a)
        fro = project_onto_basis(Xb_c, U)
        fro_var = float((fro ** 2).sum())
        
        gain = (rot_var - fro_var) / max(1e-12, fro_var)
        
        transition_data = {
            "from_layer": int(a),
            "to_layer": int(b),
            "mean_principal_angle_deg": mean_ang,
            "rotated_var_frac": rot_var / max(1e-12, total_var),
            "frozen_var_frac": fro_var / max(1e-12, total_var),
            "variance_gain_vs_frozen": gain,
        }
        transitions.append(transition_data)
        
        # by_layer entry for easier plotting
        by_layer[str(a)] = {
            "rotation_deg_to_next": mean_ang,
            "mean_principal_angle_deg": mean_ang,
            "to_layer": int(b),
            "variance_gain": gain,
        }
    
    # Summary statistics
    mean_rot = float(np.nanmean([t["mean_principal_angle_deg"] for t in transitions])) if transitions else float('nan')
    mean_gain = float(np.nanmean([t["variance_gain_vs_frozen"] for t in transitions])) if transitions else float('nan')
    
    return {
        "transitions": transitions,
        "by_layer": by_layer,
        "summary": {"mean_rotation_deg": mean_rot, "mean_gain": mean_gain},
        # Top-level convenience key
        "rotation_mean_deg": mean_rot,
    }


def _compute_aggregate_analysis(
    per_prompt_data: List[Dict],
    extraction: 'ModelExtraction',
    layers_sorted: List[int],
) -> Dict[str, Any]:
    """Compute aggregate analysis metrics from per-prompt trajectory data.
    
    This function computes summary statistics that would otherwise require
    post-hoc analysis of the trajectory data. By computing them during the
    main analysis run, we save reprocessing time and ensure consistency.
    
    Returns dict with:
        - critical_layers: Peak/min locations for key metrics
        - factor_effects: Cohen's d for each factor at key layers
        - group_separation: Within vs between group correlations
        - identical_trajectories: Prompt pairs with identical metrics
        - phase_summary: Early/middle/late layer statistics
    """
    from collections import defaultdict
    
    n_prompts = len(per_prompt_data)
    n_layers = len(layers_sorted)
    metrics = ['P_energy_frac', 'D_energy_frac', 'S_energy_frac', 'F_energy_frac',
               'PR_P', 'PR_D', 'PR_S',
               'D_rotation_from_prev_deg', 'S_rotation_from_prev_deg',
               'P_rotation_from_prev_deg', 'F_norm_ratio']
    
    # Build trajectory arrays
    trajectories = {m: np.full((n_prompts, n_layers), np.nan) for m in metrics}
    prompt_groups = []
    prompt_variants = []
    
    for i, prompt in enumerate(per_prompt_data):
        prompt_groups.append(prompt['group_id'])
        prompt_variants.append(prompt['variant_id'])
        for j, layer in enumerate(layers_sorted):
            layer_data = prompt['trajectory'].get(str(layer), {})
            for m in metrics:
                val = layer_data.get(m)
                if val is not None:
                    trajectories[m][i, j] = val
    
    prompt_groups = np.array(prompt_groups)
    prompt_variants = np.array(prompt_variants)
    unique_groups = np.unique(prompt_groups)
    
    # === CRITICAL LAYERS ===
    D_energy = trajectories['D_energy_frac']
    D_mean = np.nanmean(D_energy, axis=0)
    peak_D_idx = int(np.nanargmax(D_mean))
    peak_D_layer = int(layers_sorted[peak_D_idx])
    
    P_energy = trajectories['P_energy_frac']
    P_mean = np.nanmean(P_energy, axis=0)
    max_P_idx = int(np.nanargmax(P_mean))
    max_P_layer = int(layers_sorted[max_P_idx])
    
    rotation = trajectories['D_rotation_from_prev_deg']
    # Layer 0 has no previous layer, so its column is all-NaN by construction and
    # np.nanmean warns "Mean of empty slice". The NaN is expected, not an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rot_mean = np.nanmean(rotation, axis=0)
    # Skip first layer (NaN), find min
    valid_rot = rot_mean.copy()
    valid_rot[0] = np.nan
    min_rot_idx = np.nanargmin(valid_rot) if not np.all(np.isnan(valid_rot)) else 0
    min_rot_layer = int(layers_sorted[min_rot_idx]) if not np.isnan(valid_rot[min_rot_idx]) else None
    
    # S rotation critical layer
    S_rotation = trajectories['S_rotation_from_prev_deg']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S_rot_mean = np.nanmean(S_rotation, axis=0)
    valid_S_rot = S_rot_mean.copy()
    valid_S_rot[0] = np.nan
    min_S_rot_idx = np.nanargmin(valid_S_rot) if not np.all(np.isnan(valid_S_rot)) else 0
    min_S_rot_layer = int(layers_sorted[min_S_rot_idx]) if not np.isnan(valid_S_rot[min_S_rot_idx]) else None
    
    # F_energy critical layer (minimum = max structured capture)
    F_energy = trajectories['F_energy_frac']
    F_mean = np.nanmean(F_energy, axis=0)
    min_F_idx = int(np.nanargmin(F_mean)) if not np.all(np.isnan(F_mean)) else 0
    min_F_layer = int(layers_sorted[min_F_idx])
    
    critical_layers = {
        "peak_D_energy": {
            "layer": peak_D_layer,
            "frac_depth": round(peak_D_layer / max(layers_sorted) if layers_sorted else 0, 3),
            "value": round(float(D_mean[peak_D_idx]), 6),
        },
        "max_P_energy": {
            "layer": max_P_layer,
            "value": round(float(P_mean[max_P_idx]), 6),
        },
        "min_D_rotation": {
            "layer": min_rot_layer,
            "value": round(float(np.nanmin(valid_rot)), 2) if min_rot_layer else None,
        },
        "min_S_rotation": {
            "layer": min_S_rot_layer,
            "value": round(float(np.nanmin(valid_S_rot)), 2) if min_S_rot_layer else None,
        },
        "min_F_energy": {
            "layer": min_F_layer,
            "value": round(float(F_mean[min_F_idx]), 6),
        },
    }
    
    # === FACTOR EFFECTS ===
    def parse_factors(variant_id):
        """Parse variant ID like 'G01_A0B0C1D1' into factor dict."""
        code = variant_id.split('_')[-1]
        factors = {}
        idx = 0
        while idx < len(code):
            if code[idx].isalpha():
                f = code[idx]
                level = ''
                idx += 1
                while idx < len(code) and code[idx].isdigit():
                    level += code[idx]
                    idx += 1
                factors[f] = int(level) if level else 0
        return factors
    
    prompt_factors = {f: [] for f in ['A', 'B', 'C', 'D']}
    for v in prompt_variants:
        factors = parse_factors(v)
        for f in ['A', 'B', 'C', 'D']:
            prompt_factors[f].append(factors.get(f, -1))
    for f in ['A', 'B', 'C', 'D']:
        prompt_factors[f] = np.array(prompt_factors[f])
    
    # Compute at key layers: peak D and final layer
    key_layers_idx = [peak_D_idx, len(layers_sorted) - 1]
    key_layers = [layers_sorted[i] for i in key_layers_idx]
    
    factor_effects = {"by_layer": {}, "summary": {}}
    for layer_idx, layer in zip(key_layers_idx, key_layers):
        layer_effects = {}
        for f in ['A', 'B', 'C', 'D']:
            level0_mask = prompt_factors[f] == 0
            level1_mask = prompt_factors[f] == 1
            
            if level0_mask.sum() < 2 or level1_mask.sum() < 2:
                layer_effects[f] = {"cohen_d": None}
                continue
            
            vals0 = D_energy[level0_mask, layer_idx]
            vals1 = D_energy[level1_mask, layer_idx]
            
            valid0 = ~np.isnan(vals0)
            valid1 = ~np.isnan(vals1)
            
            if valid0.sum() > 1 and valid1.sum() > 1:
                pooled_std = np.sqrt((np.nanvar(vals0) + np.nanvar(vals1)) / 2)
                cohen_d = (np.nanmean(vals1) - np.nanmean(vals0)) / pooled_std if pooled_std > 1e-12 else 0
                layer_effects[f] = {"cohen_d": round(float(cohen_d), 4)}
            else:
                layer_effects[f] = {"cohen_d": None}
        
        factor_effects["by_layer"][str(layer)] = layer_effects
    
    # Find dominant factor at peak D layer
    peak_layer_str = str(peak_D_layer)
    if peak_layer_str in factor_effects["by_layer"]:
        peak_effects = factor_effects["by_layer"][peak_layer_str]
        valid_effects = {f: abs(v["cohen_d"]) for f, v in peak_effects.items() if v["cohen_d"] is not None}
        if valid_effects:
            dominant = max(valid_effects, key=valid_effects.get)
            invariant = [f for f, d in valid_effects.items() if d < 0.1]
            factor_effects["summary"] = {
                "dominant_factor": dominant,
                "dominant_cohen_d": round(valid_effects[dominant], 4),
                "invariant_factors": invariant,
            }
    
    # === GROUP SEPARATION ===
    group_separation = {}
    for metric_name in ['P_energy_frac', 'D_energy_frac', 'S_energy_frac', 'F_energy_frac',
                         'D_rotation_from_prev_deg', 'S_rotation_from_prev_deg',
                         'PR_D', 'PR_S', 'PR_P', 'F_norm_ratio']:
        traj = trajectories[metric_name]
        if metric_name.endswith('_prev_deg'):
            traj = traj[:, 1:]  # Skip first layer
        
        valid_mask = ~np.isnan(traj).any(axis=1)
        if valid_mask.sum() < 3:
            continue
        
        traj_clean = traj[valid_mask]
        groups_clean = prompt_groups[valid_mask]
        
        if len(traj_clean) < 2:
            continue
        
        # A constant trajectory row has zero variance, so np.corrcoef divides by 0
        # and warns. The resulting NaN correlations are dropped below rather than
        # averaged in; the warning itself carries no information here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_matrix = np.corrcoef(traj_clean)
        
        within_corrs = []
        between_corrs = []
        for i in range(len(traj_clean)):
            for j in range(i+1, len(traj_clean)):
                _c = corr_matrix[i, j]
                if not np.isfinite(_c):
                    continue          # zero-variance row: undefined, not zero
                if groups_clean[i] == groups_clean[j]:
                    within_corrs.append(_c)
                else:
                    between_corrs.append(_c)
        
        if within_corrs and between_corrs:
            within_mean = float(np.mean(within_corrs))
            between_mean = float(np.mean(between_corrs))
            pooled_std = np.sqrt((np.var(within_corrs) + np.var(between_corrs)) / 2)
            cohen_d = (within_mean - between_mean) / pooled_std if pooled_std > 1e-12 else 0
            
            group_separation[metric_name] = {
                "within_r": round(within_mean, 4),
                "between_r": round(between_mean, 4),
                "cohen_d": round(float(cohen_d), 4),
            }
    
    # === IDENTICAL TRAJECTORIES ===
    identical_pairs = []
    checked_metrics = ['P_energy_frac', 'D_energy_frac', 'S_energy_frac', 'F_energy_frac',
                       'PR_P', 'PR_D', 'PR_S', 'D_rotation_from_prev_deg',
                       'S_rotation_from_prev_deg', 'P_rotation_from_prev_deg']
    
    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            all_identical = True
            for m in checked_metrics:
                traj_i = trajectories[m][i]
                traj_j = trajectories[m][j]
                # Skip NaN comparison
                valid = ~(np.isnan(traj_i) | np.isnan(traj_j))
                if valid.sum() > 0:
                    max_diff = np.abs(traj_i[valid] - traj_j[valid]).max()
                    if max_diff > 1e-9:
                        all_identical = False
                        break
            
            if all_identical:
                # Determine which factor differs
                f_i = parse_factors(prompt_variants[i])
                f_j = parse_factors(prompt_variants[j])
                diff_factors = [f for f in ['A', 'B', 'C', 'D'] if f_i.get(f) != f_j.get(f)]
                
                identical_pairs.append({
                    "idx1": int(i),
                    "idx2": int(j),
                    "group": prompt_groups[i],
                    "variant1": prompt_variants[i],
                    "variant2": prompt_variants[j],
                    "differs_in": diff_factors,
                })
    
    identical_summary = {
        "count": len(identical_pairs),
        "all_factor_A_only": all(p["differs_in"] == ["A"] for p in identical_pairs) if identical_pairs else None,
        "affected_groups": list(set(p["group"] for p in identical_pairs)),
        "pairs": identical_pairs[:20],  # Limit to first 20 for file size
        "total_pairs": len(identical_pairs),
    }
    
    # === PHASE SUMMARY ===
    n_layers_total = len(layers_sorted)
    early_end = n_layers_total // 4
    late_start = 3 * n_layers_total // 4
    
    phase_summary = {
        "early": {
            "layers": [int(layers_sorted[0]), int(layers_sorted[min(early_end, n_layers_total-1)])],
            "mean_D_energy": round(float(np.nanmean(D_energy[:, :early_end])), 6) if early_end > 0 else None,
            "mean_D_rotation": round(float(np.nanmean(rotation[:, 1:early_end])), 2) if early_end > 1 else None,
            "mean_S_rotation": round(float(np.nanmean(S_rotation[:, 1:early_end])), 2) if early_end > 1 else None,
            "mean_F_energy": round(float(np.nanmean(F_energy[:, :early_end])), 6) if early_end > 0 else None,
            "mean_F_norm_ratio": round(float(np.nanmean(trajectories['F_norm_ratio'][:, :early_end])), 6) if early_end > 0 else None,
        },
        "middle": {
            "layers": [int(layers_sorted[early_end]), int(layers_sorted[min(late_start, n_layers_total-1)])],
            "mean_D_energy": round(float(np.nanmean(D_energy[:, early_end:late_start])), 6) if late_start > early_end else None,
            "mean_D_rotation": round(float(np.nanmean(rotation[:, early_end:late_start])), 2) if late_start > early_end else None,
            "mean_S_rotation": round(float(np.nanmean(S_rotation[:, early_end:late_start])), 2) if late_start > early_end else None,
            "mean_F_energy": round(float(np.nanmean(F_energy[:, early_end:late_start])), 6) if late_start > early_end else None,
            "mean_F_norm_ratio": round(float(np.nanmean(trajectories['F_norm_ratio'][:, early_end:late_start])), 6) if late_start > early_end else None,
        },
        "late": {
            "layers": [int(layers_sorted[late_start]), int(layers_sorted[-1])],
            "mean_D_energy": round(float(np.nanmean(D_energy[:, late_start:])), 6) if late_start < n_layers_total else None,
            "mean_D_rotation": round(float(np.nanmean(rotation[:, late_start:])), 2) if late_start < n_layers_total else None,
            "mean_S_rotation": round(float(np.nanmean(S_rotation[:, late_start:])), 2) if late_start < n_layers_total else None,
            "mean_F_energy": round(float(np.nanmean(F_energy[:, late_start:])), 6) if late_start < n_layers_total else None,
            "mean_F_norm_ratio": round(float(np.nanmean(trajectories['F_norm_ratio'][:, late_start:])), 6) if late_start < n_layers_total else None,
        },
    }
    
    return {
        "critical_layers": critical_layers,
        "factor_effects": factor_effects,
        "group_separation": group_separation,
        "identical_trajectories": identical_summary,
        "phase_summary": phase_summary,
    }


# Key results: rotation_mean_deg (D subspace rotation between layers),
#   per-prompt P_energy_frac, D_energy_frac, S_energy_frac trajectories,
#   identical_trajectories (for Part H input)
# This is the most compute-intensive analysis function (~30% of analysis time)
def compute_part_F_rotation_with_per_prompt(
    extraction: ModelExtraction,
    P_bases: Dict[int, Dict[str, np.ndarray]],
    k_max: int = 64,
    s_small_r: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Part F Extended: Per-prompt P-D-S energy trajectories and D rotation analysis.
    
    This function computes comprehensive per-prompt trajectory data through the
    P-D-S decomposition across all layers, enabling detailed analysis of how
    individual prompts traverse the representational geometry.
    
    THEORETICAL BACKGROUND
    ----------------------
    The P-D-S decomposition partitions the hidden state h at each layer:
    
        h = h_P + h_D + h_S + h_residual
        
    Where:
        - P (Predictive): Span of unembedding vectors for expected tokens.
          Represents directions that directly contribute to next-token prediction.
          
        - D (Discursive): Top-k principal components of H_perp (P-orthogonal residual),
          where k = round(PR(H_perp)). Captures directions where prompts diverge
          from each other — the "what makes this prompt different" subspace.
          
        - S (Stylistic): Top-k PCA of H after projecting out P and D.
          Captures the next layer of variance after prediction and divergence.
          Represents slower-varying structure not explained by P or D.
          
        - Residual: What remains after removing P, D, S components.
    
    PER-PROMPT METRICS COMPUTED
    ---------------------------
    For each prompt at each layer, we compute:
    
    1. P_energy_frac = ||h_P||² / ||h||²
       Fraction of total energy in predictive subspace.
       High values indicate the representation is "ready to predict."
       
    2. D_energy_frac = ||h_D||² / ||h_perp||²
       Fraction of P-orthogonal energy captured by divergent subspace.
       High values indicate prompt has strong individuating signal.
       
    3. S_energy_frac = ||h_S||² / ||h_perp||²
       Fraction of P-orthogonal energy captured by semantic subspace.
       High values indicate prompt aligns with shared family structure.
       
    4. PR_P (Participation Ratio in P)
       Effective dimensionality of the prompt's P component.
       Computed as PR(h_P) = (Σ|c_i|²)² / Σ|c_i|⁴ where c_i are P coefficients.
       
    5. PR_D (Participation Ratio in D)
       Effective dimensionality of the prompt's D component.
       Computed as PR(h_D) = (Σ|c_i|²)² / Σ|c_i|⁴ where c_i are D coefficients.
       
    6. D_rotation_from_prev_deg (D rotation)
       Angle between this prompt's D projection at layer L vs L-1.
       Measures how much the prompt's "divergent direction" rotates.
       
    7. S_rotation_from_prev_deg (S rotation) 
       Angle between this prompt's S projection at layer L vs L-1.
       Measures how much the prompt's "stylistic direction" rotates.
       
    8. P_rotation_from_prev_deg (P rotation) [RENAMED from P_D_rotation_from_prev_deg]
       Angle between this prompt's P projection at layer L vs L-1.
       Measures how much the prompt's "predictive direction" rotates.
       
    9. F_energy_frac 
       Fraction of total energy NOT captured by P, D, or S.
       Tracks the unstructured high-dimensional residual.
       
    10. PR_S 
        Participation ratio of S coefficients — effective S dimensionality.
        
    11. F_norm_ratio 
        ||h_F|| / ||h_layer0|| — tracks whether the unstructured residual
        is actively compressed through layers, independent of P/D/S growth.
    
    AGGREGATE ANALYSIS [NEW]
    ------------------------
    The per_prompt_results now include an "aggregate_analysis" section with:
    - critical_layers: Peak D_energy, max P_energy, min rotation layer
    - factor_effects: Cohen's d for each factor (A/B/C/D) at key layers
    - group_separation: Within vs between group trajectory correlations
    - identical_trajectories: Prompt pairs with identical metrics
    - phase_summary: Early/middle/late layer statistics
    
    EXPECTED PATTERNS
    -----------------
    - Early layers: High D_energy_frac (representations diverging)
    - Late layers: High S_energy_frac (settling into semantic territory)
    - Within-family: Similar trajectories (clustering in trajectory space)
    - Between-family: Different trajectories (separation in trajectory space)
    
    Args:
        extraction: ModelExtraction containing hidden states and metadata
        P_bases: Dict mapping layer -> group_id -> P basis matrix
        k_max: Maximum dimensionality for D basis (default: 64)
        s_small_r: Maximum rank for S basis (default: 16)
        
    Returns:
        Tuple of (aggregate_results, per_prompt_results):
        
        aggregate_results: Dict with keys:
            - transitions: List of layer-to-layer rotation statistics
            - by_layer: Dict mapping layer -> aggregate metrics
            - summary: Overall summary statistics
            - rotation_mean_deg: Convenience key for mean rotation
            - per_prompt_file: Filename pointer to per-prompt data
            
        per_prompt_results: Dict with keys:
            - metadata: Comprehensive metadata for traceability
            - metric_definitions: Detailed explanation of each metric
            - prompts: List of per-prompt trajectory data
    
    Output File Structure
    ---------------------
    The per_prompt_results dict is designed to be saved as JSON with full
    traceability. See metadata and metric_definitions fields for documentation
    that travels with the data.
    """
    from datetime import datetime, timezone
    
    layers_sorted = sorted(extraction.layers.keys())
    n_prompts = len(extraction.group_ids)
    hidden_dim = extraction.layers[layers_sorted[0]].H.shape[1]
    
    # Build group index
    group_to_idx = {}
    for i, g in enumerate(extraction.group_ids):
        group_to_idx.setdefault(g, []).append(i)
    n_groups = len(group_to_idx)
    
    # ========================================
    # PRECOMPUTE BASES PER LAYER
    # ========================================
    # We need P, D, and S bases at each layer
    
    D_bases = {}
    S_bases = {}
    H_by_layer = {}
    HP_by_layer = {}
    Hperp_by_layer = {}
    k_D_per_layer = {}
    k_S_per_layer = {}
    
    for layer in layers_sorted:
        H = extraction.layers[layer].H
        H_by_layer[layer] = H
        
        # Project each prompt onto its group's P basis
        HP = np.zeros_like(H)
        for i, g in enumerate(extraction.group_ids):
            HP[i:i+1, :] = project_onto_basis(H[i:i+1, :], P_bases[layer].get(i, np.zeros((H.shape[1], 1), dtype=np.float32)))
        HP_by_layer[layer] = HP
        Hperp = H - HP
        Hperp_by_layer[layer] = Hperp
        
        # D basis: top-k PCs of H_perp where k = round(PR)
        pr = participation_ratio(Hperp, center=True)
        k = max(1, min(int(round(pr)), k_max, Hperp.shape[1], Hperp.shape[0]-1 if Hperp.shape[0] > 1 else 1))
        D_bases[layer] = pca_basis(Hperp, k=k, center=True)
        k_D_per_layer[layer] = k
        
        # S basis: top-k PCA of H after projecting out P and D (sequential decomposition)
        Hperp_D = Hperp - project_onto_basis(Hperp, D_bases[layer]) if D_bases[layer].size else Hperp.copy()
        s_pr = participation_ratio(Hperp_D, center=True)
        s_k = max(1, min(int(round(s_pr)), s_small_r, Hperp_D.shape[1], Hperp_D.shape[0]-1 if Hperp_D.shape[0] > 1 else 1))
        S_bases[layer] = pca_basis(Hperp_D, k=s_k, center=True) if s_k > 0 else np.zeros((hidden_dim, 0), dtype=np.float32)
        k_S_per_layer[layer] = S_bases[layer].shape[1]

    # ========================================
    # μ DIRECTION STABILITY (aggregate metric)
    # ========================================
    # Compute mean hidden state μ at each layer and decompose into PDSF components.
    # Then measure directional stability (cosine similarity) of μ and its components
    # between consecutive layers.

    mu_by_layer = {}
    mu_P_by_layer = {}
    mu_D_by_layer = {}
    mu_S_by_layer = {}
    mu_F_by_layer = {}
    mu_energy_by_layer = {}

    for layer in layers_sorted:
        H = H_by_layer[layer]
        mu = H.mean(axis=0)
        mu_by_layer[layer] = mu

        # Build multi-dimensional P basis (same approach as compute_resolution_geometry)
        unique_P_vecs = {}
        for idx in range(n_prompts):
            Bp = P_bases[layer].get(idx)
            if Bp is not None and Bp.size > 0:
                key = tuple(Bp.flatten()[:8].tolist())
                if key not in unique_P_vecs:
                    unique_P_vecs[key] = Bp.flatten()

        if unique_P_vecs:
            P_dir_matrix = np.stack(list(unique_P_vecs.values()), axis=1)
            if P_dir_matrix.shape[1] > 1:
                P_dir_matrix = orthonormal_basis_from_vectors(P_dir_matrix)
            else:
                nrm = np.linalg.norm(P_dir_matrix)
                P_dir_matrix = P_dir_matrix / max(nrm, 1e-12)
        else:
            P_dir_matrix = np.zeros((hidden_dim, 1), dtype=np.float32)

        # Sequential PDSF decomposition of μ
        mu_P = (P_dir_matrix @ (P_dir_matrix.T @ mu))
        mu_P_by_layer[layer] = mu_P

        mu_perp = mu - mu_P

        D = D_bases[layer]
        if D.size > 0:
            mu_D = D @ (D.T @ mu_perp)
        else:
            mu_D = np.zeros_like(mu)
        mu_D_by_layer[layer] = mu_D

        S = S_bases[layer]
        mu_perp_D = mu_perp - mu_D
        if S.size > 0:
            mu_S = S @ (S.T @ mu_perp_D)
        else:
            mu_S = np.zeros_like(mu)
        mu_S_by_layer[layer] = mu_S

        mu_F = mu - mu_P - mu_D - mu_S
        mu_F_by_layer[layer] = mu_F

        mu_energy = float(np.dot(mu, mu))
        mu_P_energy = float(np.dot(mu_P, mu_P))
        mu_D_energy = float(np.dot(mu_D, mu_D))
        mu_S_energy = float(np.dot(mu_S, mu_S))
        mu_F_energy = float(np.dot(mu_F, mu_F))

        mu_energy_by_layer[layer] = {
            "mu_norm": float(np.sqrt(mu_energy)),
            "mu_energy_P_frac": mu_P_energy / max(1e-12, mu_energy),
            "mu_energy_D_frac": mu_D_energy / max(1e-12, mu_energy),
            "mu_energy_S_frac": mu_S_energy / max(1e-12, mu_energy),
            "mu_energy_F_frac": mu_F_energy / max(1e-12, mu_energy),
        }

    # Compute μ direction stability between consecutive layers
    def _rotation_deg(v_prev, v_curr):
        """Rotation angle in degrees between two vectors. NaN if either is near-zero."""
        n_prev = np.linalg.norm(v_prev)
        n_curr = np.linalg.norm(v_curr)
        if n_prev < 1e-12 or n_curr < 1e-12:
            return float('nan')
        cos_val = np.dot(v_prev, v_curr) / (n_prev * n_curr)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    mu_stability_by_layer = {}
    mu_rotation_values = []

    for layer in layers_sorted:
        layer_entry = {
            "mu_norm": round(mu_energy_by_layer[layer]["mu_norm"], 6),
            "mu_energy_P_frac": round(mu_energy_by_layer[layer]["mu_energy_P_frac"], 6),
            "mu_energy_D_frac": round(mu_energy_by_layer[layer]["mu_energy_D_frac"], 6),
            "mu_energy_S_frac": round(mu_energy_by_layer[layer]["mu_energy_S_frac"], 6),
            "mu_energy_F_frac": round(mu_energy_by_layer[layer]["mu_energy_F_frac"], 6),
        }

        layer_idx = layers_sorted.index(layer)
        if layer_idx > 0:
            prev_layer = layers_sorted[layer_idx - 1]
            mu_rot = _rotation_deg(mu_by_layer[prev_layer], mu_by_layer[layer])
            mu_P_rot = _rotation_deg(mu_P_by_layer[prev_layer], mu_P_by_layer[layer])
            mu_D_rot = _rotation_deg(mu_D_by_layer[prev_layer], mu_D_by_layer[layer])
            mu_S_rot = _rotation_deg(mu_S_by_layer[prev_layer], mu_S_by_layer[layer])
            mu_F_rot = _rotation_deg(mu_F_by_layer[prev_layer], mu_F_by_layer[layer])

            layer_entry["mu_rotation_from_prev_deg"] = round(mu_rot, 4)
            layer_entry["mu_P_rotation_from_prev_deg"] = round(mu_P_rot, 4)
            layer_entry["mu_D_rotation_from_prev_deg"] = round(mu_D_rot, 4)
            layer_entry["mu_S_rotation_from_prev_deg"] = round(mu_S_rot, 4)
            layer_entry["mu_F_rotation_from_prev_deg"] = round(mu_F_rot, 4)

            if not np.isnan(mu_rot):
                mu_rotation_values.append(mu_rot)

        mu_stability_by_layer[str(layer)] = layer_entry

    _mu_stability_data = {
        "by_layer": mu_stability_by_layer,
        "rotation_values": mu_rotation_values,
    }

    # ========================================
    # AGGREGATE STATISTICS (D rotation)
    # ========================================
    transitions = []
    by_layer = {}
    
    for a, b in zip(layers_sorted[:-1], layers_sorted[1:]):
        U = D_bases[a]
        V = D_bases[b]
        ang = principal_angles(U, V)
        mean_ang = float(np.mean(ang)) if ang.size else float('nan')
        
        Xb = Hperp_by_layer[b]
        Xb_c = Xb - Xb.mean(axis=0, keepdims=True)
        total_var = float((Xb_c ** 2).sum())
        
        rot = project_onto_basis(Xb_c, V)
        rot_var = float((rot ** 2).sum())
        
        fro = project_onto_basis(Xb_c, U)
        fro_var = float((fro ** 2).sum())
        
        gain = (rot_var - fro_var) / max(1e-12, fro_var)
        
        transition_data = {
            "from_layer": int(a),
            "to_layer": int(b),
            "mean_principal_angle_deg": mean_ang,
            "rotated_var_frac": rot_var / max(1e-12, total_var),
            "frozen_var_frac": fro_var / max(1e-12, total_var),
            "variance_gain_vs_frozen": gain,
        }
        transitions.append(transition_data)
        
        by_layer[str(a)] = {
            "rotation_deg_to_next": mean_ang,
            "mean_principal_angle_deg": mean_ang,
            "to_layer": int(b),
            "variance_gain": gain,
            "k_D": k_D_per_layer[a],
            "k_S": k_S_per_layer[a],
        }
    
    # Add last layer info
    last_layer = layers_sorted[-1]
    by_layer[str(last_layer)] = {
        "k_D": k_D_per_layer[last_layer],
        "k_S": k_S_per_layer[last_layer],
    }
    
    mean_rot = float(np.nanmean([t["mean_principal_angle_deg"] for t in transitions])) if transitions else float('nan')
    mean_gain = float(np.nanmean([t["variance_gain_vs_frozen"] for t in transitions])) if transitions else float('nan')
    
    aggregate_results = {
        "transitions": transitions,
        "by_layer": by_layer,
        "summary": {"mean_rotation_deg": mean_rot, "mean_gain": mean_gain},
        "rotation_mean_deg": mean_rot,
        "per_prompt_file": "per_prompt_trajectories.json",
    }
    
    # --- Finalize μ stability summary (needs mean_rot from aggregate stats above) ---
    mu_rotation_values = _mu_stability_data["rotation_values"]
    mean_mu_rotation = float(np.nanmean(mu_rotation_values)) if mu_rotation_values else float('nan')

    if np.isfinite(mean_rot) and mean_rot > 1e-12:
        mu_D_rotation_ratio = mean_mu_rotation / mean_rot
    else:
        mu_D_rotation_ratio = float('nan')

    mu_rotation_all = [v for v in mu_rotation_values if np.isfinite(v)]
    mu_energy_F_values = [
        mu_energy_by_layer[l]["mu_energy_F_frac"] for l in layers_sorted
    ]

    mu_stability_summary = {
        "mean_mu_rotation_deg": round(mean_mu_rotation, 4),
        "mean_D_rotation_deg": round(mean_rot, 4),
        "mu_D_rotation_ratio": round(mu_D_rotation_ratio, 6),
        "mu_rotation_range": [
            round(min(mu_rotation_all), 4) if mu_rotation_all else float('nan'),
            round(max(mu_rotation_all), 4) if mu_rotation_all else float('nan'),
        ],
        "mu_energy_F_mean": round(float(np.mean(mu_energy_F_values)), 6),
        "mu_energy_F_final": round(mu_energy_F_values[-1], 6) if mu_energy_F_values else float('nan'),
    }

    aggregate_results["mu_stability"] = {
        "by_layer": _mu_stability_data["by_layer"],
        "summary": mu_stability_summary,
    }

    # ========================================
    # PER-PROMPT TRAJECTORIES
    # ========================================
    per_prompt_data = []
    
    for prompt_idx in range(n_prompts):
        prompt_trajectory = {}
        D_projections = {}  # Store for D rotation computation
        P_projections = {}  # Store for P rotation computation
        S_projections = {}  # Store for S rotation computation
        initial_h_norm = None  # Store for F_norm_ratio
        
        for layer_idx, layer in enumerate(layers_sorted):
            H = H_by_layer[layer]
            HP = HP_by_layer[layer]
            Hperp = Hperp_by_layer[layer]
            D = D_bases[layer]
            S = S_bases[layer]
            
            # Get this prompt's vectors
            h = H[prompt_idx]           # Full hidden state
            h_P = HP[prompt_idx]        # P component
            h_perp = Hperp[prompt_idx]  # P-orthogonal component
            
            # Project onto D and S bases
            h_D = project_onto_basis(h_perp.reshape(1, -1), D).flatten()
            h_S = project_onto_basis(h_perp.reshape(1, -1), S).flatten() if S.size else np.zeros_like(h_perp)
            
            D_projections[layer] = h_D
            P_projections[layer] = h_P  # Store for P rotation computation
            S_projections[layer] = h_S  # Store for S rotation computation
            
            # Cache initial h norm for F_norm_ratio
            if layer_idx == 0:
                initial_h_norm = float(np.linalg.norm(h))
            
            # Compute energies
            h_energy = float(np.dot(h, h))
            h_P_energy = float(np.dot(h_P, h_P))
            h_perp_energy = float(np.dot(h_perp, h_perp))
            h_D_energy = float(np.dot(h_D, h_D))
            h_S_energy = float(np.dot(h_S, h_S))
            
            # Compute energy fractions
            P_energy_frac = h_P_energy / max(1e-12, h_energy)
            D_energy_frac = h_D_energy / max(1e-12, h_perp_energy)
            S_energy_frac = h_S_energy / max(1e-12, h_perp_energy)
            
            # F energy: everything not in P, D, or S (as fraction of total)
            h_F_energy = max(0.0, h_energy - h_P_energy - h_D_energy - h_S_energy)
            F_energy_frac = h_F_energy / max(1e-12, h_energy)
            
            # Compute PR_P (participation ratio of P coefficients)
            # PR = (Σ|c_i|²)² / Σ|c_i|⁴
            P_basis = P_bases[layer].get(prompt_idx, np.zeros((H.shape[1], 1), dtype=np.float32))
            if P_basis.size > 0:
                P_coeffs = P_basis.T @ h  # Project onto P basis
                p2 = P_coeffs ** 2
                sum_p2 = p2.sum()
                sum_p4 = (p2 ** 2).sum()
                PR_P = float((sum_p2 ** 2) / max(1e-12, sum_p4)) if sum_p4 > 1e-12 else float('nan')
            else:
                PR_P = float('nan')
            
            # Compute PR_D (participation ratio of D coefficients)
            # PR = (Σ|c_i|²)² / Σ|c_i|⁴
            D_coeffs = D.T @ h_perp  # Project to get coefficients
            c2 = D_coeffs ** 2
            sum_c2 = c2.sum()
            sum_c4 = (c2 ** 2).sum()
            PR_D = float((sum_c2 ** 2) / max(1e-12, sum_c4)) if sum_c4 > 1e-12 else float('nan')
            
            # Compute PR_S (participation ratio of S coefficients)
            if S.size > 0:
                h_D_residual = h_perp - h_D  # S operates on D-orthogonal residual
                S_coeffs = S.T @ h_D_residual
                s2 = S_coeffs ** 2
                sum_s2 = s2.sum()
                sum_s4 = (s2 ** 2).sum()
                PR_S = float((sum_s2 ** 2) / max(1e-12, sum_s4)) if sum_s4 > 1e-12 else float('nan')
            else:
                PR_S = float('nan')
            
            # F_norm_ratio: ||h_F|| / ||h_layer0|| — tracks residual compression
            F_norm = float(np.sqrt(h_F_energy))
            F_norm_ratio = F_norm / max(1e-12, initial_h_norm)
            
            layer_data = {
                "P_energy_frac": round(P_energy_frac, 6),
                "D_energy_frac": round(D_energy_frac, 6),
                "S_energy_frac": round(S_energy_frac, 6),
                "F_energy_frac": round(F_energy_frac, 6),
                "PR_P": round(PR_P, 4),
                "PR_D": round(PR_D, 4),
                "PR_S": round(PR_S, 4),
                "F_norm_ratio": round(F_norm_ratio, 6),
            }
            
            # Compute rotation from previous layer
            if layer_idx > 0:
                prev_layer = layers_sorted[layer_idx - 1]
                
                # D rotation (existing)
                h_D_prev = D_projections[prev_layer]
                norm_D_prev = np.linalg.norm(h_D_prev)
                norm_D_curr = np.linalg.norm(h_D)
                
                if norm_D_prev > 1e-12 and norm_D_curr > 1e-12:
                    cos_angle_D = np.dot(h_D_prev, h_D) / (norm_D_prev * norm_D_curr)
                    cos_angle_D = np.clip(cos_angle_D, -1.0, 1.0)
                    D_rotation_deg = float(np.degrees(np.arccos(cos_angle_D)))
                else:
                    D_rotation_deg = float('nan')
                
                layer_data["D_rotation_from_prev_deg"] = round(D_rotation_deg, 2)
                
                # P rotation
                h_P_prev = P_projections[prev_layer]
                norm_P_prev = np.linalg.norm(h_P_prev)
                norm_P_curr = np.linalg.norm(h_P)
                
                if norm_P_prev > 1e-12 and norm_P_curr > 1e-12:
                    cos_angle_P = np.dot(h_P_prev, h_P) / (norm_P_prev * norm_P_curr)
                    cos_angle_P = np.clip(cos_angle_P, -1.0, 1.0)
                    P_rotation_deg = float(np.degrees(np.arccos(cos_angle_P)))
                else:
                    P_rotation_deg = float('nan')
                
                layer_data["P_rotation_from_prev_deg"] = round(P_rotation_deg, 2)
                
                # S rotation
                h_S_prev = S_projections[prev_layer]
                norm_S_prev = np.linalg.norm(h_S_prev)
                norm_S_curr = np.linalg.norm(h_S)
                
                if norm_S_prev > 1e-12 and norm_S_curr > 1e-12:
                    cos_angle_S = np.dot(h_S_prev, h_S) / (norm_S_prev * norm_S_curr)
                    cos_angle_S = np.clip(cos_angle_S, -1.0, 1.0)
                    S_rotation_deg = float(np.degrees(np.arccos(cos_angle_S)))
                else:
                    S_rotation_deg = float('nan')
                
                layer_data["S_rotation_from_prev_deg"] = round(S_rotation_deg, 2)
            
            prompt_trajectory[str(layer)] = layer_data
        
        per_prompt_data.append({
            "prompt_idx": prompt_idx,
            "group_id": extraction.group_ids[prompt_idx],
            "variant_id": extraction.variant_ids[prompt_idx],
            "trajectory": prompt_trajectory,
        })
    
    # ========================================
    # AGGREGATE ANALYSIS
    # ========================================
    aggregate_analysis = _compute_aggregate_analysis(
        per_prompt_data, extraction, layers_sorted
    )

    # ========================================
    # COMPREHENSIVE METADATA
    # ========================================
    per_prompt_results = {
        "metadata": {
            "description": "Per-prompt P-D-S-F energy trajectories with D/S/P rotation analysis ",
            "model_id": extraction.model_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_prompts": n_prompts,
            "n_groups": n_groups,
            "n_layers": len(layers_sorted),
            "hidden_dim": hidden_dim,
            "layers": [int(l) for l in layers_sorted],
            "groups": sorted(group_to_idx.keys()),
            "parameters": {
                "k_max": k_max,
                "s_small_r": s_small_r,
            },
            "basis_dimensions": {
                "k_D_per_layer": {str(k): v for k, v in k_D_per_layer.items()},
                "k_S_per_layer": {str(k): v for k, v in k_S_per_layer.items()},
            },
        },
        "metric_definitions": {
            "P_energy_frac": {
                "formula": "||h_P||^2 / ||h||^2",
                "description": "Fraction of total hidden state energy in the Predictive (P) subspace",
                "interpretation": "High values indicate representation is oriented toward next-token prediction",
                "range": "[0, 1]",
            },
            "D_energy_frac": {
                "formula": "||h_D||^2 / ||h_perp||^2",
                "description": "Fraction of P-orthogonal energy captured by Discursive (D) subspace",
                "interpretation": "High values indicate strong individuating/divergent signal for this prompt",
                "range": "[0, 1]",
            },
            "S_energy_frac": {
                "formula": "||h_S||^2 / ||h_perp||^2",
                "description": "Fraction of P-orthogonal energy captured by Stylistic (S) subspace",
                "interpretation": "High values indicate alignment with shared family semantic structure",
                "range": "[0, 1]",
            },
            "F_energy_frac": {
                "formula": "max(0, ||h||^2 - ||h_P||^2 - ||h_D||^2 - ||h_S||^2) / ||h||^2",
                "description": "Fraction of total energy in F (flat/residual) subspace — everything not in P, D, or S",
                "interpretation": "High-dimensional unstructured component; high values indicate P+D+S capture little",
                "range": "[0, 1]",
                "note": "Normalized to total energy (not h_perp), so P_energy_frac + (1-P_energy_frac)*(D_energy_frac+S_energy_frac) + F_energy_frac ≈ 1",
            },
            "PR_P": {
                "formula": "(sum(c_i^2))^2 / sum(c_i^4) where c_i are P basis coefficients",
                "description": "Participation ratio of the prompt's P projection coefficients",
                "interpretation": "Effective number of P dimensions used by this prompt",
                "range": "[1, k_P]",
            },
            "PR_D": {
                "formula": "(sum(c_i^2))^2 / sum(c_i^4) where c_i are D basis coefficients",
                "description": "Participation ratio of the prompt's D projection coefficients",
                "interpretation": "Effective number of D dimensions used by this prompt",
                "range": "[1, k_D]",
            },
            "PR_S": {
                "formula": "(sum(c_i^2))^2 / sum(c_i^4) where c_i are S basis coefficients",
                "description": "Participation ratio of the prompt's S projection coefficients",
                "interpretation": "Effective number of S dimensions used by this prompt",
                "range": "[1, k_S]",
            },
            "D_rotation_from_prev_deg": {
                "formula": "arccos(h_D[L-1] · h_D[L] / (||h_D[L-1]|| ||h_D[L]||))",
                "description": "Angle between D projections at consecutive layers",
                "interpretation": "How much the prompt's divergent direction rotates between layers",
                "range": "[0, 180] degrees",
                "note": "Not present for first layer",
            },
            "S_rotation_from_prev_deg": {
                "formula": "arccos(h_S[L-1] · h_S[L] / (||h_S[L-1]|| ||h_S[L]||))",
                "description": "Angle between S projections at consecutive layers",
                "interpretation": "How much the prompt's stylistic/semantic direction rotates between layers",
                "range": "[0, 180] degrees",
                "note": "Not present for first layer",
            },
            "P_rotation_from_prev_deg": {
                "formula": "arccos(h_P[L-1] · h_P[L] / (||h_P[L-1]|| ||h_P[L]||))",
                "description": "Angle between P projections at consecutive layers",
                "interpretation": "How much the prompt's predictive direction rotates between layers",
                "range": "[0, 180] degrees",
                "note": "Not present for first layer. Formerly mislabeled as P_D_rotation_from_prev_deg",
            },
            "F_norm_ratio": {
                "formula": "||h_F|| / ||h_layer0||",
                "description": "Norm of F component relative to initial hidden state norm",
                "interpretation": "Tracks whether unstructured residual is compressed or maintained through layers. Decreasing = active compression.",
                "range": "[0, inf)",
                "note": "Denominator is the prompt's layer-0 hidden state norm, constant across layers",
            },
            "mu_stability_note": {
                "description": "μ direction stability metrics are aggregate (not per-prompt). They measure how the mean hidden state μ = mean(H, axis=0) and its PDSF components rotate between consecutive layers.",
                "location": "Top-level 'mu_stability' key in this file (not inside each prompt's trajectory).",
            },
            "mu_rotation_from_prev_deg": {
                "formula": "arccos(μ[L-1] · μ[L] / (||μ[L-1]|| ||μ[L]||))",
                "description": "Angle between mean hidden state μ at consecutive layers",
                "interpretation": "Low values (near 0°) indicate the shared representation is directionally stable. Compare with D rotation (typically 30-60°).",
                "range": "[0, 180] degrees",
            },
            "mu_D_rotation_ratio": {
                "formula": "mean_mu_rotation_deg / mean_D_rotation_deg",
                "description": "Ratio of μ rotation to D subspace rotation across depth",
                "interpretation": "Values << 1 indicate the shared infrastructure is much more stable than the divergent subspace. E.g., ratio=0.05 means μ is 20× more stable than D.",
                "range": "[0, inf)",
            },
            "mu_energy_P_frac": {
                "formula": "||μ_P||² / ||μ||²",
                "description": "Fraction of μ energy in the Predictive subspace",
                "interpretation": "Expected to be small — different prompts predict different tokens, so P directions partially cancel in the mean.",
                "range": "[0, 1]",
            },
            "mu_energy_F_frac": {
                "formula": "||μ_F||² / ||μ||²",
                "description": "Fraction of μ energy in the F (residual) subspace",
                "interpretation": "Expected to be large — μ should primarily live in F (shared infrastructure).",
                "range": "[0, 1]",
            },
        },
        "subspace_definitions": {
            "P": "Span of unembedding vectors for expected answer tokens. Represents prediction-relevant directions.",
            "D": "Top-k PCs of P-orthogonal hidden states (H_perp), where k = round(PR(H_perp)). Captures cross-prompt variation.",
            "S": "Top-k PCA of H_perp after projecting out D. Sequential orthogonal decomposition capturing next layer of variance.",
            "F": "Residual: h - h_P - h_D - h_S. High-dimensional unstructured component not captured by P, D, or S.",
            "h_perp": "P-orthogonal component: h_perp = h - proj_P(h). The non-predictive residual.",
        },
        "aggregate_analysis": aggregate_analysis,
        "mu_stability": aggregate_results.get("mu_stability", {}),
        "prompts": per_prompt_data,
    }
    
    return aggregate_results, per_prompt_results


# ============================================================
# PART H: INJECTIVITY ANALYSIS
# ============================================================
#
# Analyzes whether the mapping prompt → h_final is injective.
# Uses final layer hidden states to assess:
#   1. Global injectivity - pairwise similarity across all prompts
#   2. Identical-metric pair analysis - do prompts with same trajectory
#      metrics have identical h vectors?
#
# This addresses the question: identical trajectory metrics could mean
# truly identical h vectors, or could mean different h vectors that
# happen to have the same projection statistics.

# Key results: n_potential_collisions, within_group_cosine vs between_group_cosine,
#   separation_cohen_d, identical_metric_pairs (from Part F)
def compute_part_H_injectivity(
    extraction,
    identical_pairs: List[Tuple[int, int]] = None,
    group_ids: List[str] = None,
    variant_ids: List[str] = None,
    final_layer: int = None,
    similarity_threshold: float = 0.9999,
) -> Dict[str, Any]:
    """
    Compute injectivity analysis on final hidden states.
    
    Args:
        extraction: Extraction object with .layers[layer].H containing hidden states
        identical_pairs: Optional list of (idx1, idx2) pairs flagged as having
                        identical trajectory metrics (from Part F aggregate_analysis)
        group_ids: List of group_id for each prompt
        variant_ids: List of variant_id for each prompt
        final_layer: Layer index for final h (default: max layer)
        similarity_threshold: Cosine similarity above which pairs are flagged
                             as potential collisions (default: 0.9999)
    
    Returns:
        Dict containing:
        - global_injectivity: Statistics on all pairwise h similarities
        - identical_metric_pairs: Detailed analysis of flagged pairs
        - potential_collisions: Pairs with very high similarity
    
    Output structure:
    {
        "metadata": {...},
        "global_injectivity": {
            "n_prompts": 173,
            "h_final_layer": 80,
            "within_group_cosine": {"mean": 0.85, "std": 0.05, "min": 0.72, "max": 0.95},
            "between_group_cosine": {"mean": 0.42, "std": 0.12, "min": 0.15, "max": 0.68},
            "separation_cohen_d": 3.2,
            "all_pairs_cosine": {"mean": 0.52, "std": 0.18, "min": 0.15, "max": 0.95},
            "all_pairs_l2": {"mean": 12.3, "std": 2.1, "min": 5.2, "max": 18.7},
        },
        "potential_collisions": [
            {"idx_1": 12, "idx_2": 13, "cosine": 0.9999, "l2_dist": 0.01, 
             "group_1": "N03", "group_2": "N03", "same_group": true},
            ...
        ],
        "identical_metric_pairs": {
            "n_pairs": 18,
            "pairs": [
                {"idx_1": 12, "idx_2": 13, "h_cosine": 1.0, "h_l2_dist": 0.0,
                 "h_identical": true, "group_id": "N03_logic_always",
                 "variant_1": "G01_A0B0C0D0", "variant_2": "G01_A1B0C0D0"},
                ...
            ],
            "summary": {
                "n_h_identical": 18,
                "n_h_different": 0,
                "mean_cosine": 1.0,
                "mean_l2_dist": 0.0
            }
        }
    }
    """
    import numpy as np
    from typing import Dict, Any, List, Tuple
    
    # Determine final layer
    if final_layer is None:
        final_layer = max(extraction.layers.keys()) if hasattr(extraction, 'layers') else extraction.n_layers - 1
    
    # Get final hidden states
    if hasattr(extraction, 'layers'):
        H_final = extraction.layers[final_layer].H  # Shape: (n_prompts, hidden_dim)
    else:
        # Handle dict-style extraction
        H_final = extraction[final_layer]
    
    n_prompts, hidden_dim = H_final.shape
    
    # Compute norms for cosine similarity
    norms = np.linalg.norm(H_final, axis=1, keepdims=True)
    H_normalized = H_final / (norms + 1e-12)
    
    # Compute full cosine similarity matrix
    cosine_matrix = H_normalized @ H_normalized.T
    
    # Compute L2 distance matrix (using identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b)
    norm_sq = (norms ** 2).flatten()
    l2_matrix = np.sqrt(np.maximum(
        norm_sq[:, None] + norm_sq[None, :] - 2 * (H_final @ H_final.T),
        0
    ))
    
    # Global statistics (upper triangle only to avoid double-counting)
    triu_indices = np.triu_indices(n_prompts, k=1)
    all_cosines = cosine_matrix[triu_indices]
    all_l2 = l2_matrix[triu_indices]
    
    global_stats = {
        "n_prompts": int(n_prompts),
        "h_final_layer": int(final_layer),
        "hidden_dim": int(hidden_dim),
        "all_pairs_cosine": {
            "mean": float(np.mean(all_cosines)),
            "std": float(np.std(all_cosines)),
            "min": float(np.min(all_cosines)),
            "max": float(np.max(all_cosines)),
            "median": float(np.median(all_cosines)),
        },
        "all_pairs_l2": {
            "mean": float(np.mean(all_l2)),
            "std": float(np.std(all_l2)),
            "min": float(np.min(all_l2)),
            "max": float(np.max(all_l2)),
            "median": float(np.median(all_l2)),
        },
    }
    
    # Within-group vs between-group analysis (if group_ids provided)
    if group_ids is not None and len(group_ids) == n_prompts:
        within_cosines = []
        between_cosines = []
        
        for i in range(n_prompts):
            for j in range(i + 1, n_prompts):
                if group_ids[i] == group_ids[j]:
                    within_cosines.append(cosine_matrix[i, j])
                else:
                    between_cosines.append(cosine_matrix[i, j])
        
        if within_cosines:
            global_stats["within_group_cosine"] = {
                "mean": float(np.mean(within_cosines)),
                "std": float(np.std(within_cosines)),
                "min": float(np.min(within_cosines)),
                "max": float(np.max(within_cosines)),
                "n_pairs": len(within_cosines),
            }
        
        if between_cosines:
            global_stats["between_group_cosine"] = {
                "mean": float(np.mean(between_cosines)),
                "std": float(np.std(between_cosines)),
                "min": float(np.min(between_cosines)),
                "max": float(np.max(between_cosines)),
                "n_pairs": len(between_cosines),
            }
        
        # Cohen's d for separation
        if within_cosines and between_cosines:
            within_mean = np.mean(within_cosines)
            between_mean = np.mean(between_cosines)
            pooled_std = np.sqrt(
                (np.var(within_cosines) * (len(within_cosines) - 1) + 
                 np.var(between_cosines) * (len(between_cosines) - 1)) /
                (len(within_cosines) + len(between_cosines) - 2)
            )
            if pooled_std > 1e-12:
                global_stats["separation_cohen_d"] = float((within_mean - between_mean) / pooled_std)
    
    # Find potential collisions (very high similarity)
    potential_collisions = []
    for i in range(n_prompts):
        for j in range(i + 1, n_prompts):
            if cosine_matrix[i, j] >= similarity_threshold:
                collision = {
                    "idx_1": int(i),
                    "idx_2": int(j),
                    "cosine": float(cosine_matrix[i, j]),
                    "l2_dist": float(l2_matrix[i, j]),
                }
                if group_ids is not None:
                    collision["group_1"] = group_ids[i]
                    collision["group_2"] = group_ids[j]
                    collision["same_group"] = group_ids[i] == group_ids[j]
                if variant_ids is not None:
                    collision["variant_1"] = variant_ids[i]
                    collision["variant_2"] = variant_ids[j]
                potential_collisions.append(collision)
    
    # Sort by cosine similarity descending
    potential_collisions.sort(key=lambda x: x["cosine"], reverse=True)
    
    # Analyze identical-metric pairs (from Part F)
    identical_metric_analysis = None
    if identical_pairs is not None and len(identical_pairs) > 0:
        pair_results = []
        for idx_1, idx_2 in identical_pairs:
            if idx_1 < n_prompts and idx_2 < n_prompts:
                h_cosine = float(cosine_matrix[idx_1, idx_2])
                h_l2_dist = float(l2_matrix[idx_1, idx_2])
                
                pair_info = {
                    "idx_1": int(idx_1),
                    "idx_2": int(idx_2),
                    "h_cosine": round(h_cosine, 8),
                    "h_l2_dist": round(h_l2_dist, 8),
                    "h_identical": h_l2_dist < 1e-6,  # Effectively zero
                }
                
                if group_ids is not None:
                    pair_info["group_id"] = group_ids[idx_1]
                if variant_ids is not None:
                    pair_info["variant_1"] = variant_ids[idx_1]
                    pair_info["variant_2"] = variant_ids[idx_2]
                
                pair_results.append(pair_info)
        
        # Compute summary
        if pair_results:
            cosines = [p["h_cosine"] for p in pair_results]
            l2_dists = [p["h_l2_dist"] for p in pair_results]
            n_identical = sum(1 for p in pair_results if p["h_identical"])
            
            identical_metric_analysis = {
                "n_pairs": len(pair_results),
                "pairs": pair_results,
                "summary": {
                    "n_h_identical": n_identical,
                    "n_h_different": len(pair_results) - n_identical,
                    "mean_cosine": float(np.mean(cosines)),
                    "std_cosine": float(np.std(cosines)),
                    "mean_l2_dist": float(np.mean(l2_dists)),
                    "std_l2_dist": float(np.std(l2_dists)),
                    "all_identical": n_identical == len(pair_results),
                }
            }
    
    return {
        "global_injectivity": global_stats,
        "potential_collisions": potential_collisions[:50],  # Limit to top 50
        "n_potential_collisions": len(potential_collisions),
        "identical_metric_pairs": identical_metric_analysis,
    }


# ============================================================
# RESULT SAVING UTILITIES
# ============================================================
#
# These utilities provide standardized saving of analysis results
# to the unified directory structure:
#   results/SpecA/{model_name}/part_{letter}_{name}.json
#
# Each result file includes a metadata header for traceability.

def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model_name: str,
    part_letter: str,
    part_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save analysis results to standardized JSON file with metadata.
    
    Creates the output directory if it doesn't exist and saves results
    with automatic metadata inclusion and consistent file naming.
    
    Directory Structure:
        results/
        └── SpecA/
            └── {model_name}/
                ├── part_a_pr.json
                ├── part_b_angles.json
                ├── part_c_factors.json
                ├── part_d_trajectory.json
                ├── part_e_mu_landscape.json
                ├── part_f_rotation.json
                └── part_g_scramble.json
    
    Args:
        results: Dict containing analysis results
        output_dir: Base output directory (e.g., "results/SpecA")
        model_name: Model identifier for subdirectory (e.g., "meta-llama__Llama-3.1-8B-Instruct")
        part_letter: Single letter (a-g) identifying the analysis part
        part_name: Descriptive name (e.g., "pr", "angles", "trajectory")
        metadata: Optional metadata dict to include (if not already in results)
        
    Returns:
        Path to saved file
        
    Example:
        >>> save_results(
        ...     results=part_a_results,
        ...     output_dir="results/SpecA",
        ...     model_name="meta-llama__Llama-3.1-8B-Instruct",
        ...     part_letter="a",
        ...     part_name="pr",
        ...     metadata=metadata_dict,
        ... )
        'results/SpecA/meta-llama__Llama-3.1-8B-Instruct/part_a_pr.json'
    """
    from pathlib import Path
    import json
    
    # Create model directory if not exists
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct filename: part_{letter}_{name}.json
    filename = f"part_{part_letter}_{part_name}.json"
    filepath = model_dir / filename
    
    # Add metadata if provided and not already present
    output = results.copy()
    if metadata is not None and "metadata" not in output:
        output = {"metadata": metadata, **output}
    
    # Save with pretty printing
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    return str(filepath)


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model_key: str,
    experiment_type: str,
    test_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save analysis results.
    
    New naming convention:
        {model_key}-{experiment_type}-{test_name}.json
        
    Examples:
        llama_8b-SpecA-part_a_pr.json
        llama_8b-SpecA-part_h_injectivity.json
        qwen_32b-SpecB-S_scramble.json
    
    Args:
        results: Dict containing analysis results
        output_dir: Full output directory path (e.g., "results/SpecA/llama_8b")
        model_key: Short model key (e.g., "llama_8b", "qwen_32b")
        experiment_type: "SpecA" or "SpecB"
        test_name: Test identifier (e.g., "part_a_pr", "S_scramble")
        metadata: Optional metadata dict to include
        
    Returns:
        Path to saved file
    """
    from pathlib import Path
    import json
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Construct filename with new convention
    filename = f"{model_key}-{experiment_type}-{test_name}.json"
    filepath = output_path / filename
    
    # Add metadata if provided and not already present
    output = results.copy()
    if metadata is not None and "metadata" not in output:
        output = {"metadata": metadata, **output}
    
    # Save with pretty printing
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    return str(filepath)


def get_output_path(
    base_dir: str,
    model_key: str,
    experiment_type: str = "SpecA",
) -> str:
    """Get the standardized output directory path using short model key.
    
    New directory structure uses short model keys for consistency between
    SpecA and SpecB, and for more readable paths.
    
    Args:
        base_dir: Base results directory (e.g., "/workspace/results")
        model_key: Short model key from MODEL_REGISTRY (e.g., "llama_8b", "qwen_32b")
        experiment_type: "SpecA" or "SpecB"
        
    Returns:
        Full path to model output directory
        
    Example:
        >>> get_output_path_v2("/workspace/results", "llama_8b", "SpecA")
        '/workspace/results/SpecA/llama_8b'
        >>> get_output_path_v2("/workspace/results", "qwen_32b", "SpecB")  
        '/workspace/results/SpecB/qwen_32b'
    """
    from pathlib import Path
    return str(Path(base_dir) / experiment_type / model_key)


def get_output_files(model_output_dir: str, model_key: str) -> Dict[str, str]:
    """Get dict of output file paths for all SpecA parts using new naming convention.
    
    Args:
        model_output_dir: Path to model's output directory
        model_key: Short model key (e.g., "llama_8b")
        
    Returns:
        Dict mapping part names to full file paths
        
    Example:
        >>> get_output_files_v2("/workspace/results/SpecA/llama_8b", "llama_8b")
        {
            "part_a": "/workspace/results/SpecA/llama_8b/llama_8b-SpecA-part_a_pr.json",
            "part_b": "/workspace/results/SpecA/llama_8b/llama_8b-SpecA-part_b_angles.json",
            ...
        }
    """
    from pathlib import Path
    
    base = Path(model_output_dir)
    return {
        "part_a": str(base / f"{model_key}-SpecA-part_a_pr.json"),
        "part_b": str(base / f"{model_key}-SpecA-part_b_angles.json"),
        "part_c": str(base / f"{model_key}-SpecA-part_c_factors.json"),
        "part_d": str(base / f"{model_key}-SpecA-part_d_trajectory.json"),
        "part_e": str(base / f"{model_key}-SpecA-part_e_mu_landscape.json"),
        "part_f": str(base / f"{model_key}-SpecA-part_f_rotation.json"),
        "part_g": str(base / f"{model_key}-SpecA-part_g_scramble.json"),
        "part_h": str(base / f"{model_key}-SpecA-part_h_injectivity.json"),
        "per_prompt": str(base / f"{model_key}-SpecA-per_prompt_trajectories.json"),
    }


def get_specB_output_files(model_output_dir: str, model_key: str) -> Dict[str, str]:
    """Get dict of output file paths for all SpecB tests using new naming convention.
    
    Args:
        model_output_dir: Path to model's SpecB output directory
        model_key: Short model key (e.g., "llama_8b")
        
    Returns:
        Dict mapping test names to full file paths
    """
    from pathlib import Path
    
    base = Path(model_output_dir)
    return {
        "baselines": str(base / f"{model_key}-SpecB-baselines.json"),
        "S_scramble": str(base / f"{model_key}-SpecB-S_scramble.json"),
        "D_scramble": str(base / f"{model_key}-SpecB-D_scramble.json"),
        "Random8_scramble": str(base / f"{model_key}-SpecB-Random8_scramble.json"),
    }



# ============================================================
# Part N — six-estimator scaling rank (ported from V11.6 post-persistent-hook, 2026-07-02)
# Feeds Paper 1 Table 1 / Figure 2 / §4.1. Pure numpy; reuses participation_ratio_from_cov_eigs above.
# ============================================================

def stable_rank(eigs: np.ndarray, eps: float = 1e-12) -> float:
    """Standard stable rank: Σλ / max(λ).
    
    Rudelson & Vershynin definition. Less sensitive to heavy tails than PR
    because it uses linear (not quadratic) weighting of eigenvalues.
    
    Input: eigenvalues of covariance matrix (not singular values).
    Must be non-negative.
    
    NOTE: An earlier version of the Part N report used Σλ² / (max λ)²,
    which is NOT the standard stable rank. This function uses the correct
    definition.
    """
    eigs = np.maximum(eigs, 0)
    max_eig = np.max(eigs)
    if max_eig < eps:
        return 0.0
    return float(np.sum(eigs) / (max_eig + eps))


def spectral_entropy_rank(eigs: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank via spectral entropy: exp(-Σ p_i log p_i).
    
    Roy & Bhattacharya (2007). Captures distributional flatness of the
    eigenvalue spectrum. A perfectly flat spectrum (all eigenvalues equal)
    gives rank = d. A single dominant eigenvalue gives rank ≈ 1.
    """
    eigs = np.maximum(eigs, 0)
    total = np.sum(eigs) + eps
    p = eigs / total
    p = p[p > eps]
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def explained_variance_rank(eigs: np.ndarray, threshold: float = 0.95) -> int:
    """Minimum k such that top-k eigenvalues explain ≥ threshold of total variance.
    
    Simple integer-valued rank measure. Useful as a complement to PR because
    it has a direct interpretation: "how many dimensions do you need to
    capture X% of the variance?"
    """
    eigs = np.sort(np.maximum(eigs, 0))[::-1]
    if len(eigs) == 0:
        return 0
    cumsum = np.cumsum(eigs)
    total = cumsum[-1]
    if total < 1e-12:
        return 0
    for k, c in enumerate(cumsum, 1):
        if c / total >= threshold:
            return k
    return len(eigs)


def compute_part_N_scaling_rank_suite(
    H_layer: np.ndarray,
    P_bases_for_layer: Optional[Dict[int, np.ndarray]],
    D_basis: np.ndarray,
    S_basis: np.ndarray,
    layer_label: str = "",
    n_bootstrap: int = 200,
    seed: int = 0,
    skip_F_bootstrap: bool = False,
) -> Dict[str, Any]:
    """Multi-estimator effective rank for scaling invariance stress-testing.
    
    Computes 7 rank estimators (PR, stable rank, spectral entropy rank,
    k@90%, k@95%, k@99%) and stores the normalized eigenvalue spectrum
    for cross-model Wasserstein comparison. Runs on D, S, and F subspaces.
    
    Bootstrap over prompts (rows of H_layer) for confidence intervals.
    
    Args:
        H_layer: (n_prompts, d_model) hidden states at one layer.
        P_bases_for_layer: Dict[prompt_idx, (d_model, k_P)] per-prompt P bases,
            OR None to skip P removal. This matches the notebook's
            P_bases[layer] structure from build_unified_P_bases().
        D_basis: (d_model, k_D) orthonormal D basis.
        S_basis: (d_model, k_S) orthonormal S basis.
        layer_label: Descriptive label (e.g., "L16_50pct").
        n_bootstrap: Number of bootstrap resamples over prompts.
        seed: Random seed for reproducibility.
        skip_F_bootstrap: If True, compute F point estimates only (no bootstrap
            CIs). F bootstrap is the dominant cost because it requires SVD on
            full (n_prompts × d_model) matrices. D and S bootstrap are cheap
            because they operate on projected (n_prompts × k_D) and
            (n_prompts × k_S) matrices where k_D, k_S << d_model.
    
    Returns:
        Dict with per-subspace rank estimates, bootstrap CIs, and
        normalized eigenvalue spectra. See output schema in Part N docs.
    """
    rng = np.random.RandomState(seed)
    n_prompts, d_model = H_layer.shape
    
    def _compute_eigs(X: np.ndarray) -> np.ndarray:
        """Covariance eigenvalues from SVD (numerically stable)."""
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.zeros(min(X.shape), dtype=np.float64)
        return (s ** 2) / max(1, X.shape[0] - 1)
    
    def _all_estimators(eigs: np.ndarray) -> Dict[str, float]:
        """Compute all 6 scalar rank estimators from eigenvalues."""
        return {
            "PR": float(participation_ratio_from_cov_eigs(eigs)),
            "stable_rank": float(stable_rank(eigs)),
            "entropy_rank": float(spectral_entropy_rank(eigs)),
            "k_90": int(explained_variance_rank(eigs, 0.90)),
            "k_95": int(explained_variance_rank(eigs, 0.95)),
            "k_99": int(explained_variance_rank(eigs, 0.99)),
        }
    
    def _analyze_subspace(X_proj: np.ndarray, do_bootstrap: bool = True) -> Dict[str, Any]:
        """Full analysis for one subspace: point estimates + optional bootstrap CIs + spectrum."""
        eigs = _compute_eigs(X_proj)
        point = _all_estimators(eigs)
        
        # Normalized spectrum for Wasserstein comparison
        norm_spec = eigs / (np.sum(eigs) + 1e-12)
        # Keep only non-negligible components (> 0.1% of total)
        significant = norm_spec[norm_spec > 1e-3]
        
        result = {}
        if do_bootstrap:
            # Bootstrap CIs
            boot_estimates = {k: [] for k in point.keys()}
            for _ in range(n_bootstrap):
                idx = rng.choice(n_prompts, size=n_prompts, replace=True)
                b_eigs = _compute_eigs(X_proj[idx])
                b_est = _all_estimators(b_eigs)
                for k, v in b_est.items():
                    boot_estimates[k].append(v)
            
            for k in point.keys():
                vals = np.array(boot_estimates[k])
                result[k] = {
                    "point": point[k],
                    "ci_low": float(np.percentile(vals, 2.5)),
                    "ci_high": float(np.percentile(vals, 97.5)),
                    "bootstrap_std": float(np.std(vals)),
                }
        else:
            # Point estimates only (no CIs)
            for k in point.keys():
                result[k] = {
                    "point": point[k],
                    "ci_low": None,
                    "ci_high": None,
                    "bootstrap_std": None,
                    "bootstrap_skipped": True,
                }
        
        result["normalized_spectrum"] = significant.tolist()
        result["n_significant_components"] = len(significant)
        
        return result
    
    # Project H_layer onto each subspace
    results = {"layer_label": layer_label, "n_prompts": n_prompts, 
               "n_bootstrap": n_bootstrap, "seed": seed, "subspaces": {}}
    
    # D subspace: project onto D basis
    H_D = H_layer @ D_basis  # (n_prompts, k_D)
    results["subspaces"]["D"] = _analyze_subspace(H_D, do_bootstrap=True)
    
    # S subspace: project onto S basis (after removing P and D)
    # Must follow the sequential orthogonal decomposition
    # P removal uses per-prompt P bases (each prompt has its own unembedding vector)
    H_noP = np.copy(H_layer)
    if P_bases_for_layer is not None:
        for i in range(n_prompts):
            Bp = P_bases_for_layer.get(i)
            if Bp is not None and Bp.size > 0:
                # Bp is (d_model, k_P), typically (d_model, 1)
                proj = (H_layer[i] @ Bp) @ Bp.T
                H_noP[i] = H_layer[i] - proj
    H_noPD = H_noP - (H_noP @ D_basis) @ D_basis.T
    H_S = H_noPD @ S_basis  # (n_prompts, k_S)
    results["subspaces"]["S"] = _analyze_subspace(H_S, do_bootstrap=True)
    
    # F subspace: full residual after removing P + D + S projections
    # F bootstrap is the dominant cost (SVD on n × d_model), so optionally skip it.
    H_noPDS = H_noPD - (H_noPD @ S_basis) @ S_basis.T  # (n_prompts, d_model)
    results["subspaces"]["F"] = _analyze_subspace(H_noPDS, do_bootstrap=not skip_F_bootstrap)
    
    return results


def compute_part_N_multi_layer(
    layer_to_H: Dict[int, np.ndarray],
    P_bases: Dict[int, Dict[int, np.ndarray]],
    D_basis_by_layer: Dict[int, np.ndarray],
    S_basis_by_layer: Dict[int, np.ndarray],
    total_layers: int,
    n_bootstrap: int = 200,
    seed: int = 0,
    target_depths: Optional[List[float]] = None,
    skip_F_bootstrap: bool = False,
) -> Dict[str, Any]:
    """Run Part N at multiple layers for depth-robustness check.
    
    Running at a single layer leaves the invariance claim vulnerable to
    "you cherry-picked the layer." Running at 5 layers shows whether
    the result holds across depth.
    
    Unlike the single-layer function, this accepts per-layer D and S bases
    (which is correct, since D and S are recomputed at each layer).
    
    Args:
        layer_to_H: Dict mapping layer index → (n_prompts, d_model) hidden states.
        P_bases: Dict[layer_idx, Dict[prompt_idx, (d_model, k_P)]] per-prompt P bases.
            Matches notebook's P_bases structure from build_unified_P_bases().
        D_basis_by_layer: Dict[layer_idx, (d_model, k_D)] D basis per layer.
        S_basis_by_layer: Dict[layer_idx, (d_model, k_S)] S basis per layer.
        total_layers: Total number of layers in the model.
        n_bootstrap: Bootstrap resamples per layer.
        seed: Random seed.
        target_depths: Target depth fractions. Default [0.12, 0.25, 0.50, 0.75, 0.95].
            Selects the closest available layer from layer_to_H.
        skip_F_bootstrap: If True, skip bootstrap for F subspace (point estimate
            only). Saves ~80% of compute time on large models.
    
    Returns:
        Dict with per-layer Part N results and a cross-layer summary.
    """
    if target_depths is None:
        target_depths = [0.12, 0.25, 0.50, 0.75, 0.95]
    
    available_layers = sorted(layer_to_H.keys())
    
    # Filter to layers where we have all three bases
    usable_layers = [L for L in available_layers
                     if L in D_basis_by_layer and L in S_basis_by_layer]
    if not usable_layers:
        return {"error": "No layers with both D and S bases available",
                "available_layers": available_layers,
                "D_layers": sorted(D_basis_by_layer.keys()),
                "S_layers": sorted(S_basis_by_layer.keys())}
    
    selected = []
    for depth in target_depths:
        target_layer = int(depth * total_layers)
        closest = min(usable_layers, key=lambda L: abs(L - target_layer))
        if closest not in [s[0] for s in selected]:
            selected.append((closest, depth))
    
    per_layer = {}
    for li, (layer_idx, target_depth) in enumerate(selected):
        H = layer_to_H[layer_idx]
        if not isinstance(H, np.ndarray):
            H = np.array(H)
        
        depth_pct = int(100 * layer_idx / total_layers)
        label = f"L{layer_idx}_{depth_pct}pct"
        
        # Get per-prompt P bases for this layer
        P_for_layer = P_bases.get(layer_idx) if P_bases else None
        
        import sys
        print(f"      [{li+1}/{len(selected)}] {label}: ", end="", flush=True)
        
        per_layer[label] = compute_part_N_scaling_rank_suite(
            H, P_for_layer,
            D_basis_by_layer[layer_idx],
            S_basis_by_layer[layer_idx],
            layer_label=label, n_bootstrap=n_bootstrap, seed=seed,
            skip_F_bootstrap=skip_F_bootstrap,
        )
        
        # Quick summary for this layer
        d_pr = per_layer[label].get("subspaces", {}).get("D", {}).get("PR", {})
        d_val = d_pr.get("point", 0) if isinstance(d_pr, dict) else 0
        print(f"D_PR={d_val:.1f}, done")
    
    # Cross-layer summary: check if estimates are consistent
    summary = {"n_layers": len(per_layer), "layer_labels": list(per_layer.keys())}
    for subspace in ["D", "S", "F"]:
        pr_values = []
        for label, result in per_layer.items():
            sub = result.get("subspaces", {}).get(subspace, {})
            pr_data = sub.get("PR", {})
            if isinstance(pr_data, dict):
                pr_values.append(pr_data.get("point", float('nan')))
        if pr_values:
            summary[f"{subspace}_PR_across_layers"] = {
                "mean": float(np.nanmean(pr_values)),
                "std": float(np.nanstd(pr_values)),
                "values": pr_values,
            }
    
    return {"per_layer": per_layer, "summary": summary}

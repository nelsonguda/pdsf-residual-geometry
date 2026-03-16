from __future__ import annotations
"""pds_spirality.py

Spirality measures for residual stream trajectories (Part M).

Part of: PDSF — Prediction-Anchored Decomposition into Functional Subspaces

Paper:   "Scale-Invariant Prediction-Proximal Structure in Transformer Residual Streams"
         Nelson Guda, 2026
         arXiv: XXXX.XXXXX
Repo:    https://github.com/nelsonguda/pdsf-residual-geometry

License: MIT (code), CC-BY 4.0 (data)


Purpose:
    Quantifies helical/spiral structure in the residual stream trajectory
    across layers, and measures how PDSF interventions disrupt that structure.
    Piggybacks on Part G's per-prompt forward passes — no additional forward
    passes required.

Paper references:
    Companion paper (rotation hierarchy): Spectral concentration, phase linearity,
    and winding number measures.

Measures:
    1. Cosine similarity ringing — pairwise cosine similarity between hidden
       states at successive layers. Helical structure produces oscillatory
       off-diagonal decay.
    2. Phase linearity — project onto consecutive PC pairs, extract angle per
       layer, measure R² of angle vs layer. Perfect helix → R² ≈ 1.
    3. Spectral concentration — DFT of the cosine similarity row; fraction of
       power in top-k frequencies. Sharp spectral peaks ↔ coherent rotation.

Inputs:
    Hidden state trajectories from Part G forward passes (Dict[layer → ndarray]).

Outputs:
    Spirality profiles, disruption metrics, correlation with Part G/F measures.

Dependencies:
    numpy, scipy (optional, for correlation analyses only)
"""


import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# ============================================================
# CORE SPIRALITY COMPUTATION
# ============================================================

# Companion paper, rotation hierarchy analysis
def compute_cosine_similarity_matrix(
    H_trajectory: Dict[int, np.ndarray],
    layers: List[int],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for hidden states across layers.

    Args:
        H_trajectory: Dict mapping layer_idx → hidden state vector [d_model].
                      These are the full (un-decomposed) residual stream vectors.
        layers: Sorted list of layer indices to include.

    Returns:
        cos_sim_matrix: (n_layers, n_layers) symmetric matrix of cosine similarities.
    """
    n = len(layers)
    vecs = []
    for layer in layers:
        h = H_trajectory[layer]
        norm = np.linalg.norm(h)
        vecs.append(h / norm if norm > 1e-12 else h)
    V = np.stack(vecs, axis=0)  # (n_layers, d_model)
    cos_sim = V @ V.T  # (n_layers, n_layers)
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    return cos_sim


# Spectral concentration of cosine-similarity decay.
# High concentration = coherent multi-rate rotation. companion paper (rotation hierarchy).
def compute_ringing_spectrum(
    cos_sim_matrix: np.ndarray,
    reference_row: int = 0,
) -> Dict[str, Any]:
    """
    Measure spectral concentration of cosine similarity decay — the "ringing"
    signature of helical structure.

    Method: Take a row of the cosine similarity matrix (default: first layer).
    Apply DFT along the layer axis.  Measure what fraction of total power is
    concentrated in the top-k frequency bins.

    Helical structure → sharp spectral peaks → high concentration.
    Destroyed structure → flat spectrum → low concentration.

    Args:
        cos_sim_matrix: (n_layers, n_layers) from compute_cosine_similarity_matrix.
        reference_row: Which row to analyze (default 0 = earliest layer).

    Returns:
        Dict with:
        - spectral_concentration_top1: fraction of power in single strongest frequency
        - spectral_concentration_top3: fraction in top 3 frequencies
        - dominant_frequency: index of the peak frequency (in bins)
        - dominant_period_layers: period of dominant oscillation (in layers)
        - power_spectrum: full power spectrum array (for plotting)
    """
    n = cos_sim_matrix.shape[0]
    if n < 4:
        return _empty_ringing_result()

    row = cos_sim_matrix[reference_row].copy()

    # Remove DC component (mean) to focus on oscillatory structure
    row_centered = row - np.mean(row)

    # Hann window reduces spectral leakage at series boundaries
    window = np.hanning(n)
    row_windowed = row_centered * window

    # DFT — only positive frequencies (real signal)
    fft_vals = np.fft.rfft(row_windowed)
    power = np.abs(fft_vals) ** 2

    # Skip DC bin (index 0)
    power_no_dc = power[1:]
    total_power = np.sum(power_no_dc)

    if total_power < 1e-15:
        return _empty_ringing_result()

    # Sort by power (descending) to find top-k
    sorted_idx = np.argsort(power_no_dc)[::-1]
    top1_power = power_no_dc[sorted_idx[0]]
    top3_power = np.sum(power_no_dc[sorted_idx[:min(3, len(sorted_idx))]])

    # Dominant frequency (in bins, +1 because we skipped DC)
    dominant_bin = sorted_idx[0] + 1
    dominant_period = n / dominant_bin if dominant_bin > 0 else float('inf')

    return {
        "spectral_concentration_top1": float(top1_power / total_power),
        "spectral_concentration_top3": float(top3_power / total_power),
        "dominant_frequency_bin": int(dominant_bin),
        "dominant_period_layers": float(dominant_period),
        "power_spectrum": power.tolist(),
    }


def _empty_ringing_result() -> Dict[str, Any]:
    return {
        "spectral_concentration_top1": None,
        "spectral_concentration_top3": None,
        "dominant_frequency_bin": None,
        "dominant_period_layers": None,
        "power_spectrum": [],
    }


# Phase R² measures constant angular velocity in PCA-projected 2D planes.
# Companion paper, rotation hierarchy analysis.
def compute_phase_linearity(
    H_trajectory: Dict[int, np.ndarray],
    layers: List[int],
    n_pc_pairs: int = 3,
) -> Dict[str, Any]:
    """
    Measure phase linearity in PCA-projected 2D planes.

    Method:
      1. Stack hidden states across layers → (n_layers, d_model)
      2. PCA across the layer axis
      3. For each consecutive PC pair (PC1-PC2, PC3-PC4, ...), extract angle
         at each layer via arctan2
      4. Fit linear regression of angle vs layer index
      5. R² near 1 = coherent rotation at constant angular velocity

    Args:
        H_trajectory: Dict mapping layer_idx → hidden state vector [d_model].
        layers: Sorted list of layer indices.
        n_pc_pairs: Number of PC pairs to analyze (default 3 → 6 PCs).

    Returns:
        Dict with:
        - pc_pairs: List of dicts, one per PC pair, each containing:
            - r_squared: R² of angle vs layer (1.0 = perfect helix)
            - angular_velocity: slope (radians per layer)
            - variance_explained: fraction of variance in this PC pair
            - angles: list of angles (radians) per layer
        - mean_r_squared: average R² across all PC pairs
        - weighted_r_squared: variance-weighted average R²
    """
    n = len(layers)
    if n < 4:
        return _empty_phase_result(n_pc_pairs)

    # Stack into matrix (n_layers, d_model)
    H = np.stack([H_trajectory[l] for l in layers], axis=0).astype(np.float64)

    # Center across layers
    H_centered = H - H.mean(axis=0, keepdims=True)

    # PCA via SVD (thin SVD — we only need top 2*n_pc_pairs components)
    n_components = min(2 * n_pc_pairs, n - 1, H.shape[1])
    try:
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return _empty_phase_result(n_pc_pairs)

    total_var = np.sum(S ** 2)
    if total_var < 1e-15:
        return _empty_phase_result(n_pc_pairs)

    # Project onto PCs: scores = U * S (already from SVD)
    scores = U[:, :n_components] * S[:n_components]

    pc_pair_results = []
    for pair_idx in range(n_pc_pairs):
        c1 = 2 * pair_idx
        c2 = 2 * pair_idx + 1
        if c2 >= n_components:
            break

        x = scores[:, c1]
        y = scores[:, c2]
        angles = np.arctan2(y, x)

        # Unwrap to handle 2π jumps
        angles_unwrapped = np.unwrap(angles)

        # Linear regression: angle = a * layer_index + b
        layer_indices = np.arange(n, dtype=np.float64)
        # Use numpy polyfit (degree 1)
        try:
            coeffs = np.polyfit(layer_indices, angles_unwrapped, 1)
            slope, intercept = coeffs
            predicted = slope * layer_indices + intercept
            ss_res = np.sum((angles_unwrapped - predicted) ** 2)
            ss_tot = np.sum((angles_unwrapped - np.mean(angles_unwrapped)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        except (np.linalg.LinAlgError, ValueError):
            slope = 0.0
            r_squared = 0.0

        var_explained = (S[c1] ** 2 + S[c2] ** 2) / total_var

        pc_pair_results.append({
            "pc_pair": [c1, c2],
            "r_squared": float(np.clip(r_squared, 0.0, 1.0)),
            "angular_velocity": float(slope),
            "variance_explained": float(var_explained),
            "angles_unwrapped": angles_unwrapped.tolist(),
        })

    # Summary statistics
    if pc_pair_results:
        r2_vals = [p["r_squared"] for p in pc_pair_results]
        var_vals = [p["variance_explained"] for p in pc_pair_results]
        mean_r2 = float(np.mean(r2_vals))
        total_var_covered = sum(var_vals)
        weighted_r2 = float(
            sum(r * v for r, v in zip(r2_vals, var_vals)) / total_var_covered
        ) if total_var_covered > 1e-15 else 0.0
    else:
        mean_r2 = 0.0
        weighted_r2 = 0.0

    return {
        "pc_pairs": pc_pair_results,
        "mean_r_squared": mean_r2,
        "weighted_r_squared": weighted_r2,
    }


def _empty_phase_result(n_pc_pairs: int) -> Dict[str, Any]:
    return {
        "pc_pairs": [],
        "mean_r_squared": None,
        "weighted_r_squared": None,
    }


def compute_winding_number(
    H_trajectory: Dict[int, np.ndarray],
    layers: List[int],
    n_pc_pairs: int = 3,
) -> Dict[str, Any]:
    """
    Compute cumulative winding (total angle swept) across layers in PCA planes.

    Coherent spirals → smooth, monotonic accumulation of angle.
    Disrupted rotation → erratic, reduced total winding.

    This is a simpler, more robust measure than phase linearity.
    It answers: "How much total rotation happened?"

    Args:
        H_trajectory: Dict mapping layer_idx → hidden state vector [d_model].
        layers: Sorted list of layer indices.
        n_pc_pairs: Number of PC pairs to analyze.

    Returns:
        Dict with:
        - total_winding: list of total absolute angle swept per PC pair (radians)
        - mean_total_winding: average total winding across pairs
        - monotonicity: fraction of layer transitions where angle increases
                        (1.0 = perfectly monotonic rotation)
    """
    n = len(layers)
    if n < 4:
        return {"total_winding": [], "mean_total_winding": None, "monotonicity": None}

    H = np.stack([H_trajectory[l] for l in layers], axis=0).astype(np.float64)
    H_centered = H - H.mean(axis=0, keepdims=True)

    n_components = min(2 * n_pc_pairs, n - 1, H.shape[1])
    try:
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"total_winding": [], "mean_total_winding": None, "monotonicity": None}

    scores = U[:, :n_components] * S[:n_components]
    windings = []
    monotonicities = []

    for pair_idx in range(n_pc_pairs):
        c1 = 2 * pair_idx
        c2 = 2 * pair_idx + 1
        if c2 >= n_components:
            break

        angles = np.arctan2(scores[:, c2], scores[:, c1])
        angles_unwrapped = np.unwrap(angles)

        # Total winding = total absolute angle swept
        diffs = np.diff(angles_unwrapped)
        total = float(np.sum(np.abs(diffs)))
        windings.append(total)

        # Monotonicity = fraction of steps in the dominant direction
        if len(diffs) > 0:
            n_positive = np.sum(diffs > 0)
            n_negative = np.sum(diffs < 0)
            mono = max(n_positive, n_negative) / len(diffs)
            monotonicities.append(float(mono))

    return {
        "total_winding": windings,
        "mean_total_winding": float(np.mean(windings)) if windings else None,
        "monotonicity": float(np.mean(monotonicities)) if monotonicities else None,
    }


# ============================================================
# COMBINED SPIRALITY PROFILE
# ============================================================

def compute_spirality_profile(
    H_trajectory: Dict[int, np.ndarray],
    layers: List[int],
    n_pc_pairs: int = 3,
    include_full_spectrum: bool = False,
) -> Dict[str, Any]:
    """
    Compute the complete spirality profile for a hidden-state trajectory.

    This is the main entry point — called once per (prompt × condition) by
    Part G's inner loop.

    Args:
        H_trajectory: Dict mapping layer_idx → hidden state [d_model].
                      For baseline: raw H_baseline from forward pass.
                      For intervened: H_scrambled from intervened forward pass.
        layers: Sorted list of layer indices present in H_trajectory.
        n_pc_pairs: Number of PC pairs for phase/winding analysis.
        include_full_spectrum: If True, include raw power spectrum arrays
                              (large — set False for production, True for debug).

    Returns:
        Dict with three sub-dicts:
        - ringing: spectral concentration metrics
        - phase: phase linearity metrics
        - winding: cumulative winding metrics
        - summary: top-level scalars for easy aggregation
    """
    # Filter layers to those actually present
    available_layers = sorted([l for l in layers if l in H_trajectory])
    if len(available_layers) < 4:
        return _empty_spirality_profile()

    # 1. Cosine similarity ringing
    cos_sim = compute_cosine_similarity_matrix(H_trajectory, available_layers)
    # Use middle row as reference (less edge-effect than first or last)
    mid_row = len(available_layers) // 2
    ringing = compute_ringing_spectrum(cos_sim, reference_row=mid_row)

    if not include_full_spectrum:
        ringing.pop("power_spectrum", None)

    # 2. Phase linearity
    phase = compute_phase_linearity(H_trajectory, available_layers, n_pc_pairs)

    # Strip per-layer angles from production output (save space)
    if not include_full_spectrum:
        for pp in phase.get("pc_pairs", []):
            pp.pop("angles_unwrapped", None)

    # 3. Winding number
    winding = compute_winding_number(H_trajectory, available_layers, n_pc_pairs)

    # 4. Summary scalars (for easy cross-condition comparison)
    summary = {
        "spectral_concentration_top1": ringing.get("spectral_concentration_top1"),
        "spectral_concentration_top3": ringing.get("spectral_concentration_top3"),
        "phase_r2_weighted": phase.get("weighted_r_squared"),
        "phase_r2_mean": phase.get("mean_r_squared"),
        "mean_total_winding": winding.get("mean_total_winding"),
        "winding_monotonicity": winding.get("monotonicity"),
        "n_layers_used": len(available_layers),
    }

    return {
        "ringing": ringing,
        "phase": phase,
        "winding": winding,
        "summary": summary,
    }


def _empty_spirality_profile() -> Dict[str, Any]:
    return {
        "ringing": _empty_ringing_result(),
        "phase": _empty_phase_result(3),
        "winding": {"total_winding": [], "mean_total_winding": None, "monotonicity": None},
        "summary": {
            "spectral_concentration_top1": None,
            "spectral_concentration_top3": None,
            "phase_r2_weighted": None,
            "phase_r2_mean": None,
            "mean_total_winding": None,
            "winding_monotonicity": None,
            "n_layers_used": 0,
        },
    }


# ============================================================
# AGGREGATION ACROSS PROMPTS
# ============================================================

def aggregate_spirality_results(
    prompt_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate spirality summary scalars across prompts for a given condition.

    Args:
        prompt_results: List of spirality profile dicts (from compute_spirality_profile).

    Returns:
        Dict with mean/std/median for each summary scalar, plus n_prompts.
    """
    keys = [
        "spectral_concentration_top1",
        "spectral_concentration_top3",
        "phase_r2_weighted",
        "phase_r2_mean",
        "mean_total_winding",
        "winding_monotonicity",
    ]

    agg = {"n_prompts": len(prompt_results)}
    for key in keys:
        vals = []
        for pr in prompt_results:
            v = pr.get("summary", {}).get(key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if vals:
            arr = np.array(vals)
            agg[f"{key}_mean"] = float(np.mean(arr))
            agg[f"{key}_std"] = float(np.std(arr))
            agg[f"{key}_median"] = float(np.median(arr))
            agg[f"{key}_n"] = len(vals)
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
            agg[f"{key}_median"] = None
            agg[f"{key}_n"] = 0

    return agg


def compute_spirality_disruption(
    baseline_profile: Dict[str, Any],
    intervened_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute the disruption to spirality caused by an intervention.

    For each summary scalar, reports:
      - baseline value
      - intervened value
      - absolute change
      - relative change (intervened / baseline - 1)

    Args:
        baseline_profile: spirality profile from baseline forward pass
        intervened_profile: spirality profile from intervened forward pass

    Returns:
        Dict of per-metric disruption measures.
    """
    keys = [
        "spectral_concentration_top1",
        "spectral_concentration_top3",
        "phase_r2_weighted",
        "phase_r2_mean",
        "mean_total_winding",
        "winding_monotonicity",
    ]

    disruption = {}
    for key in keys:
        b = baseline_profile.get("summary", {}).get(key)
        i = intervened_profile.get("summary", {}).get(key)
        if b is not None and i is not None and np.isfinite(b) and np.isfinite(i):
            disruption[f"{key}_baseline"] = float(b)
            disruption[f"{key}_intervened"] = float(i)
            disruption[f"{key}_abs_change"] = float(i - b)
            disruption[f"{key}_rel_change"] = float(i / b - 1.0) if abs(b) > 1e-15 else None
        else:
            disruption[f"{key}_baseline"] = b if b is not None else None
            disruption[f"{key}_intervened"] = i if i is not None else None
            disruption[f"{key}_abs_change"] = None
            disruption[f"{key}_rel_change"] = None

    return disruption


def aggregate_spirality_disruptions(
    disruptions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate disruption measures across prompts for a given (subspace × intervention).

    Args:
        disruptions: List of dicts from compute_spirality_disruption.

    Returns:
        Dict with mean/std for each disruption metric.
    """
    if not disruptions:
        return {"n_prompts": 0}

    # Collect all keys from the first disruption
    all_keys = [k for k in disruptions[0].keys()]

    agg = {"n_prompts": len(disruptions)}
    for key in all_keys:
        vals = [d[key] for d in disruptions if d.get(key) is not None and np.isfinite(d[key])]
        if vals:
            arr = np.array(vals)
            agg[f"{key}_mean"] = float(np.mean(arr))
            agg[f"{key}_std"] = float(np.std(arr))
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None

    return agg


# ============================================================
# BASELINE SPIRALITY FROM layer_to_H CACHE
# ============================================================
# Runs on the FULL prompt set (not just Part G's subset).
# Uses the same layer_to_H cache that Parts A–F use, so zero
# additional forward passes.  Produces a standalone Part M
# baseline file that can be correlated with Part F rotation data.
# ============================================================

def compute_baseline_spirality_from_cache(
    layer_to_H: Dict[int, np.ndarray],
    n_prompts: int,
    layers: Optional[List[int]] = None,
    n_pc_pairs: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute baseline spirality profiles for ALL prompts from the cached
    hidden state tensors (layer_to_H).

    This is independent of Part G — it runs on the full extraction cache
    that Parts A–F also use.  The output is a per-prompt list of spirality
    summaries, suitable for direct correlation with Part F rotation data.

    Args:
        layer_to_H: Dict mapping layer_idx → tensor/array of shape
                    (n_prompts, d_model).  Same format as the pipeline's
                    extraction cache.
        n_prompts: Number of prompts in the dataset.
        layers: Sorted list of layer indices to use.  If None, uses all
                layers present in layer_to_H.
        n_pc_pairs: Number of PC pairs for phase/winding analysis.
        verbose: Print progress info.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with:
        - per_prompt: List of dicts, one per prompt, each containing:
            - prompt_idx: int
            - spirality: summary scalars from compute_spirality_profile
        - aggregate: Mean/std/median of each summary scalar across prompts
        - n_prompts: int
        - n_layers: int
        - layers_used: list of layer indices
    """
    if layers is None:
        layers = sorted(layer_to_H.keys())

    if verbose:
        print(f"\nPart M baseline: computing spirality for {n_prompts} prompts "
              f"across {len(layers)} layers...")

    per_prompt_results = []

    prompt_iter = range(n_prompts)
    if show_progress:
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(prompt_iter, total=n_prompts, desc="Part M baseline spirality")
        except ImportError:
            pass

    for prompt_idx in prompt_iter:
        # Build per-prompt trajectory: layer_idx → h[d_model]
        H_trajectory = {}
        for layer_idx in layers:
            H_layer = layer_to_H[layer_idx]
            # Handle both tensor and ndarray
            if hasattr(H_layer, 'numpy'):
                h = H_layer[prompt_idx].numpy()
            else:
                h = H_layer[prompt_idx]
            H_trajectory[layer_idx] = h.astype(np.float32)

        profile = compute_spirality_profile(
            H_trajectory, layers,
            n_pc_pairs=n_pc_pairs,
            include_full_spectrum=False,
        )

        per_prompt_results.append({
            "prompt_idx": prompt_idx,
            "spirality": profile["summary"],
        })

    # Aggregate across prompts
    profiles_for_agg = [
        {"summary": pr["spirality"]} for pr in per_prompt_results
    ]
    aggregate = aggregate_spirality_results(profiles_for_agg)

    if verbose:
        print(f"  Baseline spirality computed.")
        m = aggregate
        for key in ["spectral_concentration_top1", "phase_r2_weighted",
                     "mean_total_winding", "winding_monotonicity"]:
            val = m.get(f"{key}_mean")
            std = m.get(f"{key}_std")
            if val is not None:
                print(f"    {key}: {val:.3f} ± {std:.3f}")

    return {
        "per_prompt": per_prompt_results,
        "aggregate": aggregate,
        "n_prompts": n_prompts,
        "n_layers": len(layers),
        "layers_used": layers,
    }


# ============================================================
# CROSS-METRIC CORRELATION: SPIRALITY × ROTATION
# ============================================================
# Computes Pearson correlation between spirality measures and
# Part F rotation measures across prompts.  This is the key
# analysis for establishing whether spiral structure and subspace
# rotation are measuring the same underlying phenomenon.
# ============================================================

def correlate_spirality_with_rotation(
    part_m_baseline: Dict[str, Any],
    part_f_per_prompt: Dict[str, Any],
    layers_for_rotation: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute cross-metric correlations between Part M spirality measures
    and Part F subspace rotation measures, across prompts.

    For each spirality scalar (e.g. phase_r2_weighted) and each rotation
    metric (e.g. mean D_rotation), compute Pearson r across prompts.

    This answers: "Do prompts with stronger spiral structure also show
    more/less subspace rotation?"

    Args:
        part_m_baseline: Output of compute_baseline_spirality_from_cache.
        part_f_per_prompt: The per_prompt_results dict from Part F
                          (loaded from {model}-part_f_per_prompt.json).
                          Expected structure: {"prompts": [{"prompt_idx": ...,
                          "trajectory": {"layer": {"D_rotation_from_prev_deg": ...}}}]}
        layers_for_rotation: Which layers to average rotation over.
                            If None, averages over all available layers.

    Returns:
        Dict with:
        - correlations: Dict mapping (spirality_metric, rotation_metric) →
                        {"pearson_r": float, "p_value": float, "n": int}
        - rotation_summary: Per-prompt mean rotation values used
        - spirality_summary: Per-prompt spirality values used
    """
    from scipy import stats as scipy_stats

    # ── Extract per-prompt spirality scalars ──
    spirality_keys = [
        "spectral_concentration_top1",
        "spectral_concentration_top3",
        "phase_r2_weighted",
        "phase_r2_mean",
        "mean_total_winding",
        "winding_monotonicity",
    ]

    spirality_by_prompt = {}
    for entry in part_m_baseline.get("per_prompt", []):
        idx = entry["prompt_idx"]
        spirality_by_prompt[idx] = entry.get("spirality", {})

    # ── Extract per-prompt rotation scalars from Part F ──
    rotation_keys = [
        "D_rotation_from_prev_deg",
        "S_rotation_from_prev_deg",
        "P_rotation_from_prev_deg",
    ]

    rotation_by_prompt = {}
    for entry in part_f_per_prompt.get("prompts", []):
        idx = entry["prompt_idx"]
        traj = entry.get("trajectory", {})

        # Average rotation across specified layers
        per_metric = {}
        for rk in rotation_keys:
            vals = []
            for layer_str, layer_data in traj.items():
                layer_int = int(layer_str)
                if layers_for_rotation and layer_int not in layers_for_rotation:
                    continue
                v = layer_data.get(rk)
                if v is not None and np.isfinite(v):
                    vals.append(v)
            per_metric[rk] = float(np.mean(vals)) if vals else None
        rotation_by_prompt[idx] = per_metric

    # ── Compute correlations ──
    # Find common prompt indices
    common_idx = sorted(set(spirality_by_prompt.keys()) & set(rotation_by_prompt.keys()))

    correlations = {}
    for sk in spirality_keys:
        for rk in rotation_keys:
            s_vals = []
            r_vals = []
            for idx in common_idx:
                sv = spirality_by_prompt[idx].get(sk)
                rv = rotation_by_prompt[idx].get(rk)
                if (sv is not None and np.isfinite(sv) and
                        rv is not None and np.isfinite(rv)):
                    s_vals.append(sv)
                    r_vals.append(rv)

            pair_key = f"{sk}_vs_{rk}"
            if len(s_vals) >= 5:
                r_val, p_val = scipy_stats.pearsonr(s_vals, r_vals)
                correlations[pair_key] = {
                    "pearson_r": round(float(r_val), 4),
                    "p_value": float(p_val),
                    "n": len(s_vals),
                }
            else:
                correlations[pair_key] = {
                    "pearson_r": None,
                    "p_value": None,
                    "n": len(s_vals),
                }

    return {
        "correlations": correlations,
        "n_common_prompts": len(common_idx),
        "spirality_keys": spirality_keys,
        "rotation_keys": rotation_keys,
        "layers_for_rotation": layers_for_rotation,
    }


def correlate_spirality_disruption_with_part_g(
    part_g_output: Dict[str, Any],
    scramble_layer_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute the KEY correlation: does spirality disruption track
    Part G component-angle disruption across prompts?

    For each (subspace × intervention) condition, compute Pearson r between
    spirality disruption scalars and component-angle disruption (D_angle_deg,
    F_angle_deg) across prompts.

    This is the causal bridge: if F-mix degrades spirality proportionally
    to how much it disrupts component geometry, the two frameworks are
    measuring the same thing.

    Args:
        part_g_output: Full Part G output dict (with Part M spirality
                       fields already embedded).
        scramble_layer_key: Which scramble layer to analyze (as string).
                           If None, uses first available.

    Returns:
        Dict mapping (subspace, intervention) → correlation results.
    """
    from scipy import stats as scipy_stats

    by_sl = part_g_output.get("by_scramble_layer", {})
    if not by_sl:
        return {"error": "no scramble layers found"}

    if scramble_layer_key is None:
        # Keys may be int or str depending on JSON round-tripping
        first_key = list(by_sl.keys())[0]
        scramble_layer_key = first_key
    
    # Handle int/str key mismatch (Part G stores int keys; JSON round-trip may stringify)
    if scramble_layer_key not in by_sl:
        # Try the other type
        alt_key = int(scramble_layer_key) if isinstance(scramble_layer_key, str) else str(scramble_layer_key)
        if alt_key in by_sl:
            scramble_layer_key = alt_key
        else:
            return {"error": f"scramble layer key {scramble_layer_key} not found"}

    scr_exp = by_sl[scramble_layer_key].get("scramble_experiment", {})
    prompts = scr_exp.get("by_prompt", [])

    # Target: the last measured layer (L-2 equivalent)
    layers_after = scr_exp.get("metadata", {}).get("layers_after_scramble")
    if not layers_after:
        # Infer from first prompt's first scramble result
        for pr in prompts:
            for sub_data in pr.get("by_subspace", {}).values():
                for it_data in sub_data.get("by_intervention", {}).values():
                    for scr in it_data.get("scrambles", []):
                        layers_after = sorted(int(k) for k in scr.get("by_layer", {}).keys())
                        break
                    if layers_after:
                        break
                if layers_after:
                    break
            if layers_after:
                break

    if not layers_after or len(layers_after) < 2:
        return {"error": "cannot determine post-intervention layers"}

    # Use pre-final layer (L-2) to avoid LM-head amplification
    target_layer_int = layers_after[-2]

    spirality_disruption_keys = [
        "spectral_concentration_top1_abs_change",
        "phase_r2_weighted_abs_change",
        "mean_total_winding_abs_change",
    ]
    angle_keys = ["D", "F", "S"]

    results = {}
    for pr in prompts:
        for sub, sub_data in pr.get("by_subspace", {}).items():
            for it, it_data in sub_data.get("by_intervention", {}).items():
                condition_key = f"{sub}_{it}"
                if condition_key not in results:
                    results[condition_key] = {sk: [] for sk in spirality_disruption_keys}
                    for ak in angle_keys:
                        results[condition_key][f"{ak}_angle_deg"] = []

                for scr in it_data.get("scrambles", []):
                    sd = scr.get("spirality_disruption", {})
                    # by_layer keys may be int or str depending on serialization
                    by_layer = scr.get("by_layer", {})
                    layer_div = by_layer.get(target_layer_int) or by_layer.get(str(target_layer_int), {})

                    # Spirality disruption values
                    for sk in spirality_disruption_keys:
                        v = sd.get(sk)
                        results[condition_key][sk].append(v)

                    # Component angle values
                    for ak in angle_keys:
                        v = layer_div.get(ak, {}).get("angle_deg") if isinstance(layer_div.get(ak), dict) else None
                        results[condition_key][f"{ak}_angle_deg"].append(v)

    # Now compute correlations per condition
    correlations = {}
    for condition_key, data in results.items():
        condition_corrs = {}
        for sk in spirality_disruption_keys:
            for ak in angle_keys:
                ak_full = f"{ak}_angle_deg"
                s_vals = []
                a_vals = []
                for sv, av in zip(data[sk], data[ak_full]):
                    if (sv is not None and np.isfinite(sv) and
                            av is not None and np.isfinite(av)):
                        s_vals.append(sv)
                        a_vals.append(av)

                pair_key = f"{sk}_vs_{ak_full}"
                if len(s_vals) >= 5:
                    r_val, p_val = scipy_stats.pearsonr(s_vals, a_vals)
                    condition_corrs[pair_key] = {
                        "pearson_r": round(float(r_val), 4),
                        "p_value": float(p_val),
                        "n": len(s_vals),
                    }
                else:
                    condition_corrs[pair_key] = {
                        "pearson_r": None, "p_value": None, "n": len(s_vals),
                    }
        correlations[condition_key] = condition_corrs

    return {
        "correlations_by_condition": correlations,
        "target_layer": target_layer_int,
        "scramble_layer": scramble_layer_key,
    }

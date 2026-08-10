"""pds_prompt_analysis.py

Token distribution analysis and cross-prompt-set comparison (optional diagnostic).

Part of: PDSF — Prediction-Anchored Decomposition into Functional Subspaces

Paper:   "Geometric and Behavioral Stratification in Transformer Residual Streams"
         Nelson Guda, 2026
Repo:    https://github.com/nelsonguda/pdsf-residual-geometry

License: MIT (code), CC-BY 4.0 (data)


Purpose:
    Compares token prediction behavior between prompt sets (SpecA vs SpecB).
    SpecA prompts have constrained single-token answers (peaked distributions);
    SpecB prompts are open-ended (flatter distributions). This module quantifies
    the difference in prediction sharpness.

    This module is OPTIONAL — it is not required for the main geometry or
    intervention experiments.

Paper references:
    Optional diagnostic. Quantifies the SpecA/SpecB prediction-sharpness difference
    that motivates the prompt-set design (Appendix A.2); no result from this module
    is reported in the paper.

Inputs:
    Model, tokenizer, prompt lists with IDs.

Outputs:
    Per-prompt token distribution stats (entropy, PR, top-k probs),
    aggregate comparison between prompt sets.

Dependencies:
    torch, numpy, pds_continuation (for chat template and device utilities)
"""

from __future__ import annotations


import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
__version__ = "1.0"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenDistribution:
    """Token probability distribution at the final position."""
    prompt: str
    prompt_id: str
    top_k_tokens: List[str]           # Top-k token strings
    top_k_probs: List[float]          # Top-k probabilities
    top_k_ids: List[int]              # Top-k token IDs
    entropy: float                     # Shannon entropy (nats)
    entropy_bits: float                # Shannon entropy (bits)
    participation_ratio: float         # Effective number of tokens
    top1_prob: float                   # Probability of most likely token
    top5_cumulative: float             # Cumulative prob of top 5
    top10_cumulative: float            # Cumulative prob of top 10
    
@dataclass
class PromptSetDistributionStats:
    """Aggregate statistics for a set of prompts."""
    prompt_set_name: str
    n_prompts: int
    mean_entropy: float
    std_entropy: float
    mean_pr: float
    std_pr: float
    mean_top1_prob: float
    std_top1_prob: float
    mean_top5_cumulative: float
    mean_top10_cumulative: float
    distributions: List[TokenDistribution] = field(default_factory=list)


# =============================================================================
# TOKEN DISTRIBUTION ANALYSIS
# =============================================================================

def compute_token_distribution(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    prompt_id: str = "",
    top_k: int = 20,
    model_id: str = "",
    apply_template: bool = True,
) -> TokenDistribution:
    """
    Compute the token probability distribution for the next token after a prompt.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt string
        prompt_id: Identifier for the prompt
        top_k: Number of top tokens to return
        model_id: Model identifier for chat template
        apply_template: Whether to apply chat template
    
    Returns:
        TokenDistribution with entropy, PR, and top-k tokens
    """
    from pds_continuation import get_model_input_device, apply_chat_template
    
    device = get_model_input_device(model)
    
    # Format prompt
    if apply_template and model_id:
        formatted = apply_chat_template(model_id, prompt)
    else:
        formatted = prompt
    
    # Get model output
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (vocab_size,)
    
    # Compute entropy (Shannon entropy in nats)
    # H = -Σ p(x) log p(x), only for p > 0
    nonzero_probs = probs[probs > 1e-10]
    entropy_nats = -np.sum(nonzero_probs * np.log(nonzero_probs))
    entropy_bits = entropy_nats / np.log(2)
    
    # Compute participation ratio: PR = 1 / Σ p_i^2
    # This is the effective number of tokens
    pr = 1.0 / np.sum(probs ** 2)
    
    # Get top-k tokens
    top_k_indices = np.argsort(probs)[::-1][:top_k]
    top_k_probs = probs[top_k_indices].tolist()
    top_k_ids = top_k_indices.tolist()
    top_k_tokens = [tokenizer.decode([idx]).strip() for idx in top_k_ids]
    
    # Cumulative probabilities
    sorted_probs = np.sort(probs)[::-1]
    top1_prob = float(sorted_probs[0])
    top5_cumulative = float(np.sum(sorted_probs[:5]))
    top10_cumulative = float(np.sum(sorted_probs[:10]))
    
    return TokenDistribution(
        prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
        prompt_id=prompt_id,
        top_k_tokens=top_k_tokens,
        top_k_probs=top_k_probs,
        top_k_ids=top_k_ids,
        entropy=float(entropy_nats),
        entropy_bits=float(entropy_bits),
        participation_ratio=float(pr),
        top1_prob=top1_prob,
        top5_cumulative=top5_cumulative,
        top10_cumulative=top10_cumulative,
    )


def compute_prompt_set_distributions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[Tuple[str, str]],  # List of (prompt_id, prompt_text)
    prompt_set_name: str,
    top_k: int = 20,
    model_id: str = "",
    verbose: bool = True,
) -> PromptSetDistributionStats:
    """
    Compute token distributions for a set of prompts.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompts: List of (prompt_id, prompt_text) tuples
        prompt_set_name: Name for this prompt set (e.g., "SpecA", "SpecB")
        top_k: Number of top tokens per prompt
        model_id: Model identifier
        verbose: Show progress
    
    Returns:
        PromptSetDistributionStats with aggregate statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPUTING TOKEN DISTRIBUTIONS: {prompt_set_name}")
        print(f"{'='*60}")
        print(f"  N prompts: {len(prompts)}")
    
    distributions = []
    
    try:
        from tqdm import tqdm
        iterator = tqdm(prompts, desc=f"{prompt_set_name} distributions") if verbose else prompts
    except ImportError:
        iterator = prompts
    
    for prompt_id, prompt_text in iterator:
        dist = compute_token_distribution(
            model, tokenizer, prompt_text, prompt_id, top_k, model_id
        )
        distributions.append(dist)
    
    # Compute aggregate statistics
    entropies = [d.entropy for d in distributions]
    prs = [d.participation_ratio for d in distributions]
    top1s = [d.top1_prob for d in distributions]
    top5s = [d.top5_cumulative for d in distributions]
    top10s = [d.top10_cumulative for d in distributions]
    
    stats = PromptSetDistributionStats(
        prompt_set_name=prompt_set_name,
        n_prompts=len(distributions),
        mean_entropy=float(np.mean(entropies)),
        std_entropy=float(np.std(entropies)),
        mean_pr=float(np.mean(prs)),
        std_pr=float(np.std(prs)),
        mean_top1_prob=float(np.mean(top1s)),
        std_top1_prob=float(np.std(top1s)),
        mean_top5_cumulative=float(np.mean(top5s)),
        mean_top10_cumulative=float(np.mean(top10s)),
        distributions=distributions,
    )
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Entropy:  {stats.mean_entropy:.2f} ± {stats.std_entropy:.2f} nats")
        print(f"    PR:       {stats.mean_pr:.1f} ± {stats.std_pr:.1f} effective tokens")
        print(f"    Top-1:    {stats.mean_top1_prob:.1%} ± {stats.std_top1_prob:.1%}")
        print(f"    Top-5:    {stats.mean_top5_cumulative:.1%}")
        print(f"    Top-10:   {stats.mean_top10_cumulative:.1%}")
    
    return stats


def compare_prompt_set_distributions(
    stats_a: PromptSetDistributionStats,
    stats_b: PromptSetDistributionStats,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare token distribution statistics between two prompt sets.
    
    Returns dict with comparison metrics and statistical tests.
    """
    from scipy import stats as scipy_stats
    
    # Extract arrays
    entropy_a = [d.entropy for d in stats_a.distributions]
    entropy_b = [d.entropy for d in stats_b.distributions]
    pr_a = [d.participation_ratio for d in stats_a.distributions]
    pr_b = [d.participation_ratio for d in stats_b.distributions]
    top1_a = [d.top1_prob for d in stats_a.distributions]
    top1_b = [d.top1_prob for d in stats_b.distributions]
    
    # Statistical tests (Mann-Whitney U for non-parametric comparison)
    entropy_stat, entropy_p = scipy_stats.mannwhitneyu(entropy_a, entropy_b, alternative='two-sided')
    pr_stat, pr_p = scipy_stats.mannwhitneyu(pr_a, pr_b, alternative='two-sided')
    top1_stat, top1_p = scipy_stats.mannwhitneyu(top1_a, top1_b, alternative='two-sided')
    
    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.std(x)**2 + (ny-1)*np.std(y)**2) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
    
    comparison = {
        "set_a": stats_a.prompt_set_name,
        "set_b": stats_b.prompt_set_name,
        "n_a": stats_a.n_prompts,
        "n_b": stats_b.n_prompts,
        "entropy": {
            "mean_a": stats_a.mean_entropy,
            "mean_b": stats_b.mean_entropy,
            "diff": stats_b.mean_entropy - stats_a.mean_entropy,
            "cohens_d": cohens_d(entropy_a, entropy_b),
            "p_value": entropy_p,
        },
        "participation_ratio": {
            "mean_a": stats_a.mean_pr,
            "mean_b": stats_b.mean_pr,
            "diff": stats_b.mean_pr - stats_a.mean_pr,
            "cohens_d": cohens_d(pr_a, pr_b),
            "p_value": pr_p,
        },
        "top1_prob": {
            "mean_a": stats_a.mean_top1_prob,
            "mean_b": stats_b.mean_top1_prob,
            "diff": stats_b.mean_top1_prob - stats_a.mean_top1_prob,
            "cohens_d": cohens_d(top1_a, top1_b),
            "p_value": top1_p,
        },
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPARISON: {stats_a.prompt_set_name} vs {stats_b.prompt_set_name}")
        print(f"{'='*60}")
        print(f"\n{'Metric':<20} {'A':>12} {'B':>12} {'Diff':>12} {'Cohen d':>10} {'p-value':>10}")
        print("-" * 76)
        for metric in ["entropy", "participation_ratio", "top1_prob"]:
            m = comparison[metric]
            print(f"{metric:<20} {m['mean_a']:>12.3f} {m['mean_b']:>12.3f} {m['diff']:>+12.3f} {m['cohens_d']:>10.2f} {m['p_value']:>10.4f}")
    
    return comparison


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_token_distribution_bars(
    dist: TokenDistribution,
    ax=None,
    top_k: int = 10,
    title: str = None,
):
    """Plot bar chart of top-k token probabilities for a single prompt."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    tokens = dist.top_k_tokens[:top_k]
    probs = dist.top_k_probs[:top_k]
    
    # Escape special characters for display
    tokens_display = [t.replace('$', r'\$').replace('_', r'\_') if t else '[empty]' for t in tokens]
    
    bars = ax.barh(range(len(tokens)), probs, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens_display, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    
    # Add probability labels
    for i, (bar, p) in enumerate(zip(bars, probs)):
        ax.text(p + 0.01, i, f'{p:.3f}', va='center', fontsize=8)
    
    if title:
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title(f'H={dist.entropy:.2f} nats, PR={dist.participation_ratio:.1f}', fontsize=10)
    
    return ax


def plot_distribution_comparison_figure(
    stats_a: PromptSetDistributionStats,
    stats_b: PromptSetDistributionStats,
    n_examples: int = 3,
    save_path: Optional[Path] = None,
):
    """
    Create a comprehensive comparison figure.
    
    Layout:
    - Row 1: Example distributions from set A (n_examples)
    - Row 2: Example distributions from set B (n_examples)
    - Row 3: Histograms comparing entropy, PR, top-1 prob
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(15, 12))
    
    # Row 1: Examples from set A
    for i in range(min(n_examples, len(stats_a.distributions))):
        ax = fig.add_subplot(3, n_examples + 1, i + 1)
        plot_token_distribution_bars(stats_a.distributions[i], ax=ax, top_k=8)
        if i == 0:
            ax.set_ylabel(f'{stats_a.prompt_set_name}\n\nToken', fontsize=10)
    
    # Row 2: Examples from set B  
    for i in range(min(n_examples, len(stats_b.distributions))):
        ax = fig.add_subplot(3, n_examples + 1, n_examples + 2 + i)
        plot_token_distribution_bars(stats_b.distributions[i], ax=ax, top_k=8)
        if i == 0:
            ax.set_ylabel(f'{stats_b.prompt_set_name}\n\nToken', fontsize=10)
    
    # Row 3: Histograms
    entropy_a = [d.entropy for d in stats_a.distributions]
    entropy_b = [d.entropy for d in stats_b.distributions]
    pr_a = [d.participation_ratio for d in stats_a.distributions]
    pr_b = [d.participation_ratio for d in stats_b.distributions]
    top1_a = [d.top1_prob for d in stats_a.distributions]
    top1_b = [d.top1_prob for d in stats_b.distributions]
    
    # Entropy histogram
    ax1 = fig.add_subplot(3, 3, 7)
    ax1.hist(entropy_a, bins=20, alpha=0.7, label=stats_a.prompt_set_name, color='blue')
    ax1.hist(entropy_b, bins=20, alpha=0.7, label=stats_b.prompt_set_name, color='orange')
    ax1.set_xlabel('Entropy (nats)')
    ax1.set_ylabel('Count')
    ax1.set_title('Entropy Distribution')
    ax1.legend()
    
    # PR histogram
    ax2 = fig.add_subplot(3, 3, 8)
    ax2.hist(pr_a, bins=20, alpha=0.7, label=stats_a.prompt_set_name, color='blue')
    ax2.hist(pr_b, bins=20, alpha=0.7, label=stats_b.prompt_set_name, color='orange')
    ax2.set_xlabel('Participation Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('PR Distribution (Effective # Tokens)')
    ax2.legend()
    
    # Top-1 histogram
    ax3 = fig.add_subplot(3, 3, 9)
    ax3.hist(top1_a, bins=20, alpha=0.7, label=stats_a.prompt_set_name, color='blue')
    ax3.hist(top1_b, bins=20, alpha=0.7, label=stats_b.prompt_set_name, color='orange')
    ax3.set_xlabel('Top-1 Probability')
    ax3.set_ylabel('Count')
    ax3.set_title('Top-1 Probability Distribution')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_cumulative_probability_curves(
    stats_a: PromptSetDistributionStats,
    stats_b: PromptSetDistributionStats,
    save_path: Optional[Path] = None,
):
    """
    Plot cumulative probability curves showing how many tokens capture X% of probability.
    
    Steep curve = sharp distribution (SpecA expected)
    Gradual curve = broad distribution (SpecB expected)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute average cumulative curves
    max_k = min(20, min(len(stats_a.distributions[0].top_k_probs), 
                        len(stats_b.distributions[0].top_k_probs)))
    
    cum_a = np.zeros(max_k)
    cum_b = np.zeros(max_k)
    
    for dist in stats_a.distributions:
        cum_a += np.cumsum(dist.top_k_probs[:max_k])
    cum_a /= len(stats_a.distributions)
    
    for dist in stats_b.distributions:
        cum_b += np.cumsum(dist.top_k_probs[:max_k])
    cum_b /= len(stats_b.distributions)
    
    x = np.arange(1, max_k + 1)
    ax.plot(x, cum_a, 'b-o', label=f'{stats_a.prompt_set_name} (mean)', linewidth=2, markersize=4)
    ax.plot(x, cum_b, 'r-s', label=f'{stats_b.prompt_set_name} (mean)', linewidth=2, markersize=4)
    
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')
    
    ax.set_xlabel('Number of Top Tokens')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Token Probability Curves')
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


# =============================================================================
# SPECA-STYLE ANALYSIS FOR SPECB PROMPTS
# =============================================================================

def run_specA_analysis_on_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[Tuple[str, str, str]],  # List of (group_id, variant_id, prompt_text)
    model_id: str,
    output_dir: Path,
    model_key: str,
    layer_fraction: float = 0.70,
    config: Optional[Dict] = None,
    precomputed_bases: Optional[Any] = None,  # Accept SpecB2Bases object
    precomputed_H: Optional[np.ndarray] = None,  # Accept pre-extracted hidden states
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run SpecA-style trajectory analysis on any prompt set.
    
    This extracts hidden states and computes P/D/S decomposition metrics
    without running generation experiments.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompts: List of (group_id, variant_id, prompt_text) tuples
        model_id: Model identifier
        output_dir: Directory for output files
        model_key: Model key for filenames
        layer_fraction: Which layer to analyze (0.0-1.0)
        config: Optional config dict (p_rank, d_rank, s_rank)
        precomputed_bases: Optional SpecB2Bases object to reuse (skips basis computation)
        precomputed_H: Optional pre-extracted hidden states array (skips extraction)
        verbose: Show progress
    
    Returns:
        Dict with analysis results including:
        - P/D/S energy fractions
        - Participation ratios
        - Within/between group angles
        - Per-prompt trajectories (if multiple layers)
    """
    from pds_continuation import (
        get_model_layers, get_model_input_device, apply_chat_template,
        participation_ratio, pca_basis, project_onto_basis, project_out_basis,
        extract_hidden_at_layer,  # Use shared extraction function
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = config or {}
    n_layers = len(get_model_layers(model))
    target_layer = int(n_layers * layer_fraction)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SPECA-STYLE ANALYSIS ON CUSTOM PROMPTS")
        print(f"{'='*70}")
        print(f"  N prompts: {len(prompts)}")
        print(f"  Target layer: {target_layer}/{n_layers} ({layer_fraction*100:.0f}% depth)")
        if precomputed_bases is not None:
            print(f"  Using precomputed bases ✓")
        if precomputed_H is not None:
            print(f"  Using precomputed hidden states ✓")
    
    # Group prompts by family
    family_prompts = {}
    prompt_list = []
    family_ids = []
    
    for group_id, variant_id, prompt_text in prompts:
        family_prompts.setdefault(group_id, []).append((variant_id, prompt_text))
        prompt_list.append(prompt_text)
        family_ids.append(group_id)
    
    if verbose:
        print(f"  N families: {len(family_prompts)}")
    
    # === HIDDEN STATE EXTRACTION ===
    # Use precomputed H if provided, otherwise extract
    if precomputed_H is not None:
        H = precomputed_H
        if verbose:
            print(f"  Using precomputed H: shape {H.shape}")
    else:
        if verbose:
            print(f"\n  Extracting hidden states at layer {target_layer}...")
        H = extract_hidden_at_layer(
            model, tokenizer, prompt_list, target_layer,
            model_id=model_id, show_progress=verbose
        )
        if verbose:
            print(f"  H shape: {H.shape}")
    
    # === P/D/S BASIS COMPUTATION ===
    # Use precomputed bases if provided, otherwise compute using canonical functions
    from pds_continuation import compute_P_bases_from_predictions, compute_D_basis_global, compute_S_basis_global
    
    d_rank = config.get("d_rank", "adaptive")
    s_rank = config.get("s_rank", "adaptive")
    
    if precomputed_bases is not None:
        if verbose:
            print(f"\n  Using precomputed P/D/S bases")
        P_bases = precomputed_bases.P_bases
        D_basis = precomputed_bases.D_basis
        S_basis = precomputed_bases.S_basis
        # Try to get info from metadata, or create empty dicts
        P_info = precomputed_bases.metadata.get("P_info", {})
        D_info = precomputed_bases.metadata.get("D_info", {})
        S_info = precomputed_bases.metadata.get("S_info", {})
    else:
        if verbose:
            print(f"\n  Computing unified P/D/S bases...")
        # Need model and predicted tokens for P basis
        pred_token_ids = config.get("pred_token_ids")
        unembed_np = config.get("unembed_np")
        if pred_token_ids is None or unembed_np is None:
            raise ValueError("pred_token_ids and unembed_np must be provided in config for P basis computation")
        P_bases, P_info = compute_P_bases_from_predictions(pred_token_ids, unembed_np)
        
        if verbose:
            print(f"  Computing D basis...")
        D_basis, D_info = compute_D_basis_global(H, P_bases, k_D=d_rank)
        
        if verbose:
            print(f"  Computing S basis...")
        S_basis, S_info = compute_S_basis_global(H, P_bases, D_basis, k_S=s_rank)
    
    # Compute per-prompt decomposition
    if verbose:
        print(f"\n  Computing per-prompt decomposition...")
    
    per_prompt_results = []
    
    for i, (h, fid) in enumerate(zip(H, family_ids)):
        Pb = P_bases.get(i, np.zeros((H.shape[1], 0), dtype=np.float32))
        
        # Project onto subspaces
        h_P = project_onto_basis(h, Pb)
        h_perp = h - h_P
        h_D = project_onto_basis(h_perp, D_basis)
        h_S = project_onto_basis(h_perp - h_D, S_basis)
        h_F = h - h_P - h_D - h_S
        
        # Compute energies
        e_total = float(np.dot(h, h))
        e_P = float(np.dot(h_P, h_P))
        e_D = float(np.dot(h_D, h_D))
        e_S = float(np.dot(h_S, h_S))
        e_F = float(np.dot(h_F, h_F))
        
        per_prompt_results.append({
            "prompt_idx": i,
            "family_id": fid,
            "energy_total": e_total,
            "energy_P": e_P,
            "energy_D": e_D,
            "energy_S": e_S,
            "energy_F": e_F,
            "frac_P": e_P / e_total if e_total > 0 else 0,
            "frac_D": e_D / e_total if e_total > 0 else 0,
            "frac_S": e_S / e_total if e_total > 0 else 0,
            "frac_F": e_F / e_total if e_total > 0 else 0,
        })
    
    # Compute participation ratios
    pr_P = participation_ratio(H, center=True)
    
    # Project out P and compute PR of remainder
    H_perp = np.zeros_like(H)
    for i, fid in enumerate(family_ids):
        Pb = P_bases.get(fid, np.zeros((H.shape[1], 0), dtype=np.float32))
        H_perp[i] = project_out_basis(H[i:i+1, :], Pb)[0]
    pr_P_perp = participation_ratio(H_perp, center=True)
    
    # Compute within-family and between-family D-subspace angles
    if verbose:
        print(f"\n  Computing family-wise D angles...")
    
    family_D_bases = {}
    for fid, idx_list in family_prompts.items():
        indices = [i for i, f in enumerate(family_ids) if f == fid]
        if len(indices) >= 2:
            Hf_perp = H_perp[indices]
            basis, _ = pca_basis(Hf_perp, k=min(8, len(indices)-1), center=True)
            family_D_bases[fid] = basis
    
    # Compute angles between family D bases
    within_angles = []
    between_angles = []
    
    families = list(family_D_bases.keys())
    for i, fid_i in enumerate(families):
        for j, fid_j in enumerate(families):
            if i >= j:
                continue
            
            # Compute principal angles between bases
            B_i = family_D_bases[fid_i]
            B_j = family_D_bases[fid_j]
            
            # Principal angles via SVD of B_i.T @ B_j
            if B_i.shape[1] > 0 and B_j.shape[1] > 0:
                M = B_i.T @ B_j
                s = np.linalg.svd(M, compute_uv=False)
                s = np.clip(s, -1, 1)
                angles = np.arccos(s) * 180 / np.pi
                mean_angle = float(np.mean(angles))
                between_angles.append(mean_angle)
    
    # Aggregate statistics
    results = {
        "metadata": {
            "model_key": model_key,
            "n_prompts": len(prompts),
            "n_families": len(family_prompts),
            "target_layer": target_layer,
            "n_layers": n_layers,
            "layer_fraction": layer_fraction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "participation_ratios": {
            "PR_P": float(pr_P),
            "PR_P_perp": float(pr_P_perp),
            "PR_ratio": float(pr_P / pr_P_perp) if pr_P_perp > 0 else None,
        },
        "subspace_info": {
            "P": P_info,
            "D": D_info,
            "S": S_info,
        },
        "energy_fractions": {
            "mean_P": float(np.mean([r["frac_P"] for r in per_prompt_results])),
            "mean_D": float(np.mean([r["frac_D"] for r in per_prompt_results])),
            "mean_S": float(np.mean([r["frac_S"] for r in per_prompt_results])),
            "mean_F": float(np.mean([r["frac_F"] for r in per_prompt_results])),
        },
        "family_angles": {
            "between_family_mean": float(np.mean(between_angles)) if between_angles else None,
            "between_family_std": float(np.std(between_angles)) if between_angles else None,
            "n_pairs": len(between_angles),
        },
        "per_prompt": per_prompt_results,
    }
    
    # Save results
    output_file = output_dir / f"{model_key}-SpecA_style_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n  Results:")
        print(f"    PR(P):     {results['participation_ratios']['PR_P']:.2f}")
        print(f"    PR(P⊥):    {results['participation_ratios']['PR_P_perp']:.2f}")
        print(f"    Mean P energy:  {results['energy_fractions']['mean_P']:.1%}")
        print(f"    Mean D energy:  {results['energy_fractions']['mean_D']:.1%}")
        print(f"    Mean S energy:  {results['energy_fractions']['mean_S']:.1%}")
        if results['family_angles']['between_family_mean']:
            print(f"    Between-family D angle: {results['family_angles']['between_family_mean']:.1f}°")
        print(f"\n  Saved to: {output_file}")
    
    return results


# =============================================================================
# FILE I/O
# =============================================================================

def save_distribution_stats(stats: PromptSetDistributionStats, output_dir: Path, model_key: str) -> str:
    """Save distribution statistics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = {
        "metadata": {
            "prompt_set_name": stats.prompt_set_name,
            "n_prompts": stats.n_prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": {
            "mean_entropy": stats.mean_entropy,
            "std_entropy": stats.std_entropy,
            "mean_pr": stats.mean_pr,
            "std_pr": stats.std_pr,
            "mean_top1_prob": stats.mean_top1_prob,
            "std_top1_prob": stats.std_top1_prob,
            "mean_top5_cumulative": stats.mean_top5_cumulative,
            "mean_top10_cumulative": stats.mean_top10_cumulative,
        },
        "distributions": [
            {
                "prompt": d.prompt,
                "prompt_id": d.prompt_id,
                "entropy": d.entropy,
                "entropy_bits": d.entropy_bits,
                "participation_ratio": d.participation_ratio,
                "top1_prob": d.top1_prob,
                "top5_cumulative": d.top5_cumulative,
                "top10_cumulative": d.top10_cumulative,
                "top_k_tokens": d.top_k_tokens,
                "top_k_probs": d.top_k_probs,
            }
            for d in stats.distributions
        ],
    }
    
    filepath = output_dir / f"{model_key}-token_distributions-{stats.prompt_set_name}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return str(filepath)


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

def run_prompt_comparison_analysis(
    model: torch.nn.Module,
    tokenizer: Any,
    specA_prompts: List[Tuple[str, str]],  # (prompt_id, prompt_text)
    specB_prompts: List[Tuple[str, str]],  # (prompt_id, prompt_text)
    output_dir: Path,
    model_key: str,
    model_id: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete prompt comparison analysis.
    
    This computes:
    1. Token distributions for both prompt sets
    2. Statistical comparison
    3. Visualization figures
    
    Returns:
        Dict with all results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print("PROMPT COMPARISON ANALYSIS")
        print("="*70)
    
    # Compute distributions
    stats_A = compute_prompt_set_distributions(
        model, tokenizer, specA_prompts, "SpecA", 
        model_id=model_id, verbose=verbose
    )
    
    stats_B = compute_prompt_set_distributions(
        model, tokenizer, specB_prompts, "SpecB",
        model_id=model_id, verbose=verbose
    )
    
    # Compare
    comparison = compare_prompt_set_distributions(stats_A, stats_B, verbose=verbose)
    
    # Save results
    file_A = save_distribution_stats(stats_A, output_dir, model_key)
    file_B = save_distribution_stats(stats_B, output_dir, model_key)
    
    # Save comparison
    comparison_file = output_dir / f"{model_key}-distribution_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate figures
    try:
        fig1 = plot_distribution_comparison_figure(
            stats_A, stats_B, n_examples=3,
            save_path=output_dir / f"{model_key}-distribution_comparison.png"
        )
        
        fig2 = plot_cumulative_probability_curves(
            stats_A, stats_B,
            save_path=output_dir / f"{model_key}-cumulative_curves.png"
        )
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not generate figures: {e}")
        fig1, fig2 = None, None
    
    results = {
        "specA_stats": stats_A,
        "specB_stats": stats_B,
        "comparison": comparison,
        "file_paths": {
            "specA_distributions": file_A,
            "specB_distributions": file_B,
            "comparison": str(comparison_file),
        }
    }
    
    if verbose:
        print("\n" + "="*70)
        print("="*70)
    
    return results


# =============================================================================
# GEOMETRY COMPARISON
# =============================================================================

def compare_geometry_analyses(
    results_dir: Path,
    output_file: Path,
    model_key: str,
    prompt_sets: Dict[str, str] = None,
    model_id: str = "",
    token_dist_files: Dict[str, Optional[Path]] = None,
    run_parts: Dict[str, bool] = None,
    verbose: bool = True,
    # Backward compat: old 2-set signature
    specA_results_dir: Path = None,
    specB_results_dir: Path = None,
    specA_token_dist_file: Optional[Path] = None,
    specB_token_dist_file: Optional[Path] = None,
    specA_label: str = "SpecA",
    specB_label: str = "SpecB",
) -> Dict[str, Any]:
    """
    Compare geometry analysis results across prompt sets.
    
    Supports N-way comparison. Each prompt set is identified by a label
    (e.g., "SpecA", "SpecB", "Diverse-group", "Diverse-regime") and
    results are found by matching files named:
        {model_key}-Geometry-{label}-{part_pattern}.json
    
    New interface (preferred):
        prompt_sets: Dict mapping label -> label (e.g., {"SpecA": "SpecA", "SpecB": "SpecB"})
        results_dir: Single directory containing all geometry outputs
        token_dist_files: Dict mapping label -> path to token dist file
    
    Old interface (backward compat):
        specA_results_dir, specB_results_dir, specA_label, specB_label, etc.
    """
    import glob
    
    # Handle backward compat
    if prompt_sets is None:
        if specA_results_dir is not None:
            results_dir = specA_results_dir  # All results are in the same dir
        prompt_sets = {specA_label: specA_label, specB_label: specB_label}
        token_dist_files = token_dist_files or {}
        if specA_token_dist_file:
            token_dist_files[specA_label] = specA_token_dist_file
        if specB_token_dist_file:
            token_dist_files[specB_label] = specB_token_dist_file
    
    if token_dist_files is None:
        token_dist_files = {}
    
    set_labels = list(prompt_sets.keys())
    
    if verbose:
        print("\n" + "="*70)
        print(f"GEOMETRY COMPARISON: {' vs '.join(set_labels)}")
        print("="*70)
    
    comparison = {
        "metadata": {
            "test_name": "Geometry-comparison",
            "model_key": model_key,
            "model_id": model_id,
            "prompt_sets": set_labels,
            "results_dir": str(results_dir),
        },
        "by_set": {},
        "comparison": {},
        "token_distributions": {},
        "summary": [],
    }
    
    # --- Part extractors ---
    part_extractors = {
        "part_a": {
            "file_pattern": "part_a_pr.json",
            "metrics": lambda d: {
                "PR_P_final": d.get("summary", {}).get("PR_P"),
                "PR_P_perp_final": d.get("summary", {}).get("PR_P_perp"),
                "n_prompts": d.get("summary", {}).get("n"),
            },
        },
        "part_b": {
            "file_pattern": "part_b_angles.json",
            "metrics": lambda d: {
                "within_group_angle": d.get("summary", {}).get("within_group_mean_angle"),
                "between_group_angle": d.get("summary", {}).get("between_group_mean_angle"),
                "separation_ratio": (
                    d.get("summary", {}).get("between_group_mean_angle", 0) /
                    max(d.get("summary", {}).get("within_group_mean_angle", 1), 1e-6)
                ),
            },
        },
        "part_e": {
            "file_pattern": "part_e_mu_landscape.json",
            "metrics": lambda d: {
                "mu_S_fraction": d.get("summary", {}).get("mu_energy_frac_S"),
                "mu_D_fraction": d.get("summary", {}).get("mu_energy_frac_D"),
                "mu_P_fraction": d.get("summary", {}).get("mu_energy_frac_P"),
                "mu_norm": d.get("summary", {}).get("mu_norm"),
                "k_D": d.get("summary", {}).get("k_D"),
            },
        },
        "part_g": {
            "file_pattern": "part_g_scramble.json",
            "metrics": lambda d: {
                "has_scramble_data": True,
                "scramble_layers": d.get("scramble_layers", []),
                "n_prompts_used": d.get("n_prompts_used"),
            },
        },
        "part_h": {
            "file_pattern": "part_h_injectivity.json",
            "metrics": lambda d: {
                "injectivity_cohens_d": d.get("global_injectivity", {}).get("separation_cohen_d"),
                "within_group_cosine_mean": d.get("global_injectivity", {}).get("within_group_cosine", {}).get("mean"),
                "between_group_cosine_mean": d.get("global_injectivity", {}).get("between_group_cosine", {}).get("mean"),
            },
        },
    }
    
    # Filter to only parts that were enabled in RUN_PARTS
    if run_parts:
        part_extractors = {k: v for k, v in part_extractors.items()
                           if run_parts.get(k, True)}
    
    # --- Load metrics for each prompt set ---
    for label in set_labels:
        comparison["by_set"][label] = {}
        for part_name, config in part_extractors.items():
            pattern = f"{model_key}-Geometry-{label}-{config['file_pattern']}"
            matches = list(Path(results_dir).glob(pattern))
            if matches:
                try:
                    with open(matches[0]) as f:
                        data = json.load(f)
                    comparison["by_set"][label][part_name] = config["metrics"](data)
                    if verbose:
                        print(f"  \u2713 Loaded {label} {part_name}")
                except Exception as e:
                    if verbose:
                        print(f"  \u2717 Failed to load {label} {part_name}: {e}")
            else:
                if verbose:
                    print(f"  - {label} {part_name} not found")
    
    # --- Cross-set comparison (all metrics side by side) ---
    if verbose:
        print("\nComputing comparisons...")
    
    for part_name in part_extractors.keys():
        part_comp = {}
        # Collect all metric names across sets
        all_metrics = set()
        for label in set_labels:
            all_metrics.update(comparison["by_set"].get(label, {}).get(part_name, {}).keys())
        
        for metric in all_metrics:
            vals = {}
            for label in set_labels:
                v = comparison["by_set"].get(label, {}).get(part_name, {}).get(metric)
                if v is not None:
                    vals[label] = v
            
            if len(vals) >= 2 and all(isinstance(v, (int, float)) for v in vals.values()):
                entry = {label: vals.get(label) for label in set_labels}
                # Add min/max/range for easy comparison
                numeric_vals = [v for v in vals.values() if isinstance(v, (int, float))]
                entry["min"] = min(numeric_vals)
                entry["max"] = max(numeric_vals)
                entry["range"] = max(numeric_vals) - min(numeric_vals)
                part_comp[metric] = entry
            elif vals:
                part_comp[metric] = {label: vals.get(label) for label in set_labels}
        
        if part_comp:
            comparison["comparison"][part_name] = part_comp
    
    # --- Token distributions ---
    if verbose:
        print("\nLoading token distributions...")
    
    for label in set_labels:
        td_file = token_dist_files.get(label)
        if td_file and Path(td_file).exists():
            try:
                with open(td_file) as f:
                    td_data = json.load(f)
                comparison["token_distributions"][label] = {
                    "mean_entropy": td_data.get("mean_entropy"),
                    "mean_pr": td_data.get("mean_pr"),
                    "mean_top1_prob": td_data.get("mean_top1_prob"),
                }
            except Exception:
                pass
    
    # Cross-compare token distributions
    td_metrics = ["mean_entropy", "mean_pr", "mean_top1_prob"]
    td_comp = {}
    for metric in td_metrics:
        vals = {}
        for label in set_labels:
            v = comparison["token_distributions"].get(label, {}).get(metric)
            if v is not None:
                vals[label] = v
        if len(vals) >= 2:
            td_comp[metric] = vals
    if td_comp:
        comparison["token_distributions"]["comparison"] = td_comp
    
    # --- Summary ---
    summary = []
    
    pc = comparison["comparison"]
    if pc.get("part_a", {}).get("PR_P_final"):
        d = pc["part_a"]["PR_P_final"]
        parts = [f"{label}={d[label]:.2f}" for label in set_labels if d.get(label) is not None]
        summary.append(f"PR(P): {', '.join(parts)}")
    
    if pc.get("part_b", {}).get("separation_ratio"):
        d = pc["part_b"]["separation_ratio"]
        parts = [f"{label}={d[label]:.2f}x" for label in set_labels if d.get(label) is not None]
        summary.append(f"Separation ratio: {', '.join(parts)}")
    
    if pc.get("part_e", {}).get("mu_S_fraction"):
        d = pc["part_e"]["mu_S_fraction"]
        parts = [f"{label}={d[label]*100:.1f}%" for label in set_labels if d.get(label) is not None]
        summary.append(f"\u03bc in S: {', '.join(parts)}")
    
    if pc.get("part_h", {}).get("injectivity_cohens_d"):
        d = pc["part_h"]["injectivity_cohens_d"]
        parts = [f"{label}={d[label]:.2f}" for label in set_labels if d.get(label) is not None]
        summary.append(f"Injectivity Cohen's d: {', '.join(parts)}")
    
    comparison["summary"] = summary
    
    if verbose:
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        for line in summary:
            print(f"  {line}")
    
    # Save
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    if verbose:
        print(f"\n\u2713 Saved comparison to: {output_file}")
    
    return comparison


def _first_key(d):
    """Get first key from dict or None."""
    if d and isinstance(d, dict):
        return list(d.keys())[0]
    return None


def _get_nested(d, keys):
    """Get nested dict value safely."""
    for key in keys:
        if d is None or not isinstance(d, dict):
            return None
        if key is None:
            return None
        d = d.get(key)
    return d


# =============================================================================
# MODULE INFO
# =============================================================================


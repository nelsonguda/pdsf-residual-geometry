"""pds_continuation.py

PDSF basis computation (canonical source) and SpecB continuation experiments.

Part of: PDSF — Prediction-Anchored Decomposition into Functional Subspaces

Paper:   "Geometric and Behavioral Stratification in Transformer Residual Streams"
         Nelson Guda, 2026
Repo:    https://github.com/nelsonguda/pdsf-residual-geometry

License: MIT (code), CC-BY 4.0 (data)


Purpose:
    1. Canonical P/D/S/F basis computation via sequential orthogonal decomposition.
       All other modules import basis functions from here.
    2. SpecB (Continuation) experiment: tests whether the PDSF decomposition has
       causal relevance for open-ended text generation by scrambling individual
       subspace components and measuring the effect on generated text.

Decomposition order (sequential orthogonal):
    P = unembedding vector for model's predicted next token (rank-1 per prompt)
    D = top-k PCA of (H after projecting out P)           ; see §3
    S = top-k PCA of (H after projecting out P and D)     ; see §3
    F = remainder: h - h_P - h_D - h_S                    ; see §3

Paper references:
    §3 and Appendix A.3 (PDSF decomposition methodology)
    §6.1–6.2, Figure 8 (persistent rotational scrambling; D/S divergence-timing dissociation)
    §7.1–7.3, Figure 9, Table 5 (static/dynamic partition of F)
    Appendix A.4 (intervention protocols)
    Companion paper (F sub-subspace analyses)

Key functions:
    compute_P_bases_from_predictions() — P basis from unembedding vectors
    compute_D_basis_global()           — D basis via adaptive-rank PCA
    compute_S_basis_global()           — S basis via adaptive-rank PCA
    decompose_hidden_state()           — Full P+D+S+F decomposition
    compute_specB2_bases()             — Build all bases for SpecB experiment
    run_specB2_experiment()            — P/D/S scramble on continuations
    run_F_transplant_experiment()      — F component transplant between prompts

NOTE on pca_basis():
    This module's pca_basis() returns Tuple[ndarray, ndarray] (basis, singular_values).
    The specA_analysis.py version returns only ndarray. They are NOT interchangeable.

Inputs:
    Model + tokenizer, prompt lists (SpecB JSON), hidden state cache.

Outputs:
    SpecB2Bases object, baseline/scrambled generation results, divergence summaries.

Dependencies:
    torch, numpy, tqdm
"""

from __future__ import annotations


import gc, json, time, warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import zlib
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
__version__ = "1.0"

# Above this rank, QR-based random rotation is O(rank^3) and intractable;
# fall back to permutation. See Appendix A.4.
ROTATION_MAX_RANK = 256


# === DATA STRUCTURES ===

@dataclass
class PromptRow:
    group_id: str
    variant_id: str
    prompt: str
    category: str = ""
    regime: str = ""
    mini_family_id: Optional[str] = None
    def get_family_id(self) -> str:
        return self.mini_family_id if self.mini_family_id else self.group_id

@dataclass
class PDSDecomposition:
    h_raw: np.ndarray
    h_P: np.ndarray
    h_D: np.ndarray
    h_S: np.ndarray
    h_F: np.ndarray
    energy_P: float = 0.0
    energy_D: float = 0.0
    energy_S: float = 0.0
    energy_F: float = 0.0
    energy_total: float = 0.0

@dataclass
class SpecB2Bases:
    P_bases: Dict[int, np.ndarray]   # P_bases[prompt_idx] = (d_model, 1) unembedding basis
    D_basis: np.ndarray              # (d_model, k_D)
    S_basis: np.ndarray              # (d_model, k_S)
    pred_token_ids: List[int] = field(default_factory=list)  # argmax token ID per prompt
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class BaselineResult:
    prompt: str
    prompt_len: int
    generated_ids: List[int]
    generated_text: str
    first_token_id: Optional[int]
    h_layer: np.ndarray
    decomposition: PDSDecomposition
    family_id: str

# === LINEAR ALGEBRA ===

def participation_ratio(X: np.ndarray, center: bool = True) -> float:
    """PR = effective dimensionality."""
    if X.size == 0: return 1.0
    if center: X = X - X.mean(axis=0, keepdims=True)
    try: s = np.linalg.svd(X, compute_uv=False, full_matrices=False)
    except: return 1.0
    eigs = s ** 2
    s1, s2 = float(eigs.sum()), float((eigs ** 2).sum())
    return (s1 ** 2) / s2 if s2 > 1e-12 else 1.0

def pca_basis(X: np.ndarray, k: int, center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if center: X = X - X.mean(axis=0, keepdims=True)
    try: U, s, Vt = np.linalg.svd(X, full_matrices=False)
    except: return np.zeros((X.shape[1], k), dtype=np.float32), np.zeros(min(X.shape), dtype=np.float32)
    k = max(1, min(k, Vt.shape[0], X.shape[1]))
    return Vt[:k].T.astype(np.float32), s.astype(np.float32)

def project_onto_basis(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0 or basis.shape[1] == 0: return np.zeros_like(X)
    if X.ndim == 1: return (X @ basis) @ basis.T
    return (X @ basis) @ basis.T

def project_out_basis(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0 or basis.shape[1] == 0: return X.copy()
    return X - project_onto_basis(X, basis)

# === MODEL UTILITIES ===

def get_model_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): return list(model.model.layers)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): return list(model.transformer.h)
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'): return list(model.gpt_neox.layers)
    raise ValueError(f"Unknown model: {type(model)}")

def get_model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, 'hf_device_map'):
        for key in ['model.embed_tokens', 'embed_tokens', 'transformer.wte']:
            if key in model.hf_device_map: return torch.device(f"cuda:{model.hf_device_map[key]}")
    try: return next(model.parameters()).device
    except: return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def apply_chat_template(model_id: str, prompt: str) -> str:
    m = model_id.lower()
    if "llama-3" in m or "llama-3" in m.replace(".", "-"):
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if "qwen" in m: return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if "mistral" in m or "mixtral" in m: return f"[INST] {prompt} [/INST]"
    if "gemma" in m: return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    if "phi-3" in m or "phi3" in m: return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    return prompt

def cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()

# === HIDDEN STATE EXTRACTION ===

def extract_hidden_at_layer(model, tokenizer, prompts: List[str], layer_idx: int, model_id: str = "", show_progress: bool = True) -> np.ndarray:
    device = get_model_input_device(model)
    hidden_states = []
    try:
        from tqdm import tqdm
        it = tqdm(prompts, desc=f"Layer {layer_idx}") if show_progress else prompts
    except: it = prompts
    for prompt in it:
        formatted = apply_chat_template(model_id, prompt) if model_id else prompt
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states.append(outputs.hidden_states[layer_idx][:, -1, :].detach().cpu().float().numpy()[0])
        del outputs
    return np.array(hidden_states, dtype=np.float32)

def extract_hidden_with_generation(model, tokenizer, prompt: str, layer_idx: int, n_generate: int, model_id: str = ""):
    device = get_model_input_device(model)
    formatted = apply_chat_template(model_id, prompt) if model_id else prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        h_layer = outputs.hidden_states[layer_idx][:, -1, :].detach().cpu().float().numpy()[0]
    with torch.no_grad():
        gen = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
            max_new_tokens=n_generate, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = gen[0, prompt_len:].tolist()
    return h_layer, gen_ids, tokenizer.decode(gen_ids, skip_special_tokens=True)

# === P/D/S BASIS COMPUTATION ===
#
# Canonical decomposition order (sequential orthogonal):
#   P = unembedding vector for model's predicted next token (rank-1 per prompt)
#   D = top-k PCA of (H after projecting out P)
#   S = top-k PCA of (H after projecting out P and D)
#   F = remainder: h - h_P - h_D - h_S
#
# These functions are the single source of truth for PDS bases.
# Both the Geometry and Continuation experiments use these functions.

# Canonical P basis computation. P is rank-1 per prompt: the normalized
# unembedding vector for the model's argmax prediction. See §3 / Appendix A.3.
def compute_P_bases_from_predictions(
    pred_token_ids: List[int],
    unembed_np: np.ndarray,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Compute per-prompt P basis from model's predicted next token.
    
    P is the unembedding vector for the token the model actually predicts.
    This is rank-1 per prompt and layer-independent.
    
    Args:
        pred_token_ids: List of argmax token IDs, one per prompt
        unembed_np: Unembedding matrix (vocab_size, d_model) — from model.lm_head.weight
        
    Returns:
        P_bases: Dict[int, ndarray] — P_bases[prompt_idx] = (d_model, 1) basis
        info: Dict with computation details
    """
    # Ensure (vocab, d_model) orientation
    W = unembed_np
    if W.ndim != 2:
        raise ValueError("unembed_np must be 2D")
    if W.shape[0] < W.shape[1]:
        W = W.T  # Transpose to (vocab, d_model)
    
    d_model = W.shape[1]
    P_bases = {}
    unique_tokens = set(pred_token_ids)
    
    # Pre-compute basis for each unique token (avoid redundant work)
    token_to_basis = {}
    for tid in unique_tokens:
        v = W[tid].astype(np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
        token_to_basis[tid] = v.reshape(-1, 1)  # (d_model, 1)
    
    for i, tid in enumerate(pred_token_ids):
        P_bases[i] = token_to_basis[tid]
    
    info = {
        "n_prompts": len(pred_token_ids),
        "n_unique_tokens": len(unique_tokens),
        "rank_per_prompt": 1,
        "method": "unembedding_vector",
    }
    return P_bases, info


# Second step in sequential orthogonal decomposition P→D→S→F.
# Adaptive rank via participation ratio avoids imposing fixed dimensionality. See §3 / Appendix A.3.
def compute_D_basis_global(H: np.ndarray, P_bases: Dict[int, np.ndarray], prompt_ids: List[Any] = None, k_D: Union[int, str] = "adaptive", k_min: int = 2, k_max: int = 12):
    """Compute D basis as top-k PCA of H after projecting out P.
    
    D captures the dominant variance structure orthogonal to the predictive direction.
    This is the second step in the sequential orthogonal decomposition: P → D → S → F.
    
    Args:
        H: Hidden states (n_prompts, d_model)
        P_bases: Dict[int, ndarray] — per-prompt P basis, P_bases[prompt_idx] = (d_model, 1)
        prompt_ids: Unused, kept for backward compatibility
        k_D: Rank for D basis ("adaptive" or int)
    """
    n, d = H.shape
    Hp = np.zeros_like(H)
    for i in range(n):
        Pb = P_bases.get(i)
        Hp[i] = project_out_basis(H[i:i+1, :], Pb)[0] if Pb is not None and Pb.size > 0 else H[i]
    
    # Check for issues in Hp
    if np.isnan(Hp).any() or np.isinf(Hp).any():
        print(f"  ⚠️ Hp contains NaN/Inf after projecting out P")
        Hp = np.nan_to_num(Hp, nan=0.0, posinf=0.0, neginf=0.0)
    
    pr = participation_ratio(Hp, center=True)
    if not np.isfinite(pr) or pr < 1.0:
        # Fallback: compute PR directly from eigenvalues
        Hp_centered = Hp - Hp.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(Hp_centered, compute_uv=False)
            eigs = s ** 2
            pr = (eigs.sum() ** 2) / (eigs ** 2).sum() if (eigs ** 2).sum() > 1e-12 else 2.0
        except:
            pr = 2.0  # Default to k_min
    
    k = max(k_min, min(k_max, round(pr), n-1)) if k_D == "adaptive" else max(1, min(int(k_D), k_max, n-1))
    D_basis, svs = pca_basis(Hp, k=k, center=True)
    return D_basis, {"k": int(k), "pr": float(pr), "n": n}

# Third step in sequential decomposition. Captures situational context
# after P and D are removed. See §3 / Appendix A.3.
def compute_S_basis_global(H: np.ndarray, P_bases: Dict[int, np.ndarray], D_basis: np.ndarray, prompt_ids: List[Any] = None, k_S: Union[int, str] = "adaptive", k_min: int = 4, k_max: int = 16):
    """Compute S basis as top-k PCA of H after projecting out P and D.
    
    S captures the next layer of variance structure after P and D are removed.
    This is the third step in the sequential orthogonal decomposition: P → D → S → F.
    
    Args:
        H: Hidden states (n_prompts, d_model)
        P_bases: Dict[int, ndarray] — per-prompt P basis
        D_basis: D basis (d_model, k_D)
        prompt_ids: Unused, kept for backward compatibility
        k_S: Rank for S basis ("adaptive" or int)
    """
    n, d = H.shape
    Hs = np.zeros_like(H)
    for i in range(n):
        h = H[i]
        Pb = P_bases.get(i)
        hp = project_out_basis(h, Pb) if Pb is not None and Pb.size > 0 else h.copy()
        Hs[i] = project_out_basis(hp, D_basis) if D_basis is not None and D_basis.size > 0 else hp
    
    # Check for issues in Hs
    if np.isnan(Hs).any() or np.isinf(Hs).any():
        print(f"  ⚠️ Hs contains NaN/Inf after projecting out P and D")
        Hs = np.nan_to_num(Hs, nan=0.0, posinf=0.0, neginf=0.0)
    
    pr = participation_ratio(Hs, center=True)
    if not np.isfinite(pr) or pr < 1.0:
        # Fallback: compute PR directly from eigenvalues
        Hs_centered = Hs - Hs.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(Hs_centered, compute_uv=False)
            eigs = s ** 2
            pr = (eigs.sum() ** 2) / (eigs ** 2).sum() if (eigs ** 2).sum() > 1e-12 else 4.0
        except:
            pr = 4.0  # Default to k_min
    
    k = max(k_min, min(k_max, round(pr), n-1)) if k_S == "adaptive" else max(1, min(int(k_S), k_max, n-1))
    S_basis, svs = pca_basis(Hs, k=k, center=True)
    return S_basis, {"k": int(k), "pr": float(pr), "n": n}

# Sequential orthogonal projection: P first, then D from P-residual, then
# S from P+D residual, F = remainder. Guarantees h = h_P + h_D + h_S + h_F
# with mutual orthogonality. See §3 / Appendix A.3.
def decompose_hidden_state(h: np.ndarray, P_basis, D_basis, S_basis) -> PDSDecomposition:
    d = h.shape[0]
    Pb = P_basis if P_basis is not None and P_basis.size > 0 else np.zeros((d, 0), dtype=np.float32)
    Db = D_basis if D_basis is not None and D_basis.size > 0 else np.zeros((d, 0), dtype=np.float32)
    Sb = S_basis if S_basis is not None and S_basis.size > 0 else np.zeros((d, 0), dtype=np.float32)
    hP = project_onto_basis(h, Pb)
    hp = h - hP
    hD = project_onto_basis(hp, Db)
    hsr = hp - hD
    hS = project_onto_basis(hsr, Sb)
    hr = hsr - hS
    return PDSDecomposition(h, hP, hD, hS, hr, float(np.dot(hP,hP)), float(np.dot(hD,hD)), float(np.dot(hS,hS)), float(np.dot(hr,hr)), float(np.dot(h,h)))

def compute_specB2_bases(model, tokenizer, rows: List[PromptRow], scramble_layer: int, model_id: str, config: Dict[str, Any], verbose: bool = True, H_precomputed: np.ndarray = None, pred_token_ids: List[int] = None, unembed_np: np.ndarray = None) -> SpecB2Bases:
    """Compute unified P/D/S bases for continuation experiments.
    
    Uses sequential orthogonal decomposition:
      P = unembedding vector for model's predicted next token (rank-1 per prompt)
      D = top-k PCA of (H after projecting out P)
      S = top-k PCA of (H after projecting out P and D)
    
    Args:
        model, tokenizer: Model for extraction (not needed if H_precomputed provided)
        rows: Prompt data
        scramble_layer: Layer to extract hidden states from
        model_id: Model identifier for chat template
        config: Dict with d_rank, s_rank settings
        H_precomputed: Pre-extracted hidden states (n_prompts, d_model). If provided, skips extraction.
        pred_token_ids: Pre-computed predicted token IDs per prompt. If None, runs a forward pass.
        unembed_np: Unembedding matrix (vocab_size, d_model). Required for P basis.
    """
    if verbose: print("="*60 + "\nCOMPUTING PDS BASES (unified)\n" + "="*60)
    _bases_start = time.time()
    d_rank, s_rank = config.get("d_rank", "adaptive"), config.get("s_rank", "adaptive")
    prompts = [r.prompt for r in rows]
    
    # Step 1: Hidden states
    if H_precomputed is not None:
        H = H_precomputed
        if verbose: print(f"\nStep 1: Using pre-extracted hidden states at layer {scramble_layer}, shape {H.shape}")
    else:
        if verbose: print(f"\nStep 1: Extracting hidden states at layer {scramble_layer}...")
        H = extract_hidden_at_layer(model, tokenizer, prompts, scramble_layer, model_id=model_id, show_progress=verbose)
        if verbose: print(f"  Shape: {H.shape}")
    
    # NaN/Inf check
    if np.isnan(H).any():
        nan_count = np.isnan(H).sum()
        nan_rows = np.isnan(H).any(axis=1).sum()
        print(f"  ⚠️ WARNING: H contains {nan_count} NaN values in {nan_rows} rows!")
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isinf(H).any():
        inf_count = np.isinf(H).sum()
        print(f"  ⚠️ WARNING: H contains {inf_count} Inf values!")
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"  H stats: mean={H.mean():.4f}, std={H.std():.4f}, min={H.min():.4f}, max={H.max():.4f}")
    
    # Step 2: Get predicted token IDs (for P basis)
    if pred_token_ids is None:
        if verbose: print(f"\nStep 2: Generating predicted tokens for P basis...")
        pred_token_ids = _get_predicted_tokens(model, tokenizer, prompts, model_id, verbose)
    else:
        if verbose: print(f"\nStep 2: Using pre-computed predicted tokens ({len(set(pred_token_ids))} unique)")
    
    # Step 3: Get unembedding matrix (for P basis)
    if unembed_np is None:
        if verbose: print(f"  Extracting unembedding matrix...")
        unembed_weight = model.get_output_embeddings().weight
        
        # Handle meta tensors (common with device_map="auto" on large/quantized models)
        if unembed_weight.device.type == 'meta':
            if verbose: print(f"  ⚠️ Output embeddings on meta device, searching for materialized weights...")
            found = False
            for name, param in model.named_parameters():
                if 'lm_head' in name and 'weight' in name and param.device.type != 'meta':
                    unembed_weight = param
                    found = True
                    if verbose: print(f"    Found: {name} on {param.device}")
                    break
                if 'embed_tokens' in name and 'weight' in name and param.device.type != 'meta':
                    unembed_weight = param
                    found = True
                    if verbose: print(f"    Found (tied): {name} on {param.device}")
                    break
            if not found:
                raise RuntimeError(
                    "Cannot find materialized embedding weights. "
                    "Model may not be fully loaded — check device_map configuration."
                )
        
        unembed_np = unembed_weight.detach().cpu().float().numpy()
    
    # Step 4: P basis (unembedding vectors for predicted tokens)
    if verbose: print(f"\nStep 3: Computing P bases (unembedding-based)...")
    P_bases, P_info = compute_P_bases_from_predictions(pred_token_ids, unembed_np)
    if verbose: print(f"  {P_info['n_unique_tokens']} unique tokens, rank-1 per prompt")
    
    # Step 5: D basis (top PCA after projecting out P)
    if verbose: print(f"\nStep 4: Computing D basis...")
    D_basis, D_info = compute_D_basis_global(H, P_bases, k_D=d_rank)
    if verbose: print(f"  Shape: {D_basis.shape}, PR={D_info['pr']:.2f}")
    
    # Step 6: S basis (top PCA after projecting out P and D)
    if verbose: print(f"\nStep 5: Computing S basis...")
    S_basis, S_info = compute_S_basis_global(H, P_bases, D_basis, k_S=s_rank)
    if verbose: print(f"  Shape: {S_basis.shape}, PR={S_info['pr']:.2f}")
    
    # Energy stats
    eP, eD, eS, eR, eT = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_decompositions = 0
    for i in range(len(rows)):
        dc = decompose_hidden_state(H[i], P_bases.get(i), D_basis, S_basis)
        if np.isfinite(dc.energy_total) and dc.energy_total > 0:
            eP += dc.energy_P; eD += dc.energy_D; eS += dc.energy_S; eR += dc.energy_F; eT += dc.energy_total
            valid_decompositions += 1
    
    if valid_decompositions == 0 or eT < 1e-12:
        print(f"  ⚠️ WARNING: No valid decompositions! valid={valid_decompositions}, total_energy={eT}")
        stats = {"energy_P_frac": 0.0, "energy_D_frac": 0.0, "energy_S_frac": 0.0, "energy_F_frac": 1.0}
    else:
        stats = {"energy_P_frac": eP/eT, "energy_D_frac": eD/eT, "energy_S_frac": eS/eT, "energy_F_frac": eR/eT}
    
    if verbose: print(f"\n  Energy: P={stats['energy_P_frac']:.3f}, D={stats['energy_D_frac']:.3f}, S={stats['energy_S_frac']:.3f}")
    if verbose:
        _bases_elapsed = time.time() - _bases_start
        _bm, _bs = divmod(int(_bases_elapsed), 60)
        print("\n" + "="*60 + f"\n✓ BASES COMPUTED [{_bm}:{_bs:02d}]\n" + "="*60)
    meta = {"model_id": model_id, "scramble_layer": scramble_layer, "n_prompts": len(rows),
            "P_info": P_info, "D_info": D_info, "S_info": S_info,
            "timestamp": datetime.now(timezone.utc).isoformat()}
    return SpecB2Bases(P_bases=P_bases, D_basis=D_basis, S_basis=S_basis,
                       pred_token_ids=pred_token_ids, metadata=meta, stats=stats)


def _get_predicted_tokens(model, tokenizer, prompts: List[str], model_id: str, verbose: bool = True) -> List[int]:
    """Run one forward pass per prompt to get argmax predicted token ID."""
    device = get_model_input_device(model)
    pred_ids = []
    try:
        from tqdm import tqdm
        it = tqdm(prompts, desc="Predicting tokens") if verbose else prompts
    except ImportError:
        it = prompts
    for prompt in it:
        formatted = apply_chat_template(model_id, prompt) if model_id else prompt
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1, :]
        pred_ids.append(int(logits.argmax(dim=-1).item()))
        del outputs
    return pred_ids

# === SCRAMBLE OPERATIONS ===

# Default scramble method. "rotation" applies a random orthogonal rotation
# within the subspace (basis-independent, preserves energy). See §6.1 and Appendix A.4.
SCRAMBLE_METHOD = "rotation"

def scramble_coefficients_permutation(h_comp: np.ndarray, basis: np.ndarray, seed: int) -> np.ndarray:
    """
    Scramble by permuting coefficients in basis.
    
    This shuffles which basis vector gets which coefficient value.
    - Preserves the set of coefficient values
    - Preserves total energy
    - Basis-dependent: effect depends on PCA ordering
    
    Args:
        h_comp: Component vector to scramble (in model space)
        basis: Orthonormal basis matrix [d_model, rank]
        seed: Random seed for reproducibility
    
    Returns:
        Scrambled component (in model space)
    """
    if basis.size == 0 or basis.shape[1] == 0: 
        return h_comp.copy()
    coeffs = h_comp @ basis  # Project onto basis
    perm = np.random.RandomState(seed).permutation(len(coeffs))
    h_in = coeffs @ basis.T  # Original component in basis
    return coeffs[perm] @ basis.T + (h_comp - h_in)  # Scrambled + residual


def scramble_coefficients_rotation(h_comp: np.ndarray, basis: np.ndarray, seed: int) -> np.ndarray:
    """
    Scramble by applying random rotation within the subspace.
    
    This rotates the vector to a random orientation within the subspace.
    - Preserves total energy (||c'|| = ||c||)
    - Basis-independent: same effect regardless of PCA ordering
    - Mathematically cleaner for testing "does direction matter?"
    
    Args:
        h_comp: Component vector to scramble (in model space)
        basis: Orthonormal basis matrix [d_model, rank]
        seed: Random seed for reproducibility
    
    Returns:
        Scrambled component (in model space)
    """
    if basis.size == 0 or basis.shape[1] == 0: 
        return h_comp.copy()
    
    rank = basis.shape[1]

    # Dense QR rotation is O(rank^3) and becomes infeasible for large residual/flat spaces
    # (rank≈d_model). In that regime the experiment can appear to hang or be killed for memory/time.
    # We therefore fall back to coefficient permutation above ROTATION_MAX_RANK.
    if rank > ROTATION_MAX_RANK:
        return scramble_coefficients_permutation(h_comp, basis, seed)

    coeffs = h_comp @ basis  # Project onto basis [rank]
    
    # Generate random orthogonal matrix via QR decomposition
    rng = np.random.RandomState(seed)
    R, _ = np.linalg.qr(rng.randn(rank, rank).astype(np.float32))
    
    # Rotate coefficients within subspace
    coeffs_rot = R @ coeffs  # [rank]
    
    # Reconstruct in model space
    h_in = coeffs @ basis.T  # Original component
    h_rot = coeffs_rot @ basis.T  # Rotated component
    
    return h_rot + (h_comp - h_in)  # Rotated + residual (anything not in basis)


def scramble_coefficients(h_comp: np.ndarray, basis: np.ndarray, seed: int, method: str = None) -> np.ndarray:
    """
    Scramble coefficients using the specified method.
    
    Args:
        h_comp: Component vector to scramble (in model space)
        basis: Orthonormal basis matrix [d_model, rank]
        seed: Random seed for reproducibility
        method: "rotation" or "permutation" (default: global SCRAMBLE_METHOD)
    
    Returns:
        Scrambled component (in model space)
    
    Methods:
        - "rotation": Random orthogonal rotation within subspace (recommended)
          Basis-independent, preserves energy, tests if direction matters
        - "permutation": Shuffle coefficient assignments
          Basis-dependent, can be more severe, tests if specific PCs matter
    """
    method = method or SCRAMBLE_METHOD
    
    if method == "permutation":
        return scramble_coefficients_permutation(h_comp, basis, seed)
    else:  # default to rotation
        return scramble_coefficients_rotation(h_comp, basis, seed)


def scramble_P_component(h, P_basis, D_basis, S_basis, seed):
    dc = decompose_hidden_state(h, P_basis, D_basis, S_basis)
    return scramble_coefficients(dc.h_P, P_basis, seed) + dc.h_D + dc.h_S + dc.h_F

def scramble_D_component(h, P_basis, D_basis, S_basis, seed):
    dc = decompose_hidden_state(h, P_basis, D_basis, S_basis)
    return dc.h_P + scramble_coefficients(dc.h_D, D_basis, seed) + dc.h_S + dc.h_F

def scramble_S_component(h, P_basis, D_basis, S_basis, seed):
    """
    Scramble S (Stylistic) component using configured scramble method.
    
    S is the top-k PCA dimensions capturing surface texture, elaboration,
    and word choice. Typically ~16 dimensions, ~5% of energy.
    
    The Framework (F) component is left unchanged.
    """
    dc = decompose_hidden_state(h, P_basis, D_basis, S_basis)
    return dc.h_P + dc.h_D + scramble_coefficients(dc.h_S, S_basis, seed) + dc.h_F


# === GENERATION WITH HOOKS ===

def generate_with_scramble(model, tokenizer, prompt: str, scramble_layer: int, scramble_fn, P_basis, D_basis, S_basis, n_generate: int, seed: int, model_id: str = "", once: bool = False):
    device = get_model_input_device(model)
    formatted = apply_chat_template(model_id, prompt) if model_id else prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    P_np = P_basis if isinstance(P_basis, np.ndarray) else P_basis.cpu().numpy()
    D_np = D_basis if isinstance(D_basis, np.ndarray) else D_basis.cpu().numpy()
    S_np = S_basis if isinstance(S_basis, np.ndarray) else S_basis.cpu().numpy()
    hook_state = {"fired": False, "handle": None}
    def hook_fn(module, inp, output):
        hook_state["fired"] = True
        ht = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        h_np = ht[:, -1, :][0].detach().cpu().float().numpy()
        h_scr = scramble_fn(h_np, P_np, D_np, S_np, seed)
        ht_new = ht.clone()
        ht_new[:, -1, :] = torch.tensor(h_scr, device=ht.device, dtype=ht.dtype)
        if once and hook_state["handle"] is not None:
            hook_state["handle"].remove()
            hook_state["handle"] = None
        return (ht_new,) + rest if rest else ht_new
    layers = get_model_layers(model)
    block_idx = scramble_layer - 1
    if block_idx < 0 or block_idx >= len(layers): raise IndexError(f"Invalid layer {scramble_layer}")
    handle = layers[block_idx].register_forward_hook(hook_fn)
    hook_state["handle"] = handle
    try:
        with torch.no_grad():
            gen = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                max_new_tokens=n_generate, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    finally:
        if hook_state["handle"] is not None:
            hook_state["handle"].remove()
    gen_ids = gen[0, prompt_len:].tolist()
    return {"generated_ids": gen_ids, "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True), "hook_fired": hook_state["fired"]}

def generate_with_P_scramble(model, tokenizer, prompt, scramble_layer, P_basis, D_basis, S_basis, n_generate, seed, model_id=""):
    return generate_with_scramble(model, tokenizer, prompt, scramble_layer, scramble_P_component, P_basis, D_basis, S_basis, n_generate, seed, model_id)

def generate_with_D_scramble(model, tokenizer, prompt, scramble_layer, P_basis, D_basis, S_basis, n_generate, seed, model_id=""):
    return generate_with_scramble(model, tokenizer, prompt, scramble_layer, scramble_D_component, P_basis, D_basis, S_basis, n_generate, seed, model_id)

def generate_with_S_scramble(model, tokenizer, prompt, scramble_layer, P_basis, D_basis, S_basis, n_generate, seed, model_id=""):
    """Generate with S (Stylistic) component scrambled."""
    return generate_with_scramble(model, tokenizer, prompt, scramble_layer, scramble_S_component, P_basis, D_basis, S_basis, n_generate, seed, model_id)

def generate_baseline(model, tokenizer, prompt: str, n_generate: int, model_id: str = ""):
    device = get_model_input_device(model)
    formatted = apply_chat_template(model_id, prompt) if model_id else prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
            max_new_tokens=n_generate, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = gen[0, prompt_len:].tolist()
    return {"prompt": prompt, "prompt_len": int(prompt_len), "generated_ids": gen_ids, 
            "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True), 
            "first_token_id": gen_ids[0] if gen_ids else None}

def generate_baselines_with_decomposition(model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases, scramble_layer: int, n_generate: int, model_id: str, verbose: bool = True) -> Dict[str, BaselineResult]:
    baselines = {}
    try:
        from tqdm import tqdm
        it = tqdm(enumerate(rows), total=len(rows), desc="Baselines") if verbose else enumerate(rows)
    except: it = enumerate(rows)
    for i, row in it:
        h, gen_ids, gen_text = extract_hidden_with_generation(model, tokenizer, row.prompt, scramble_layer, n_generate, model_id)
        Pb = bases.P_bases.get(i, np.zeros((h.shape[0], 0), dtype=np.float32))
        dc = decompose_hidden_state(h, Pb, bases.D_basis, bases.S_basis)
        key = f"{row.group_id}_{row.variant_id}"
        baselines[key] = BaselineResult(row.prompt, len(tokenizer.encode(apply_chat_template(model_id, row.prompt) if model_id else row.prompt)),
            gen_ids, gen_text, gen_ids[0] if gen_ids else None, h, dc, row.get_family_id())
    return baselines


def rebaseline_at_layer(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases,
    existing_baselines: Dict[str, BaselineResult], new_layer: int,
    model_id: str, verbose: bool = True
) -> Dict[str, BaselineResult]:
    """Re-extract hidden states and decomposition at a different layer,
    reusing generated text/ids from existing baselines.
    
    Generated text is layer-independent (unmodified model produces the same output
    regardless of which layer we plan to intervene at). Only the hidden state
    extraction and PDSF decomposition are layer-specific. This saves one full
    generation forward pass per prompt compared to generate_baselines_with_decomposition.
    
    Args:
        model, tokenizer: Model and tokenizer
        rows: Prompt rows (same order as existing baselines)
        bases: Bases computed at new_layer (from compute_specB2_bases)
        existing_baselines: Baselines from a previous layer (for generated text reuse)
        new_layer: Layer index for hidden state extraction
        model_id: Model ID for chat template
        verbose: Show progress bar
    
    Returns:
        Dict[str, BaselineResult] with decomposition at new_layer but same generated text
    """
    baselines = {}
    try:
        from tqdm import tqdm
        it = tqdm(enumerate(rows), total=len(rows), desc=f"Rebaseline L{new_layer}") if verbose else enumerate(rows)
    except: it = enumerate(rows)
    for i, row in it:
        key = f"{row.group_id}_{row.variant_id}"
        existing = existing_baselines.get(key)
        if existing is None:
            continue
        
        # Re-extract hidden state at new_layer (single forward pass, no generation)
        device = get_model_input_device(model)
        formatted = apply_chat_template(model_id, row.prompt) if model_id else row.prompt
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            h = outputs.hidden_states[new_layer][:, -1, :].detach().cpu().float().numpy()[0]
        
        # Decompose at new layer using new bases
        Pb = bases.P_bases.get(i, np.zeros((h.shape[0], 0), dtype=np.float32))
        dc = decompose_hidden_state(h, Pb, bases.D_basis, bases.S_basis)
        
        # Reuse generated text from existing baselines (layer-independent)
        # NOTE: `existing` may be a BaselineResult instance (fresh run) OR a plain dict
        # (loaded from JSON via load_specB2_baselines). Support both.
        if isinstance(existing, dict):
            ex_prompt = existing.get("prompt")
            ex_prompt_len = existing.get("prompt_len")
            ex_generated_ids = existing.get("generated_ids")
            ex_generated_text = existing.get("generated_text")
            ex_first_token_id = existing.get("first_token_id")
            ex_family_id = existing.get("family_id", "")
        else:
            ex_prompt = existing.prompt
            ex_prompt_len = existing.prompt_len
            ex_generated_ids = existing.generated_ids
            ex_generated_text = existing.generated_text
            ex_first_token_id = existing.first_token_id
            ex_family_id = existing.family_id

        baselines[key] = BaselineResult(
            ex_prompt, ex_prompt_len,
            ex_generated_ids, ex_generated_text,
            ex_first_token_id, h, dc, ex_family_id
        )
    return baselines

# === COMPARISON ===

def compare_generations(baseline: Dict, scrambled: Dict) -> Dict:
    b_ids, s_ids = baseline.get("generated_ids", []), scrambled.get("generated_ids", [])
    b_txt, s_txt = baseline.get("generated_text", ""), scrambled.get("generated_text", "")
    div_tok = None
    for i in range(min(len(b_ids), len(s_ids))):
        if b_ids[i] != s_ids[i]: div_tok = i; break
    if div_tok is None and len(b_ids) != len(s_ids): div_tok = min(len(b_ids), len(s_ids))
    b_w, s_w = b_txt.split(), s_txt.split()
    div_word = None
    for i in range(min(len(b_w), len(s_w))):
        if b_w[i] != s_w[i]: div_word = i; break
    if div_word is None and len(b_w) != len(s_w): div_word = min(len(b_w), len(s_w))
    return {"first_divergence_token": div_tok, "first_divergence_word": div_word, "identical": b_ids == s_ids,
            "prefix_same_frac_tokens": div_tok / max(1, max(len(b_ids), len(s_ids))) if div_tok is not None else 1.0}

def compute_divergence_summary(results: List[Dict]) -> Dict:
    """Aggregate per-prompt divergence records.

    Rows come in two shapes. Most conditions nest the comparison under "effect";
    run_F_subspace_rotation_experiment splats it onto the row itself and leaves
    "effect" as None. Reading only "effect" silently produced an empty summary for
    F-topK and F-dynamic — identical_pct 0.0 and mean_divergence_token None — even
    though every per-prompt field was correct. Accept either shape.
    """
    n = len(results)
    if n == 0: return {"n_prompts": 0}
    divs, ident, imm = [], 0, 0
    for r in results:
        ef = r.get("effect")
        if not isinstance(ef, dict):
            ef = r                      # fields live on the row itself
        if ef.get("identical"): ident += 1
        else:
            dt = ef.get("first_divergence_token")
            if dt is not None: divs.append(dt)
            if dt == 0: imm += 1
    sm = {"n_prompts": n, "identical_count": ident, "identical_pct": 100*ident/n,
          "immediate_divergence_count": imm, "immediate_divergence_pct": 100*imm/max(1, n-ident)}
    if divs: sm["mean_divergence_token"] = float(np.mean(divs)); sm["median_divergence_token"] = float(np.median(divs)); sm["std_divergence_token"] = float(np.std(divs))
    else: sm["mean_divergence_token"] = sm["median_divergence_token"] = sm["std_divergence_token"] = None
    return sm

# === FILE I/O ===

def to_jsonable(x):
    if x is None or isinstance(x, (str, int, float, bool)): return x
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().float().tolist()
    if isinstance(x, dict): return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [to_jsonable(v) for v in x]
    if isinstance(x, PDSDecomposition): return {"h_F": x.h_F.tolist() if isinstance(x.h_F, np.ndarray) else x.h_F, "energy_P": x.energy_P, "energy_D": x.energy_D, "energy_S": x.energy_S, "energy_F": x.energy_F, "energy_total": x.energy_total}
    if isinstance(x, BaselineResult): return {"prompt": x.prompt, "prompt_len": x.prompt_len, "generated_ids": x.generated_ids, "generated_text": x.generated_text, "first_token_id": x.first_token_id, "family_id": x.family_id, "h_layer": x.h_layer.tolist() if isinstance(x.h_layer, np.ndarray) else x.h_layer, "decomposition": to_jsonable(x.decomposition)}
    return str(x)

def save_json(obj, path: Path, indent: int = 2):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(to_jsonable(obj), f, indent=indent)

def save_specB2_bases(bases: SpecB2Bases, output_dir: Path, model_key: str, file_prefix: str = "Continuation") -> str:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fp = output_dir / f"{model_key}-{file_prefix}-bases.json"
    save_json({"metadata": bases.metadata, "stats": bases.stats,
               "P_bases": {str(k): v.tolist() for k,v in bases.P_bases.items()},
               "D_basis": bases.D_basis.tolist(), "S_basis": bases.S_basis.tolist(),
               "pred_token_ids": bases.pred_token_ids}, fp)
    return str(fp)

def load_specB2_bases(filepath: Path) -> SpecB2Bases:
    with open(filepath) as f: data = json.load(f)
    # P_bases keys are prompt indices (int), stored as strings in JSON
    P_bases = {int(k): np.array(v, dtype=np.float32) for k,v in data["P_bases"].items()}
    return SpecB2Bases(
        P_bases=P_bases,
        D_basis=np.array(data["D_basis"], dtype=np.float32),
        S_basis=np.array(data["S_basis"], dtype=np.float32),
        pred_token_ids=data.get("pred_token_ids", []),
        metadata=data.get("metadata", {}),
        stats=data.get("stats", {}))

def specB2_bases_to_analysis_dict(bases: SpecB2Bases) -> Dict[str, Any]:
    """
    Convert SpecB2Bases to a dict format compatible with prompt analysis.
    
    This allows reusing bases computed for SpecB2 scramble experiments
    in the SpecA-style analysis without recomputation.
    
    Args:
        bases: SpecB2Bases object from compute_specB2_bases or load_specB2_bases
    
    Returns:
        Dict with P_bases, D_basis, S_basis, metadata, and stats
    """
    return {
        "P_bases": bases.P_bases,
        "D_basis": bases.D_basis,
        "S_basis": bases.S_basis,
        "metadata": bases.metadata,
        "stats": bases.stats,
        "P_info": bases.metadata.get("P_info", {}),
        "D_info": bases.metadata.get("D_info", {}),
        "S_info": bases.metadata.get("S_info", {}),
    }

def save_specB2_baselines(baselines: Dict[str, BaselineResult], output_dir: Path, model_key: str, metadata=None, file_prefix: str = "Continuation") -> str:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fp = output_dir / f"{model_key}-{file_prefix}-baselines.json"
    save_json({"metadata": metadata or {}, "baselines": {k: to_jsonable(v) for k,v in baselines.items()}}, fp)
    return str(fp)

def load_specB2_baselines(filepath: Path) -> Tuple[Dict[str, Dict], Dict]:
    """
    Load baselines from a saved JSON file.
    
    Returns:
        baselines: Dict mapping prompt keys to baseline result dicts
        metadata: Dict of metadata from the saved file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Baselines file not found: {filepath}")
    
    with open(filepath) as f:
        data = json.load(f)
    
    baselines = data.get("baselines", {})
    metadata = data.get("metadata", {})
    
    return baselines, metadata


def _bl_get(bl, attr, default=None):
    """
    Access a field from a baseline-related object, supporting:
    - BaselineResult dataclass instances (attribute access)
    - PDSDecomposition dataclass instances (attribute access) 
    - Plain dicts (loaded from cache)
    - Any other object with the attribute (getattr fallback)
    """
    if isinstance(bl, dict):
        return bl.get(attr, default)
    return getattr(bl, attr, default)


def save_specB2_results(results: List[Dict], summary: Dict, output_dir: Path, model_key: str, exp_name: str, metadata=None, file_prefix: str = "Continuation") -> str:
    """Save SpecB2 experiment results with depth in filename."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    
    # Include depth in filename if provided in metadata
    depth_pct = metadata.get("depth_pct") if metadata else None
    if depth_pct is not None:
        depth_str = f"_d{int(depth_pct)}"
    else:
        depth_str = ""
    
    fp = output_dir / f"{model_key}-{file_prefix}-{exp_name}{depth_str}.json"
    save_json({
        "metadata": {
            "experiment": f"SpecB2_{exp_name}", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }, 
        "summary": summary, 
        "results": results
    }, fp)
    return str(fp)

# === PIPELINE INTEGRATION ===


# === F (FRAMEWORK) INTERVENTIONS ===
#
# The Framework (F) component is the remainder after removing:
#   P (Predictive) + D (Discursive) + S (Stylistic).
#
# These interventions probe whether the high-rank F component contains meaningful structure.
#
#   (1) F_attenuate:     h_F <- (1-alpha) * h_F  (reduce energy, preserve direction)
#   (2) F_mix:           h_F <- permute_dims(h_F) * random_sign  (destroy structure, preserve energy)
#   (3) F_transplant:    h_F_recipient <- energy_matched(h_F_donor)  (cross-prompt identity test)
#
# These must run inside the *same* SpecB2 runner path as the P/D/S scrambles, otherwise notebook
# flags will be ignored and no outputs will be written.


def _F_from_hidden(h: np.ndarray, P_basis: np.ndarray, D_basis: np.ndarray, S_basis: np.ndarray) -> Tuple[np.ndarray, PDSDecomposition]:
    """Return (h_F_vector, decomposition) where h_F_vector is the Framework component."""
    dc = decompose_hidden_state(h, P_basis, D_basis, S_basis)
    return dc.h_F, dc


def _energy_match(vec: np.ndarray, target_norm: float) -> Tuple[np.ndarray, float]:
    """Scale vec to have L2 norm == target_norm. Returns (scaled_vec, scale)."""
    vnorm = float(np.linalg.norm(vec))
    if vnorm < 1e-12 or target_norm < 1e-12:
        return vec.astype(np.float32).copy(), 1.0
    s = target_norm / vnorm
    return (vec * s).astype(np.float32), float(s)


def build_transplant_pairs(rows: List[PromptRow], policy: str = "fast", attenuate_alpha_control: float = 0.5) -> List[Dict[str, Any]]:
    """Deterministic donor/recipient pairing policy for F transplants.

    Canonical order: sort by group_id, then variant_id. No RNG.

    Detects whether prompts have regime info. If so, uses regime-aware pairing
    (within-regime + cross-regime). Otherwise, uses legacy within/cross-group pairing.

    Policies:
      - fast: recipients/group R=4
      - full: recipients/group R=6

    REGIME-AWARE PAIRING (new prompt set with regime field):
      - within_regime donors: different group, same regime
      - cross_regime donors:  different regime
      No within-group pairs (already covered by old prompt set results).

    LEGACY PAIRING (old prompt set without regime field):
      - within donors: same group, offsets +1,+2
      - cross donors:  group+1 and group+6 (mod C)

    Controls (both modes):
      - 10% identity controls (donor==recipient)
      - 10% F attenuate controls (donor=None, alpha=attenuate_alpha_control)
    """
    policy = (policy or "fast").lower()
    if policy not in ("fast", "full"):
        raise ValueError(f"Unknown pair policy: {policy}")

    R = 4 if policy == "fast" else 6

    by_group: Dict[str, List[PromptRow]] = {}
    for r in rows:
        by_group.setdefault(r.group_id, []).append(r)

    group_ids = sorted(by_group.keys())
    if not group_ids:
        return []

    for gid in group_ids:
        by_group[gid] = sorted(by_group[gid], key=lambda x: str(x.variant_id))

    C = len(group_ids)

    # Detect regime-aware mode
    has_regime = any(r.regime for r in rows)

    # Build regime -> groups mapping if available
    regime_to_groups: Dict[str, List[str]] = {}
    group_to_regime: Dict[str, str] = {}
    if has_regime:
        for r in rows:
            regime_to_groups.setdefault(r.regime, set()).add(r.group_id)
            group_to_regime[r.group_id] = r.regime
        regime_to_groups = {k: sorted(v) for k, v in regime_to_groups.items()}
        regime_ids = sorted(regime_to_groups.keys())

    # Recipient list in canonical order (first R variants per group)
    recipients: List[Tuple[str, int]] = []
    for gid in group_ids:
        for j in range(min(R, len(by_group[gid]))):
            recipients.append((gid, j))

    nR = len(recipients)
    n_ident = int(np.ceil(0.10 * nR))
    n_abla = int(np.ceil(0.10 * nR))

    def key_for(gid: str, j: int) -> str:
        rr = by_group[gid][j]
        return f"{rr.group_id}_{rr.variant_id}"

    pairs: List[Dict[str, Any]] = []

    # Identity controls
    for idx in range(n_ident):
        gid, j = recipients[idx]
        rec = by_group[gid][j]
        pairs.append({
            "pair_id": f"{policy}|identity|{gid}|{rec.variant_id}",
            "pair_type": "identity_control",
            "pair_policy": policy,
            "donor_key": key_for(gid, j),
            "recipient_key": key_for(gid, j),
            "donor_group_id": gid,
            "recipient_group_id": gid,
            "donor_category": rec.category,
            "recipient_category": rec.category,
            "donor_regime": getattr(rec, 'regime', ''),
            "recipient_regime": getattr(rec, 'regime', ''),
            "donor_variant_id": rec.variant_id,
            "recipient_variant_id": rec.variant_id,
            "energy_match": False,
            "attenuate_alpha": None,
        })

    # F attenuate controls
    for idx in range(n_ident, min(n_ident + n_abla, nR)):
        gid, j = recipients[idx]
        rec = by_group[gid][j]
        pairs.append({
            "pair_id": f"{policy}|F_attenuate_control|{gid}|{rec.variant_id}",
            "pair_type": "F_attenuate_control",
            "pair_policy": policy,
            "donor_key": None,
            "recipient_key": key_for(gid, j),
            "donor_group_id": None,
            "recipient_group_id": gid,
            "donor_category": None,
            "recipient_category": rec.category,
            "donor_regime": None,
            "recipient_regime": getattr(rec, 'regime', ''),
            "donor_variant_id": None,
            "recipient_variant_id": rec.variant_id,
            "energy_match": True,
            "attenuate_alpha": float(attenuate_alpha_control),
        })

    if has_regime:
        # REGIME-AWARE PAIRING
        for gid, j in recipients:
            rec = by_group[gid][j]
            rec_regime = group_to_regime.get(gid, "")

            # Within-regime donors: different group, same regime
            same_regime_groups = [g for g in regime_to_groups.get(rec_regime, []) if g != gid]
            for donor_gid in same_regime_groups[:2]:  # Up to 2 within-regime donors
                dj = min(j, len(by_group[donor_gid]) - 1)
                donor = by_group[donor_gid][dj]
                pairs.append({
                    "pair_id": f"{policy}|within_regime|{donor_gid}->{gid}|{donor.variant_id}->{rec.variant_id}",
                    "pair_type": "within_regime",
                    "pair_policy": policy,
                    "donor_key": key_for(donor_gid, dj),
                    "recipient_key": key_for(gid, j),
                    "donor_group_id": donor_gid,
                    "recipient_group_id": gid,
                    "donor_category": donor.category,
                    "recipient_category": rec.category,
                    "donor_regime": group_to_regime.get(donor_gid, ""),
                    "recipient_regime": rec_regime,
                    "donor_variant_id": donor.variant_id,
                    "recipient_variant_id": rec.variant_id,
                    "energy_match": True,
                    "attenuate_alpha": None,
                })

            # Cross-regime donors: different regime (near + far by regime index)
            ri = regime_ids.index(rec_regime) if rec_regime in regime_ids else 0
            cross_regimes = [regime_ids[(ri + 1) % len(regime_ids)],
                             regime_ids[(ri + len(regime_ids)//2) % len(regime_ids)]]
            for cr in cross_regimes[:2]:
                cross_groups = regime_to_groups.get(cr, [])
                if not cross_groups:
                    continue
                donor_gid = cross_groups[0]
                dj = min(j, len(by_group[donor_gid]) - 1)
                donor = by_group[donor_gid][dj]
                pairs.append({
                    "pair_id": f"{policy}|cross_regime|{donor_gid}->{gid}|{donor.variant_id}->{rec.variant_id}",
                    "pair_type": "cross_regime",
                    "pair_policy": policy,
                    "donor_key": key_for(donor_gid, dj),
                    "recipient_key": key_for(gid, j),
                    "donor_group_id": donor_gid,
                    "recipient_group_id": gid,
                    "donor_category": donor.category,
                    "recipient_category": rec.category,
                    "donor_regime": cr,
                    "recipient_regime": rec_regime,
                    "donor_variant_id": donor.variant_id,
                    "recipient_variant_id": rec.variant_id,
                    "energy_match": True,
                    "attenuate_alpha": None,
                })
    else:
        # LEGACY PAIRING (old prompt set without regime)
        kw, kx = 2, 2
        for gid, j in recipients:
            rec = by_group[gid][j]

            # Within-group
            for off in (1, 2)[:kw]:
                dj = (j + off) % len(by_group[gid])
                donor = by_group[gid][dj]
                pairs.append({
                    "pair_id": f"{policy}|within|{gid}|{donor.variant_id}->{rec.variant_id}",
                    "pair_type": "within",
                    "pair_policy": policy,
                    "donor_key": key_for(gid, dj),
                    "recipient_key": key_for(gid, j),
                    "donor_group_id": gid,
                    "recipient_group_id": gid,
                    "donor_category": donor.category,
                    "recipient_category": rec.category,
                    "donor_variant_id": donor.variant_id,
                    "recipient_variant_id": rec.variant_id,
                    "energy_match": True,
                    "attenuate_alpha": None,
                })

            # Cross-group: near + far (mod C)
            gi = group_ids.index(gid)
            cross_groups = [group_ids[(gi + 1) % C], group_ids[(gi + 6) % C]][:kx]
            for cg in cross_groups:
                dj = min(j, len(by_group[cg]) - 1)
                donor = by_group[cg][dj]
                pairs.append({
                    "pair_id": f"{policy}|cross|{cg}->{gid}|{donor.variant_id}->{rec.variant_id}",
                    "pair_type": "cross",
                    "pair_policy": policy,
                    "donor_key": key_for(cg, dj),
                    "recipient_key": key_for(gid, j),
                    "donor_group_id": cg,
                    "recipient_group_id": gid,
                    "donor_category": donor.category,
                    "recipient_category": rec.category,
                    "donor_variant_id": donor.variant_id,
                    "recipient_variant_id": rec.variant_id,
                    "energy_match": True,
                    "attenuate_alpha": None,
                })

    return pairs


def run_F_attenuate_experiment(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases, baselines: Dict[str, BaselineResult],
    scramble_layer: int, n_generate: int, seed: int, model_id: str, alpha: float, verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attenuate the Framework (F) component for each prompt."""
    from tqdm import tqdm
    results: List[Dict[str, Any]] = []
    for row in (tqdm(rows, desc="F attenuate") if verbose else rows):
        key = f"{row.group_id}_{row.variant_id}"
        bl = baselines[key]
        bl_fid = _bl_get(bl, "family_id", "")
        _hdim = bases.D_basis.shape[0]
        Pb = bases.P_bases.get(bl_fid, np.zeros((_hdim, 0), dtype=np.float32))

        def fn(h, P_basis, D_basis, S_basis, _seed):
            h_F_vec, dc = _F_from_hidden(h, P_basis, D_basis, S_basis)
            h_F_new = ((1.0 - float(alpha)) * h_F_vec).astype(np.float32)
            return (dc.h_P + dc.h_D + dc.h_S + h_F_new).astype(np.float32)

        scr = generate_with_scramble(model, tokenizer, row.prompt, scramble_layer, fn, Pb, bases.D_basis, bases.S_basis, n_generate, seed, model_id, once=True)
        ef = compare_generations({"generated_ids": _bl_get(bl, "generated_ids", []), "generated_text": _bl_get(bl, "generated_text", "")}, scr)
        results.append({
            "prompt_key": key,
            "group_id": row.group_id,
            "variant_id": row.variant_id,
            "category": row.category,
            "baseline_text": _bl_get(bl, "generated_text", ""),
            "scrambled_text": scr.get("generated_text"),
            "effect": ef,
            "F_alpha": float(alpha),
            "hook_fired": bool(scr.get("hook_fired")),
        })

    return results, compute_divergence_summary(results)


def run_F_mix_experiment(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases, baselines: Dict[str, BaselineResult],
    scramble_layer: int, n_generate: int, seed: int, model_id: str, verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Destroy coordinate structure in F space using permute+sign flips."""
    from tqdm import tqdm
    results: List[Dict[str, Any]] = []
    
    row_iter = tqdm(rows, desc="F mix") if verbose else rows
    for row in row_iter:
        key = f"{row.group_id}_{row.variant_id}"
        bl = baselines[key]
        bl_fid = _bl_get(bl, "family_id", "")
        _hdim = bases.D_basis.shape[0]
        Pb = bases.P_bases.get(bl_fid, np.zeros((_hdim, 0), dtype=np.float32))

        def fn(h, P_basis, D_basis, S_basis, _seed):
            h_F_vec, dc = _F_from_hidden(h, P_basis, D_basis, S_basis)
            # zlib.crc32, not the built-in hash(): Python randomizes str hashing
            # per process (PEP 456), so hash(key) gives a different permutation on
            # every kernel restart and F-mix outputs are then irreproducible.
            # crc32 is stable across processes, versions and platforms.
            rng = np.random.RandomState(seed + (zlib.crc32(key.encode("utf-8")) % 10_000_000))
            perm = rng.permutation(h_F_vec.shape[0])
            signs = rng.choice([-1.0, 1.0], size=h_F_vec.shape[0]).astype(np.float32)
            h_F_new = (h_F_vec[perm] * signs).astype(np.float32)
            return (dc.h_P + dc.h_D + dc.h_S + h_F_new).astype(np.float32)

        scr = generate_with_scramble(model, tokenizer, row.prompt, scramble_layer, fn, Pb, bases.D_basis, bases.S_basis, n_generate, seed, model_id, once=True)
        ef = compare_generations({"generated_ids": _bl_get(bl, "generated_ids", []), "generated_text": _bl_get(bl, "generated_text", "")}, scr)
        results.append({
            "prompt_key": key,
            "group_id": row.group_id,
            "variant_id": row.variant_id,
            "category": row.category,
            "baseline_text": _bl_get(bl, "generated_text", ""),
            "scrambled_text": scr.get("generated_text"),
            "effect": ef,
            "hook_fired": bool(scr.get("hook_fired")),
        })

    return results, compute_divergence_summary(results)


def run_F_transplant_experiment(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases, baselines: Dict[str, BaselineResult],
    scramble_layer: int, n_generate: int, seed: int, model_id: str,
    pair_policy: str = "fast", attenuate_alpha_control: float = 0.5, verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Replace recipient F vector with donor F vector (energy matched by default)."""
    pairs = build_transplant_pairs(rows, policy=pair_policy, attenuate_alpha_control=attenuate_alpha_control)
    if verbose:
        print(f"\n  Flat transplant pairs: {len(pairs)} (policy={pair_policy}, alpha_control={attenuate_alpha_control})")

    donor_F: Dict[str, np.ndarray] = {}
    for k, bl in baselines.items():
        decomp = _bl_get(bl, "decomposition", {})
        h_res = _bl_get(decomp, "h_F", None)
        if h_res is None:
            h_res = _bl_get(decomp, "h_residual", None)
        if h_res is not None:
            h_res = np.asarray(h_res, dtype=np.float32)
        else:
            # Recompute from h_layer if decomposition vectors weren't cached
            h_layer = _bl_get(bl, "h_layer", None)
            if h_layer is not None:
                h_layer = np.asarray(h_layer, dtype=np.float32)
                bl_fid = _bl_get(bl, "family_id", "")
                Pb = bases.P_bases.get(bl_fid, np.zeros((h_layer.shape[0], 0), dtype=np.float32))
                h_res, _ = _F_from_hidden(h_layer, Pb, bases.D_basis, bases.S_basis)
            else:
                raise ValueError(f"Baseline '{k}' has no h_F or h_layer — cannot run flat transplant from cache. "
                                 "Delete the cached baselines file to force regeneration.")
        donor_F[k] = h_res

    results: List[Dict[str, Any]] = []
    
    from tqdm import tqdm
    pair_iter = tqdm(pairs, desc="F transplant") if verbose else pairs
    for ps in pair_iter:
        rec_key = ps["recipient_key"]
        rec_bl = baselines[rec_key]
        rec_fid = _bl_get(rec_bl, "family_id", "")
        _hdim = bases.D_basis.shape[0]
        Pb = bases.P_bases.get(rec_fid, np.zeros((_hdim, 0), dtype=np.float32))

        if ps["pair_type"] == "F_attenuate_control":
            alpha = float(ps["attenuate_alpha"])
            def fn(h, P_basis, D_basis, S_basis, _seed):
                h_F_vec, dc = _F_from_hidden(h, P_basis, D_basis, S_basis)
                h_F_target = ((1.0 - alpha) * h_F_vec).astype(np.float32)
                return (dc.h_P + dc.h_D + dc.h_S + h_F_target).astype(np.float32)
            donor_key = None
            scale_used = 1.0
            F_norm_donor = None
            energy_match = True
        else:
            donor_key = ps["donor_key"]
            dF = donor_F[donor_key]
            _rec_decomp = _bl_get(rec_bl, "decomposition", {})
            _rec_hres = _bl_get(_rec_decomp, "h_F", None)
            if _rec_hres is None:
                _rec_hres = _bl_get(_rec_decomp, "h_residual", None)
            if _rec_hres is not None:
                target_norm = float(np.linalg.norm(np.asarray(_rec_hres)))
            else:
                _rec_eres = _bl_get(_rec_decomp, "energy_F", None)
                if _rec_eres is None:
                    _rec_eres = _bl_get(_rec_decomp, "energy_residual", None)
                if _rec_eres is not None and _rec_eres > 0:
                    target_norm = float(np.sqrt(_rec_eres))
                else:
                    # Last resort: use donor norm (no energy matching)
                    target_norm = float(np.linalg.norm(dF))
            if ps.get("energy_match", True):
                dF_scaled, scale_used = _energy_match(dF, target_norm)
                energy_match = True
            else:
                dF_scaled, scale_used = dF.copy(), 1.0
                energy_match = False
            F_norm_donor = float(np.linalg.norm(dF))
            def fn(h, P_basis, D_basis, S_basis, _seed):
                _, dc = _F_from_hidden(h, P_basis, D_basis, S_basis)
                return (dc.h_P + dc.h_D + dc.h_S + dF_scaled).astype(np.float32)

        scr = generate_with_scramble(model, tokenizer, _bl_get(rec_bl, "prompt", ""), scramble_layer, fn, Pb, bases.D_basis, bases.S_basis, n_generate, seed, model_id, once=True)
        ef = compare_generations({"generated_ids": _bl_get(rec_bl, "generated_ids", []), "generated_text": _bl_get(rec_bl, "generated_text", "")}, scr)
        results.append({
            "pair_id": ps["pair_id"],
            "pair_type": ps["pair_type"],
            "pair_policy": ps["pair_policy"],
            "donor_key": donor_key,
            "recipient_key": rec_key,
            "donor_group_id": ps.get("donor_group_id"),
            "recipient_group_id": ps.get("recipient_group_id"),
            "donor_category": ps.get("donor_category"),
            "recipient_category": ps.get("recipient_category"),
            "donor_variant_id": ps.get("donor_variant_id"),
            "recipient_variant_id": ps.get("recipient_variant_id"),
            "energy_match": bool(energy_match),
            "F_norm_donor": F_norm_donor,
            "F_norm_recipient": target_norm,  # reuse already-computed value
            "F_scale_applied": float(scale_used),
            "baseline_text": _bl_get(rec_bl, "generated_text", ""),
            "scrambled_text": scr.get("generated_text"),
            "effect": ef,
            "hook_fired": bool(scr.get("hook_fired")),
        })

    return results, compute_divergence_summary(results)


def run_specB2_experiment(model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases, baselines: Dict, scramble_layer: int, exp_type: str, n_generate: int, seed: int, model_id: str, verbose: bool = True):
    """Run a single SpecB2 scramble experiment (P, D, or S).
    
    Includes per-prompt error handling: if a prompt fails (e.g., tokenizer 
    doesn't support the language), it is skipped with a warning.
    """
    gen_fn = {"P": generate_with_P_scramble, "D": generate_with_D_scramble, "S": generate_with_S_scramble}[exp_type]
    if verbose: print(f"\n{'='*60}\nRUNNING {exp_type}-SCRAMBLE\n{'='*60}")
    results = []
    skipped = 0
    try:
        from tqdm import tqdm
        it = tqdm(rows, desc=f"{exp_type}-scramble") if verbose else rows
    except: it = rows
    for i, row in enumerate(it):
        key = f"{row.group_id}_{row.variant_id}"
        bl = baselines.get(key)
        if bl is None: continue
        try:
            fid = row.get_family_id()
            Pb = bases.P_bases.get(i, np.zeros((bases.D_basis.shape[0], 0), dtype=np.float32))
            scr = gen_fn(model, tokenizer, row.prompt, scramble_layer, Pb, bases.D_basis, bases.S_basis, n_generate, seed+i, model_id)
            bl_dict = {"generated_ids": bl.generated_ids if isinstance(bl, BaselineResult) else bl.get("generated_ids", []),
                       "generated_text": bl.generated_text if isinstance(bl, BaselineResult) else bl.get("generated_text", "")}
            ef = compare_generations(bl_dict, scr)
            results.append({"group_id": row.group_id, "variant_id": row.variant_id, "family_id": fid,
                "category": row.category, "regime": getattr(row, 'regime', ''),
                "prompt": row.prompt[:100] + "..." if len(row.prompt) > 100 else row.prompt,
                "baseline_text": bl_dict["generated_text"], "scrambled_text": scr["generated_text"], "effect": ef, "hook_fired": scr.get("hook_fired", False)})
        except Exception as e:
            skipped += 1
            regime_info = f" (regime={row.regime})" if getattr(row, 'regime', '') else ""
            if verbose: print(f"\n  ⚠ Skipped {key}{regime_info}: {e}")
            continue
    sm = compute_divergence_summary(results)
    if skipped > 0:
        sm["skipped_prompts"] = skipped
        if verbose: print(f"\n  ⚠ {skipped} prompts skipped due to errors")
    if verbose: print(f"\nSummary: {sm['n_prompts']} prompts, {sm['identical_pct']:.1f}% identical, {sm['immediate_divergence_pct']:.1f}% immediate")
    return results, sm

def run_specB2_for_pipeline(model, tokenizer, rows: List[PromptRow], output_dir: Path, model_key: str, model_id: str, config: Dict, options: Dict, verbose: bool = True, H_precomputed: np.ndarray = None, pred_token_ids: List[int] = None, unembed_np: np.ndarray = None):
    """Run complete SpecB2 experiment suite for pipeline integration.
    
    Config options:
        scramble_layer_fraction: 0.0-1.0, depth for scramble intervention (default 0.70)
        scramble_type: "rotation" (recommended) or "permutation"
            - "rotation": Random orthogonal rotation within subspace (basis-independent)
            - "permutation": Shuffle coefficient assignments (basis-dependent)

    Args:
        H_precomputed: Pre-extracted hidden states at scramble layer (n_prompts, d_model).
            If provided, skips redundant extraction in compute_specB2_bases.
        pred_token_ids: Pre-computed predicted token IDs per prompt.
            If provided, skips redundant forward pass in compute_specB2_bases.
        unembed_np: Pre-extracted unembedding matrix.
            If provided, skips redundant extraction in compute_specB2_bases.
        
    Options:
        use_cached_bases: If True, load bases from existing file if available (default True)
        use_cached_baselines: If True, load baselines from existing file if available (default True)
        bases_file: Path to existing bases file (optional, auto-detected if not provided)
        baselines_file: Path to existing baselines file (optional, auto-detected if not provided)
    """
    global SCRAMBLE_METHOD
    
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    _pipeline_start = time.time()
    
    def _elapsed():
        """Elapsed since pipeline start, formatted as M:SS or H:MM:SS."""
        e = time.time() - _pipeline_start
        h, rem = divmod(int(e), 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    
    scr_frac = config.get("scramble_layer_fraction", 0.70)
    n_gen = config.get("n_generate", 64)
    seed = config.get("seed", 42)
    file_prefix = config.get("file_prefix", "Continuation")
    # S scramble is always S (Stylistic = top-k PCA dims). No method dispatch needed.
    scramble_type = config.get("scramble_type", "rotation")  # "rotation" or "permutation"
    n_layers = len(get_model_layers(model))
    scr_layer = int(n_layers * scr_frac)
    depth_pct = int(scr_frac * 100)  # For filename and metadata
    
    # Set global scramble method
    SCRAMBLE_METHOD = scramble_type
    
    # Cache options
    use_cached_bases = options.get("use_cached_bases", True)
    use_cached_baselines = options.get("use_cached_baselines", True)
    
    if verbose: 
        print("\n" + "="*70 + f"\nPDSF CONTINUATION SCRAMBLE [{file_prefix}]\n" + "="*70)
        print(f"  Scramble layer: {scr_layer}/{n_layers} ({depth_pct}% depth)")
        print(f"  Scramble type: {scramble_type}")
        print(f"  Started at [{_elapsed()}]")
    
    all_res = {
        "model_key": model_key, 
        "scramble_layer": scr_layer, 
        "scramble_depth_pct": depth_pct,
        "scramble_type": scramble_type,
        "n_layers": n_layers,
        "file_paths": {}
    }
    
    # Common metadata for all experiments
    base_metadata = {
        "scramble_layer": scr_layer,
        "depth_pct": depth_pct,
        "n_layers": n_layers,
        "scramble_type": scramble_type,
    }
    
    # === BASES: Load from cache or compute ===
    bases = None
    bases_file = options.get("bases_file") or (output_dir / f"{model_key}-{file_prefix}-bases.json")
    bases_file = Path(bases_file)
    
    if use_cached_bases and bases_file.exists():
        if verbose: print(f"  [{_elapsed()}] Loading cached bases from: {bases_file.name}")
        try:
            bases = load_specB2_bases(bases_file)
            all_res["file_paths"]["bases"] = str(bases_file)
            if verbose: print(f"    ✓ Loaded bases (P: {len(bases.P_bases)} families, D: {bases.D_basis.shape}, S: {bases.S_basis.shape})")
        except Exception as e:
            if verbose: print(f"    ✗ Failed to load bases: {e}")
            bases = None
    
    if bases is None:
        if verbose: print(f"  [{_elapsed()}] Computing bases...")
        bases = compute_specB2_bases(model, tokenizer, rows, scr_layer, model_id, config, verbose, unembed_np=unembed_np)
        if options.get("save_bases", True): 
            all_res["file_paths"]["bases"] = save_specB2_bases(bases, output_dir, model_key, file_prefix=file_prefix)
    
    # === SAVE ANALYSIS STATS (for prompt_analysis_utils compatibility) ===
    if options.get("save_analysis_stats", True):
        analysis_stats = {
            "metadata": {
                "model_key": model_key,
                "model_id": model_id,
                "scramble_layer": scr_layer,
                "depth_pct": depth_pct,
                "n_layers": n_layers,
                "n_prompts": len(rows),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "participation_ratios": {
                "PR_P": bases.metadata.get("P_info", {}).get("pr") if bases.metadata.get("P_info") else None,
                "PR_D": bases.metadata.get("D_info", {}).get("pr") if bases.metadata.get("D_info") else None,
                "PR_S": bases.metadata.get("S_info", {}).get("pr") if bases.metadata.get("S_info") else None,
            },
            "energy_fractions": bases.stats,
            "subspace_info": {
                "P": bases.metadata.get("P_info", {}),
                "D": bases.metadata.get("D_info", {}),
                "S": bases.metadata.get("S_info", {}),
            },
            "subspace_dims": {
                "P_families": len(bases.P_bases),
                "D_rank": bases.D_basis.shape[1] if bases.D_basis.size > 0 else 0,
                "S_rank": bases.S_basis.shape[1] if bases.S_basis.size > 0 else 0,
            },
        }
        stats_file = output_dir / f"{model_key}-{file_prefix}-analysis_stats.json"
        save_json(analysis_stats, stats_file)
        all_res["file_paths"]["analysis_stats"] = str(stats_file)
        if verbose: print(f"  Saved analysis stats to: {stats_file.name}")
    
    # === BASELINES: Load from cache or generate ===
    baselines = None
    baselines_file = options.get("baselines_file") or (output_dir / f"{model_key}-{file_prefix}-baselines.json")
    baselines_file = Path(baselines_file)
    
    if use_cached_baselines and baselines_file.exists():
        if verbose: print(f"  [{_elapsed()}] Loading cached baselines from: {baselines_file.name}")
        try:
            baselines, bl_metadata = load_specB2_baselines(baselines_file)
            all_res["file_paths"]["baselines"] = str(baselines_file)
            if verbose: print(f"    ✓ Loaded {len(baselines)} baselines")
            
            # Warn if baselines were generated at different depth
            cached_depth = bl_metadata.get("depth_pct")
            if cached_depth is not None and cached_depth != depth_pct:
                if verbose: print(f"    ⚠ Warning: Cached baselines from depth {cached_depth}%, current depth {depth_pct}%")
            
            # Check if cached baselines have h_layer vectors needed for flat transplant
            run_F_transplant = options.get("run_F_transplant", False)
            if run_F_transplant and baselines:
                sample_bl = next(iter(baselines.values()))
                has_hlayer = sample_bl.get("h_layer") is not None if isinstance(sample_bl, dict) else hasattr(sample_bl, "h_layer")
                has_hres = False
                decomp = sample_bl.get("decomposition", {}) if isinstance(sample_bl, dict) else getattr(sample_bl, "decomposition", {})
                if isinstance(decomp, dict):
                    has_hres = (decomp.get("h_F") is not None) or (decomp.get("h_residual") is not None)
                if not has_hlayer and not has_hres:
                    if verbose: print(f"    ⚠ Cached baselines lack h_layer/h_F vectors needed for flat transplant")
                    if verbose: print(f"    → Regenerating baselines with vector data...")
                    baselines = None  # Force regeneration
        except Exception as e:
            if verbose: print(f"    ✗ Failed to load baselines: {e}")
            baselines = None
    
    if baselines is None:
        if verbose: print(f"  [{_elapsed()}] Generating baselines...")
        baselines = generate_baselines_with_decomposition(model, tokenizer, rows, bases, scr_layer, n_gen, model_id, verbose)
        if options.get("save_baselines", True): 
            all_res["file_paths"]["baselines"] = save_specB2_baselines(baselines, output_dir, model_key, {"model_id": model_id, "n": len(baselines), **base_metadata}, file_prefix=file_prefix)
    
    # === EXPERIMENTS ===
    exps = {}

    # === FLAT-SPACE INTERVENTIONS (post S-topK) ===
    # Note: F interventions are separate from P/D/S scrambles. F = remainder after P+D+S.
    run_F_attenuate = options.get("run_F_attenuate", False)
    run_F_mix = options.get("run_F_mix", False)
    run_F_transplant = options.get("run_F_transplant", False)
    pair_policy = config.get("pair_policy", "fast")
    attenuate_alpha_control = float(config.get("attenuate_alpha_control", 0.5))
    F_attenuate_alpha = float(config.get("F_attenuate_alpha", attenuate_alpha_control))

    if run_F_attenuate:
        if verbose:
            print(f"\n  Running F attenuate (alpha={F_attenuate_alpha})...")
        res_fa, sm_fa = run_F_attenuate_experiment(
            model, tokenizer, rows, bases, baselines, scr_layer, n_gen, seed, model_id,
            alpha=F_attenuate_alpha, verbose=verbose
        )
        exp_metadata = {**base_metadata, "F_alpha": F_attenuate_alpha}
        all_res["file_paths"]["F_attenuate"] = save_specB2_results(res_fa, sm_fa, output_dir, model_key, "F_attenuate", exp_metadata, file_prefix=file_prefix)
        exps["F_attenuate"] = {"summary": sm_fa}

    if run_F_mix:
        if verbose:
            print(f"\n  Running F mix (permute+sign) ...")
        res_fm, sm_fm = run_F_mix_experiment(
            model, tokenizer, rows, bases, baselines, scr_layer, n_gen, seed, model_id, verbose=verbose
        )
        exp_metadata = {**base_metadata, "F_mix_seed": seed}
        all_res["file_paths"]["F_mix"] = save_specB2_results(res_fm, sm_fm, output_dir, model_key, "F_mix", exp_metadata, file_prefix=file_prefix)
        exps["F_mix"] = {"summary": sm_fm}

    if run_F_transplant:
        if verbose:
            print(f"\n  Running F transplant (policy={pair_policy}) ...")
        res_ft, sm_ft = run_F_transplant_experiment(
            model, tokenizer, rows, bases, baselines, scr_layer, n_gen, seed, model_id,
            pair_policy=pair_policy, attenuate_alpha_control=attenuate_alpha_control, verbose=verbose
        )
        exp_metadata = {**base_metadata, "pair_policy": pair_policy, "attenuate_alpha_control": attenuate_alpha_control}
        all_res["file_paths"]["F_transplant"] = save_specB2_results(res_ft, sm_ft, output_dir, model_key, "F_transplant", exp_metadata, file_prefix=file_prefix)
        exps["F_transplant"] = {"summary": sm_ft}

    # === F-DYNAMICS SUB-SUBSPACE INTERVENTIONS (ported from V11.8 Paper 2, 2026-07-02) ===
    # Feeds Paper 1 §6.4 (F-topK rotation), §7.3 (F-dynamic rotation), §7.2 (step-offset F injection).
    # Trimmed to the three Paper-1 tests; uses only ported + release-present helpers.
    run_passive_tracking = options.get("run_passive_F_tracking", False)
    run_step_offset = options.get("run_step_offset_injection", False)
    run_F_topK_rot = options.get("run_F_topK_rotation", False)
    run_F_dynamic_rot = options.get("run_F_dynamic_rotation", False)

    if run_passive_tracking or run_step_offset or run_F_topK_rot or run_F_dynamic_rot:
        f_dyn_n_gen_passive = int(config.get("f_dynamics_passive_n_generate", 32))
        f_dyn_n_gen_offset = int(config.get("f_dynamics_offset_n_generate", 32))
        f_dyn_k_values = config.get("f_dynamics_offset_k_values", [0, 1, 2, 3, 5])
        f_dyn_store_vectors = config.get("f_dynamics_store_vectors", True)
        f_dyn_metadata = {**base_metadata, "experiment_suite": "F_dynamics_v11.8_ported"}

        # F-dynamic requires passive tracking first (produces in-memory _F_vectors)
        tracking_results = None
        if run_passive_tracking or run_F_dynamic_rot:
            if verbose:
                print("\n  Running passive F-tracking (prerequisite for F-dynamic)...")
            tracking_results, tracking_summary = run_passive_F_tracking(
                model, tokenizer, rows, bases, baselines, scr_layer,
                n_generate=f_dyn_n_gen_passive, seed=seed, model_id=model_id,
                store_vectors=f_dyn_store_vectors, verbose=verbose)
            if run_passive_tracking:
                _save = [{k: v for k, v in r.items() if not k.startswith("_")} for r in tracking_results]
                meta = {**f_dyn_metadata, "n_generate": f_dyn_n_gen_passive, "store_vectors": f_dyn_store_vectors}
                all_res["file_paths"]["passive_F_tracking"] = save_specB2_results(
                    _save, tracking_summary, output_dir, model_key, "passive_F_tracking", meta, file_prefix=file_prefix)
                exps["passive_F_tracking"] = {"summary": tracking_summary}

        # F-topK rotation: build the F-topK basis at the intervention layer, then rotate
        if run_F_topK_rot:
            H_int = extract_hidden_at_layer(model, tokenizer, [r.prompt for r in rows], scr_layer,
                                            model_id=model_id, show_progress=False)
            f_topK_basis = None; f_topK_info = None
            if H_int is not None:
                pred_tokens = _get_predicted_tokens(model, tokenizer, [r.prompt for r in rows], model_id, verbose=False)
                if hasattr(model, 'lm_head'):
                    uemb = model.lm_head.weight.detach().cpu().float().numpy()
                else:
                    uemb = model.get_output_embeddings().weight.detach().cpu().float().numpy()
                _Pb_int = {}
                for pi, tid in enumerate(pred_tokens):
                    if tid is not None and 0 <= tid < uemb.shape[0]:
                        v = uemb[tid].copy(); v = v / (np.linalg.norm(v) + 1e-12)
                        _Pb_int[pi] = v.reshape(-1, 1)
                f_topK_basis, f_topK_info = compute_F_topK_basis(H_int, _Pb_int, bases.D_basis, bases.S_basis)
                all_res["f_topK_info"] = f_topK_info
            if f_topK_basis is not None:
                if verbose:
                    print(f"\n  F-topK rotation (rank={f_topK_info['k']})...")
                ftk_r, ftk_s = run_F_subspace_rotation_experiment(
                    model, tokenizer, rows, bases, baselines, f_topK_basis, scr_layer,
                    condition_name="F_topK_rotation", n_generate=n_gen, seed=seed, model_id=model_id, verbose=verbose)
                meta = {**f_dyn_metadata, "analysis_type": "F_topK_rotation", "f_topK_rank": f_topK_info['k']}
                all_res["file_paths"]["F_topK_rotation"] = save_specB2_results(
                    ftk_r, ftk_s, output_dir, model_key, "F_topK_rotation", meta, file_prefix=file_prefix)
                exps["F_topK_rotation"] = {"summary": ftk_s}

        # F-dynamic rotation: build the temporal basis from passive tracking, then rotate
        if run_F_dynamic_rot and tracking_results is not None:
            try:
                f_dynamic_basis, f_dynamic_info = compute_F_dynamic_temporal_basis(tracking_results, k_min=2, k_max=64)
                all_res["f_dynamic_info"] = f_dynamic_info
                if verbose:
                    print(f"\n  F-dynamic rotation (rank={f_dynamic_info['k']})...")
                fdr_r, fdr_s = run_F_subspace_rotation_experiment(
                    model, tokenizer, rows, bases, baselines, f_dynamic_basis, scr_layer,
                    condition_name="F_dynamic_rotation", n_generate=n_gen, seed=seed, model_id=model_id, verbose=verbose)
                meta = {**f_dyn_metadata, "analysis_type": "F_dynamic_rotation", "f_dynamic_rank": f_dynamic_info['k']}
                all_res["file_paths"]["F_dynamic_rotation"] = save_specB2_results(
                    fdr_r, fdr_s, output_dir, model_key, "F_dynamic_rotation", meta, file_prefix=file_prefix)
                exps["F_dynamic_rotation"] = {"summary": fdr_s}
            except ValueError as e:
                if verbose:
                    print(f"    F-dynamic: skipped ({e})")

        # Step-offset F injection
        if run_step_offset:
            if verbose:
                print(f"\n  Running step-offset F injection (k={f_dyn_k_values})...")
            offset_results, offset_summary = run_step_offset_F_injection(
                model, tokenizer, rows, bases, baselines, scr_layer,
                k_values=f_dyn_k_values, n_generate=f_dyn_n_gen_offset, seed=seed, model_id=model_id, verbose=verbose)
            meta = {**f_dyn_metadata, "k_values": f_dyn_k_values, "n_generate": f_dyn_n_gen_offset}
            all_res["file_paths"]["step_offset_F_injection"] = save_specB2_results(
                offset_results, offset_summary, output_dir, model_key, "step_offset_F_injection", meta, file_prefix=file_prefix)
            exps["step_offset_F_injection"] = {"summary": offset_summary}

    # === EARLY-LAYER F INTERVENTIONS (Part G depth comparison) ===
    run_F_early = any(options.get(f"run_F_early_{t}", False) for t in ["attenuate", "mix", "transplant"])

    if run_F_early:
        # Resolve early layer
        f_early_frac = config.get("f_early_layer_fraction", "match_geometry")
        if f_early_frac == "match_geometry":
            # Use the same "early" layer logic as Part G
            f_early_layer = max(1, int(n_layers * 0.125))  # ~12.5% depth = layer 4 for 32-layer model
        else:
            f_early_layer = int(n_layers * float(f_early_frac))
        f_early_depth_pct = int(100 * f_early_layer / n_layers)

        if verbose:
            print(f"\n  === F EARLY INTERVENTIONS (layer {f_early_layer}/{n_layers}, {f_early_depth_pct}% depth) [{_elapsed()}] ===")

        # Compute bases and baselines at the early layer
        if verbose: print(f"  Computing early-layer bases at layer {f_early_layer}...")
        early_bases = compute_specB2_bases(model, tokenizer, rows, f_early_layer, model_id, config, verbose=False)

        # Reuse generated text from main baselines — only re-extract hidden states
        # and decomposition at the early layer. Generated text is layer-independent
        # (unmodified model), so we save one generation forward pass per prompt.
        if verbose: print(f"  Re-extracting hidden states at early layer (reusing generated text)...")
        early_baselines = rebaseline_at_layer(
            model, tokenizer, rows, early_bases, baselines,
            f_early_layer, model_id, verbose=verbose
        )

        early_metadata = {
            "scramble_layer": f_early_layer,
            "depth_pct": f_early_depth_pct,
            "n_layers": n_layers,
            "scramble_type": scramble_type,
            "experiment_type": "F_early_intervention",
            "rationale": "Part G depth comparison — tests output coherence under early F disruption",
        }

        depth_tag = f"d{f_early_depth_pct}"

        if options.get("run_F_early_attenuate", False):
            if verbose: print(f"  Running F early attenuate (alpha={F_attenuate_alpha})...")
            res, sm = run_F_attenuate_experiment(
                model, tokenizer, rows, early_bases, early_baselines,
                f_early_layer, n_gen, seed, model_id, alpha=F_attenuate_alpha, verbose=verbose
            )
            meta = {**early_metadata, "F_alpha": F_attenuate_alpha}
            fpath = save_specB2_results(res, sm, output_dir, model_key, f"F_early_attenuate", meta, file_prefix=file_prefix)
            all_res["file_paths"]["F_early_attenuate"] = fpath
            exps["F_early_attenuate"] = {"summary": sm}

        if options.get("run_F_early_mix", False):
            if verbose: print(f"  Running F early mix...")
            res, sm = run_F_mix_experiment(
                model, tokenizer, rows, early_bases, early_baselines,
                f_early_layer, n_gen, seed, model_id, verbose=verbose
            )
            meta = {**early_metadata, "F_mix_seed": seed}
            fpath = save_specB2_results(res, sm, output_dir, model_key, f"F_early_mix", meta, file_prefix=file_prefix)
            all_res["file_paths"]["F_early_mix"] = fpath
            exps["F_early_mix"] = {"summary": sm}

        if options.get("run_F_early_transplant", False):
            if verbose: print(f"  Running F early transplant (policy={pair_policy})...")
            res, sm = run_F_transplant_experiment(
                model, tokenizer, rows, early_bases, early_baselines,
                f_early_layer, n_gen, seed, model_id,
                pair_policy=pair_policy, attenuate_alpha_control=attenuate_alpha_control, verbose=verbose
            )
            meta = {**early_metadata, "pair_policy": pair_policy}
            fpath = save_specB2_results(res, sm, output_dir, model_key, f"F_early_transplant", meta, file_prefix=file_prefix)
            all_res["file_paths"]["F_early_transplant"] = fpath
            exps["F_early_transplant"] = {"summary": sm}

        if verbose: print(f"  ✓ F early interventions complete")
    
    # P and D scrambles
    for et in ["P", "D"]:
        if options.get(f"run_{et}_scramble", True):
            res, sm = run_specB2_experiment(model, tokenizer, rows, bases, baselines, scr_layer, et, n_gen, seed, model_id, verbose)
            exp_metadata = {**base_metadata}
            all_res["file_paths"][f"{et}_scramble"] = save_specB2_results(res, sm, output_dir, model_key, f"{et}_scramble", exp_metadata, file_prefix=file_prefix)
            exps[f"{et}_scramble"] = {"summary": sm}
    
    # S scramble — always scrambles S (Stylistic = top-k PCA dims)
    if options.get("run_S_scramble", True):
        if verbose: print(f"\n  Running S-scramble (Stylistic)...")
        res, sm = run_specB2_experiment(model, tokenizer, rows, bases, baselines, scr_layer, "S", n_gen, seed, model_id, verbose)
        exp_metadata = {**base_metadata}
        all_res["file_paths"]["S_scramble"] = save_specB2_results(res, sm, output_dir, model_key, "S_scramble", exp_metadata, file_prefix=file_prefix)
        exps["S_scramble"] = {"summary": sm}
    
    # Random Control
    if options.get("run_random_control", False):
        # Use actual D rank from computed bases for fair comparison
        # This handles the case where d_rank="adaptive"
        actual_d_rank = bases.D_basis.shape[1] if bases.D_basis.size > 0 else 8
        random_rank = config.get("random_rank")
        if random_rank is None or random_rank == "adaptive":
            random_rank = actual_d_rank  # Match actual D rank
        if verbose:
            print(f"\n  Random control using rank={random_rank} (D actual rank={actual_d_rank})")
        res_r, sm_r = run_random_control_experiment(model, tokenizer, rows, baselines, scr_layer, random_rank, n_gen, seed, model_id, verbose)
        exp_metadata = {**base_metadata, "random_rank": random_rank, "d_actual_rank": actual_d_rank}
        all_res["file_paths"]["random_control"] = save_specB2_results(res_r, sm_r, output_dir, model_key, "random_control", exp_metadata, file_prefix=file_prefix)
        exps["random_control"] = {"summary": sm_r}
    
    all_res["experiments"] = exps
    if verbose: print("\n" + "="*70 + f"\n✓ CONTINUATION SCRAMBLE COMPLETE [{file_prefix}] [{_elapsed()}]\n" + "="*70)
    return all_res

# === RANDOM CONTROL FUNCTIONS ===

def build_random_basis(hidden_dim: int, rank: int, seed: int = 42) -> np.ndarray:
    """
    Build a random orthonormal basis of specified rank.
    
    This serves as a control for P/D/S scramble experiments. If scrambling
    random directions causes similar effects to scrambling P/D/S, the effects
    are non-specific (just perturbation). If P/D/S scrambles cause different
    patterns, the subspace identity matters.
    
    Args:
        hidden_dim: Dimension of the hidden state space
        rank: Number of basis vectors to generate
        seed: Random seed for reproducibility
    
    Returns:
        Orthonormal basis of shape (hidden_dim, rank)
    """
    rng = np.random.RandomState(seed)
    R = rng.randn(hidden_dim, rank).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    return Q.astype(np.float32)


def scramble_random_component(h: np.ndarray, random_basis: np.ndarray, seed: int) -> np.ndarray:
    """
    Scramble the component of h in a random subspace.
    
    This is the control version of scramble_P/D/S_component. We project onto
    a random orthonormal basis and scramble the coefficients.
    
    Uses the global SCRAMBLE_METHOD ("rotation" or "permutation").
    
    Args:
        h: Hidden state vector of shape (hidden_dim,)
        random_basis: Random orthonormal basis of shape (hidden_dim, rank)
        seed: Random seed for scrambling
    
    Returns:
        Hidden state with random component scrambled
    """
    if random_basis.size == 0 or random_basis.shape[1] == 0:
        return h.copy()
    
    # Use unified scramble method
    return scramble_coefficients(h, random_basis, seed)


def generate_with_random_scramble(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    scramble_layer: int,
    random_basis: np.ndarray,
    n_generate: int,
    seed: int,
    model_id: str = "",
) -> Dict[str, Any]:
    """
    Generate text with random subspace scrambled at a specific layer.
    
    This is the control experiment for P/D/S scrambles. If random scramble
    causes similar divergence patterns to P/D/S scrambles, the effects are
    due to generic perturbation. If patterns differ, subspace identity matters.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt
        scramble_layer: Layer index for intervention
        random_basis: Random orthonormal basis
        n_generate: Number of tokens to generate
        seed: Random seed
        model_id: Model identifier for chat template
    
    Returns:
        Dict with generated_ids, generated_text, hook_fired
    """
    device = get_model_input_device(model)
    formatted = apply_chat_template(model_id, prompt) if model_id else prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    
    R_np = random_basis if isinstance(random_basis, np.ndarray) else random_basis.cpu().numpy()
    hook_fired = {"value": False}
    
    def hook_fn(module, inputs, output):
        hook_fired["value"] = True
        
        if isinstance(output, tuple):
            h_tensor = output[0]
            rest = output[1:]
        else:
            h_tensor = output
            rest = None
        
        h_last = h_tensor[:, -1, :]
        h_np = h_last[0].detach().cpu().float().numpy()
        h_scrambled_np = scramble_random_component(h_np, R_np, seed)
        h_scrambled = torch.tensor(h_scrambled_np, device=h_last.device, dtype=h_last.dtype)
        
        h_new = h_tensor.clone()
        h_new[:, -1, :] = h_scrambled
        
        if rest is not None:
            return (h_new,) + rest
        return h_new
    
    model_layers = get_model_layers(model)
    block_idx = scramble_layer - 1
    
    if block_idx < 0 or block_idx >= len(model_layers):
        raise IndexError(f"scramble_layer={scramble_layer} out of range")
    
    handle = model_layers[block_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=n_generate,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    
    generated_ids = gen_outputs[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {"generated_ids": generated_ids, "generated_text": generated_text, "hook_fired": hook_fired["value"]}


def run_random_control_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: List[PromptRow],
    baselines: Dict[str, BaselineResult],
    scramble_layer: int,
    random_rank: int,
    n_generate: int,
    seed: int,
    model_id: str,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run Random Control experiment as null hypothesis test.
    
    This tests whether scramble effects are due to subspace identity (P/D/S
    encode specific information) or just generic perturbation (any scramble
    causes divergence).
    
    Expected outcome:
    - If Random ≈ D-scramble: dimension count drives effect, not identity
    - If Random ≠ D-scramble: subspace identity matters (supports P/D/S hypothesis)
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        rows: List of PromptRow objects
        baselines: Dict of baseline results
        scramble_layer: Layer index for intervention
        random_rank: Rank of random basis (match to D rank for fair comparison)
        n_generate: Number of tokens to generate
        seed: Random seed
        model_id: Model identifier
        verbose: If True, show progress
    
    Returns:
        results: List of per-prompt result dicts
        summary: Summary statistics dict
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING RANDOM CONTROL (rank={random_rank})")
        print(f"{'='*60}")
        print("Hypothesis: If Random ≈ D-scramble, dimension count drives effect")
        print("            If Random ≠ D-scramble, subspace identity matters")
    
    # Build random basis
    # Get hidden_dim from first baseline
    first_baseline = next(iter(baselines.values()))
    if isinstance(first_baseline, BaselineResult):
        hidden_dim = first_baseline.h_layer.shape[0]
    else:
        # Estimate from model config
        hidden_dim = model.config.hidden_size
    
    random_basis = build_random_basis(hidden_dim, random_rank, seed)
    if verbose:
        print(f"Random basis shape: {random_basis.shape}")
    
    results = []
    
    if verbose:
        try:
            from tqdm import tqdm
            row_iter = tqdm(rows, desc="Random scramble")
        except ImportError:
            row_iter = rows
    else:
        row_iter = rows
    
    for i, row in enumerate(row_iter):
        key = f"{row.group_id}_{row.variant_id}"
        baseline = baselines.get(key)
        
        if baseline is None:
            continue
        
        # Generate with random scramble
        scrambled = generate_with_random_scramble(
            model, tokenizer, row.prompt, scramble_layer,
            random_basis, n_generate, seed + i, model_id
        )
        
        # Compare to baseline
        baseline_dict = {
            "generated_ids": baseline.generated_ids if isinstance(baseline, BaselineResult) else baseline.get("generated_ids", []),
            "generated_text": baseline.generated_text if isinstance(baseline, BaselineResult) else baseline.get("generated_text", ""),
        }
        effect = compare_generations(baseline_dict, scrambled)
        
        results.append({
            "group_id": row.group_id,
            "variant_id": row.variant_id,
            "prompt": row.prompt[:100] + "..." if len(row.prompt) > 100 else row.prompt,
            "baseline_text": baseline_dict["generated_text"],
            "scrambled_text": scrambled["generated_text"],
            "effect": effect,
            "hook_fired": scrambled.get("hook_fired", False),
        })
    
    summary = compute_divergence_summary(results)
    
    if verbose:
        print(f"\nSummary: {summary['n_prompts']} prompts, {summary['identical_pct']:.1f}% identical, {summary['immediate_divergence_pct']:.1f}% immediate")
        if summary['mean_divergence_token'] is not None:
            print(f"Mean divergence token: {summary['mean_divergence_token']:.2f}")
    
    return results, summary


# === HELPERS ===

def load_prompts_from_json(filepath: Path, tier: str = "standard") -> List[PromptRow]:
    """Load prompts from JSON, supporting both old (nested groups) and new (flat list) formats.
    
    Old format: data["groups"][].group_id, data["groups"][].variants[].variant_id, .prompt
    New format: data["prompts"][].group_id, .variant_id, .category, .regime, .prompt
    
    For old format, tier filtering is applied (lite/standard/full).
    For new format, all prompts are returned (no tier filtering).
    """
    with open(filepath) as f: data = json.load(f)
    rows = []
    
    # New format: flat list with 'prompts' key
    if "prompts" in data and isinstance(data["prompts"], list):
        for p in data["prompts"]:
            rows.append(PromptRow(
                group_id=p["group_id"],
                variant_id=p["variant_id"],
                prompt=p["prompt"],
                category=p.get("category", ""),
                regime=p.get("regime", ""),
            ))
        return rows
    
    # Old format: nested groups/variants with tier filtering
    tiers = data.get("tiers", {
        "lite": [f"group_{i:02d}" for i in range(1,7)],
        "standard": [f"group_{i:02d}" for i in range(1,11)],
        "full": [f"group_{i:02d}" for i in range(1,13)]
    })
    incl = set(tiers.get(tier, []))
    for g in data.get("groups", []):
        if g["group_id"] not in incl: continue
        for v in g["variants"]:
            rows.append(PromptRow(g["group_id"], v["variant_id"], v["prompt"], g.get("category", "")))
    return rows

def get_specB2_output_dir(base_dir: str, model_key: str) -> Path:
    return Path(base_dir) / "SpecB2" / model_key




# ============================================================
# F-topK / F-dynamic rotation + step-offset F injection (ported from V11.8 Paper 2, 2026-07-02)
# Feeds Paper 1 §6.4/Fig8, §7.3/Fig9, §7.2/Table5. Reuses existing release helpers
# (decompose_hidden_state, pca_basis[2-tuple], project_*_basis, get_model_layers, etc.).
# ============================================================

@dataclass
class FTrackingTelemetry:
    """Per-token F state captured during generation.
    
    Used by both passive tracking (read-only, modify_fn=identity) and active
    interventions (step-offset injection, etc.). Records the model's naturally-
    computed F *before* any modification is applied.
    
    HISTORY: Renamed from PersistentHookTelemetry in v4.0 (V11.6). The
    persistent hook experiments that motivated the original dataclass have
    been retired; this telemetry structure now serves the replacement
    F-dynamics experiments (passive tracking, step-offset injection).
    
    Fields:
        pre_hook_F_norms: ||F_natural|| at each generation step.
        pre_hook_F_cosine_vs_T0: cos(F_natural, F_T0) at each step.
            Tracks how far the model's F has drifted from its initial state.
        post_hook_F_norms: ||F_modified|| after the hook's modify_fn.
            For passive tracking (identity modify_fn), equals pre_hook_F_norms.
        pre_hook_F_vectors: Full F_natural vectors at each step (opt-in).
            Needed for PCA static/dynamic decomposition (Experiment C).
            Set to None by default to avoid d_model × n_generate memory
            per prompt when only scalar telemetry is needed.
        n_fires: Number of times the hook fired.
    """
    pre_hook_F_norms: List[float] = field(default_factory=list)
    pre_hook_F_cosine_vs_T0: List[float] = field(default_factory=list)
    post_hook_F_norms: List[float] = field(default_factory=list)
    pre_hook_F_vectors: Optional[List[np.ndarray]] = None  # V11.6: opt-in full vectors for PCA
    n_fires: int = 0


def generate_with_every_token_F_hook(
    model, tokenizer, prompt: str,
    scramble_layer: int,
    modify_F_fn,       # Callable: (h_F_natural: np.ndarray, dc: PDSDecomposition, token_idx: int) -> np.ndarray
    F_T0: np.ndarray,  # Baseline F for telemetry cosine comparison
    P_basis: np.ndarray, D_basis: np.ndarray, S_basis: np.ndarray,
    n_generate: int, seed: int, model_id: str = "",
    store_vectors: bool = False,
) -> Dict[str, Any]:
    """Generate with an F hook that fires at every forward pass.
    
    General-purpose every-token hook engine. The hook:
      1. Decomposes h at the target layer into PDSF.
      2. Records telemetry (F norms, cosine vs F_T0, optionally full F vectors).
      3. Calls modify_F_fn(h_F_natural, decomposition, token_idx) to get
         the replacement F vector.
      4. Reassembles h = P + D + S + F_modified and writes back.
    
    For passive tracking, pass modify_fn = lambda h, dc, t: h (identity).
    For step-offset injection, pass a closure that returns F from k steps ago.
    
    HISTORY: Renamed from generate_with_persistent_F_hook in v4.0 (V11.6).
    The function logic is unchanged; only the name and docstring were updated
    to reflect its general-purpose nature after the persistent hook experiments
    were retired.
    
    Args:
        modify_F_fn: Callable that receives the model's naturally-computed F,
            the full PDSF decomposition, and the token index (0-based), and
            returns the replacement F vector.
        F_T0: The prompt's baseline F vector at the intervention layer (T0).
            Used only for telemetry (cosine comparison), not for modification.
        store_vectors: If True, store full F vectors in telemetry for downstream
            PCA (Experiment C). Increases memory by n_generate × d_model per prompt.
    
    Returns:
        Dict with generated_ids, generated_text, hook_fired, telemetry.
    """
    device = get_model_input_device(model)
    formatted = apply_chat_template(model_id, prompt) if model_id else prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    P_np = P_basis if isinstance(P_basis, np.ndarray) else P_basis.cpu().numpy()
    D_np = D_basis if isinstance(D_basis, np.ndarray) else D_basis.cpu().numpy()
    S_np = S_basis if isinstance(S_basis, np.ndarray) else S_basis.cpu().numpy()
    
    # Telemetry state
    F_T0_norm = float(np.linalg.norm(F_T0))
    telemetry = FTrackingTelemetry()
    if store_vectors:
        telemetry.pre_hook_F_vectors = []  # Opt-in: allocate list for full F vectors
    token_counter = {"idx": 0}
    
    def hook_fn(module, inp, output):
        ht = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        
        h_np = ht[:, -1, :][0].detach().cpu().float().numpy()
        
        # Decompose to get natural F
        dc = decompose_hidden_state(h_np, P_np, D_np, S_np)
        h_F_natural = dc.h_F
        
        # --- Telemetry: record pre-hook state ---
        nat_norm = float(np.linalg.norm(h_F_natural))
        telemetry.pre_hook_F_norms.append(nat_norm)
        if F_T0_norm > 1e-12 and nat_norm > 1e-12:
            cos_vs_T0 = float(np.dot(h_F_natural, F_T0)) / (nat_norm * F_T0_norm)
            cos_vs_T0 = float(np.clip(cos_vs_T0, -1.0, 1.0))
        else:
            cos_vs_T0 = 0.0
        telemetry.pre_hook_F_cosine_vs_T0.append(cos_vs_T0)
        if telemetry.pre_hook_F_vectors is not None:
            telemetry.pre_hook_F_vectors.append(h_F_natural.copy())
        
        # --- Apply modification ---
        h_F_new = modify_F_fn(h_F_natural, dc, token_counter["idx"])
        h_modified = (dc.h_P + dc.h_D + dc.h_S + h_F_new).astype(np.float32)
        
        # --- Telemetry: record post-hook state ---
        telemetry.post_hook_F_norms.append(float(np.linalg.norm(h_F_new)))
        telemetry.n_fires += 1
        token_counter["idx"] += 1
        
        # Write back
        ht_new = ht.clone()
        ht_new[:, -1, :] = torch.tensor(h_modified, device=ht.device, dtype=ht.dtype)
        return (ht_new,) + rest if rest else ht_new
    
    layers = get_model_layers(model)
    block_idx = scramble_layer - 1
    if block_idx < 0 or block_idx >= len(layers):
        raise IndexError(f"Invalid layer {scramble_layer}")
    
    handle = layers[block_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=n_generate,
                do_sample=False,
                use_cache=False,  # V11.8: disable KV cache to avoid RoPE dim mismatch with hooked hidden states
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    
    gen_ids = gen[0, prompt_len:].tolist()
    return {
        "generated_ids": gen_ids,
        "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "hook_fired": telemetry.n_fires > 0,
        "telemetry": {
            "pre_hook_F_norms": telemetry.pre_hook_F_norms,
            "pre_hook_F_cosine_vs_T0": telemetry.pre_hook_F_cosine_vs_T0,
            "post_hook_F_norms": telemetry.post_hook_F_norms,
            "n_fires": telemetry.n_fires,
            "pre_hook_F_vectors": telemetry.pre_hook_F_vectors,  # None if store_vectors=False
        },
    }


def _get_F_T0(bl, bases: SpecB2Bases) -> np.ndarray:
    """Extract F vector at T0 from a baseline, handling both dataclass and dict.
    
    This helper is used by all F-dynamics experiments to obtain the prompt-boundary
    F vector from cached baselines (which may be BaselineResult instances from
    a fresh run, or plain dicts loaded from JSON).
    
    HISTORY: Originally part of the V11.5 persistent hook section. Retained
    in V11.6 because it is needed by passive tracking and step-offset injection.
    """
    decomp = _bl_get(bl, "decomposition", {})
    h_F = _bl_get(decomp, "h_F", None)
    if h_F is None:
        h_F = _bl_get(decomp, "h_residual", None)
    if h_F is not None:
        return np.asarray(h_F, dtype=np.float32)
    # Recompute from h_layer
    h_layer = _bl_get(bl, "h_layer", None)
    if h_layer is not None:
        h_layer = np.asarray(h_layer, dtype=np.float32)
        bl_fid = _bl_get(bl, "family_id", "")
        Pb = bases.P_bases.get(bl_fid, np.zeros((h_layer.shape[0], 0), dtype=np.float32))
        h_F_vec, _ = _F_from_hidden(h_layer, Pb, bases.D_basis, bases.S_basis)
        return h_F_vec
    raise ValueError("Baseline has no h_F or h_layer — cannot extract F_T0")


def fit_exponential_decay(
    cosine_trajectory: List[float],
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Fit cos(F_t, F_0) ≈ asymptote + (1 - asymptote) · exp(-t / τ).
    
    Models the F decorrelation curve as an exponential decay toward a stable
    asymptote. The asymptote represents the static core of F (the fraction
    of F direction that persists across all generation steps). τ is the
    time constant in tokens — the number of steps for the dynamic component
    to lose (1 - 1/e) ≈ 63% of its initial alignment with F_T0.
    
    Falls back to simple heuristic estimates if scipy is unavailable or
    the optimizer fails to converge.
    
    Args:
        cosine_trajectory: Sequence of cos(F_t, F_0) values, t = 0, 1, ...
            cosine_trajectory[0] should be ~1.0 (F_0 vs F_0).
        max_iter: Maximum iterations for curve_fit.
    
    Returns:
        Dict with keys:
            tau: Time constant (tokens). Larger = slower decorrelation.
            asymptote: Stable cosine floor. ~0.37 in persistent hook data.
            r_squared: Goodness of fit (1.0 = perfect exponential).
    """
    y = np.array(cosine_trajectory, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64)
    
    if len(y) < 3:
        return {"tau": float('nan'), "asymptote": float(y[-1]) if len(y) > 0 else float('nan'), "r_squared": float('nan')}
    
    # Heuristic fallback estimates
    tail_frac = max(1, len(y) // 4)
    asymptote_est = float(np.mean(y[-tail_frac:]))
    # Half-life estimate: find first t where y < (1 + asymptote_est) / 2
    half_target = (1.0 + asymptote_est) / 2.0
    tau_est = 2.0  # default
    for i, val in enumerate(y):
        if val < half_target:
            tau_est = max(0.5, i / np.log(2))
            break
    
    try:
        from scipy.optimize import curve_fit
        
        def model_fn(t, asymptote, tau):
            return asymptote + (1.0 - asymptote) * np.exp(-t / tau)
        
        popt, _ = curve_fit(
            model_fn, t, y,
            p0=[asymptote_est, tau_est],
            bounds=([0.0, 0.1], [1.0, 100.0]),
            maxfev=max_iter,
        )
        a_fit, tau_fit = popt
        
        # R² computation
        y_pred = model_fn(t, a_fit, tau_fit)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        
        return {"tau": float(tau_fit), "asymptote": float(a_fit), "r_squared": float(r2)}
    
    except Exception:
        # scipy unavailable or fit failed — return heuristic estimates
        return {"tau": float(tau_est), "asymptote": float(asymptote_est), "r_squared": float('nan')}


def run_passive_F_tracking(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases,
    baselines: Dict[str, BaselineResult], scramble_layer: int,
    n_generate: int = 64, seed: int = 0, model_id: str = "",
    store_vectors: bool = True, verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Passive F-tracking: read-only hook records F at every generation step.
    
    No intervention is applied — the model generates normally. The hook
    decomposes the hidden state at the target layer and records the F
    component without modifying anything. This gives the *unperturbed*
    F decorrelation rate, which is the clean measurement of F's temporal
    dynamics without the confound of frozen-F feedback.
    
    MOTIVATION:
    The persistent hook's τ ≈ 2.2 is contaminated by feedback: freezing F
    alters the model's output, which alters the pre-hook F at the next step.
    Passive tracking gives the unperturbed decorrelation rate. The difference
    between perturbed and unperturbed τ measures how much F's dynamics depend
    on the model generating coherent output.
    
    PREDICTION: τ_unperturbed > 2.2 tokens (the persistent hook's feedback
    loop accelerates decorrelation). The asymptote may also shift.
    
    Args:
        model, tokenizer: HuggingFace model and tokenizer.
        rows: Prompt data.
        bases: PDS bases at the target layer.
        baselines: Cached baselines (needed for F_T0 reference and comparison).
        scramble_layer: Layer for F extraction.
        n_generate: Tokens to generate (64 is sufficient: if τ ≈ 3-5,
            64 tokens gives ~12-20 time constants for exponential fit).
        seed: Random seed.
        model_id: Model identifier for chat template.
        store_vectors: If True, store full F vectors for downstream PCA
            (Experiment C). Increases memory by n_generate × d_model per prompt.
        verbose: Print progress.
    
    Returns:
        results: Per-prompt dicts with cosine trajectory, norm trajectory,
            exponential fit parameters, and optionally full F vectors.
        summary: Aggregate statistics (mean τ, mean asymptote, etc.).
    """
    from tqdm import tqdm
    results: List[Dict[str, Any]] = []
    row_iter = tqdm(rows, desc="Passive F-tracking") if verbose else rows
    
    for i, row in enumerate(row_iter):
        key = f"{row.group_id}_{row.variant_id}"
        bl = baselines.get(key)
        if bl is None:
            continue
        
        F_T0 = _get_F_T0(bl, bases)
        _hdim = bases.D_basis.shape[0]
        bl_fid = _bl_get(bl, "family_id", "")
        Pb = bases.P_bases.get(bl_fid, np.zeros((_hdim, 0), dtype=np.float32))
        
        # Identity modify_fn: return natural F unchanged (read-only tracking)
        def modify_fn(h_F_natural, dc, token_idx):
            return h_F_natural
        
        scr = generate_with_every_token_F_hook(
            model, tokenizer, row.prompt, scramble_layer,
            modify_fn, F_T0, Pb, bases.D_basis, bases.S_basis,
            n_generate, seed, model_id, store_vectors=store_vectors,
        )
        
        # Extract telemetry
        telem = scr.get("telemetry", {})
        cosine_traj = telem.get("pre_hook_F_cosine_vs_T0", [])
        norm_traj = telem.get("pre_hook_F_norms", [])
        
        # Fit exponential decay to cosine trajectory
        tau_fit = fit_exponential_decay(cosine_traj) if len(cosine_traj) >= 3 else {
            "tau": float('nan'), "asymptote": float('nan'), "r_squared": float('nan')
        }
        
        result = {
            "prompt_key": key, "group_id": row.group_id, "variant_id": row.variant_id,
            "category": row.category, "regime": getattr(row, 'regime', ''),
            "condition": "passive_F_tracking",
            "n_generate": n_generate,
            "cosine_trajectory": cosine_traj,
            "norm_trajectory": norm_traj,
            "tau_fit": tau_fit,
            "F_T0_norm": float(np.linalg.norm(F_T0)),
            "n_fires": telem.get("n_fires", 0),
            # Generated text for sanity check (should match baseline exactly
            # since no intervention is applied)
            "generated_text": scr.get("generated_text", ""),
            "matches_baseline": (scr.get("generated_ids", []) == 
                                 (_bl_get(bl, "generated_ids", []))),
        }
        
        # Optionally store full F vectors (large — d_model floats per step)
        if store_vectors and telem.get("pre_hook_F_vectors") is not None:
            # Store as list-of-lists for JSON serialization
            result["F_vectors_shape"] = [len(telem["pre_hook_F_vectors"]),
                                         telem["pre_hook_F_vectors"][0].shape[0] 
                                         if telem["pre_hook_F_vectors"] else 0]
            # Don't serialize full vectors to JSON — too large.
            # Instead, store a reference flag; vectors are in memory for
            # immediate PCA (Experiment C) during the same pipeline run.
            result["_F_vectors_available"] = True
            result["_F_vectors"] = telem["pre_hook_F_vectors"]  # in-memory only
        
        results.append(result)
    
    # Summary statistics
    taus = [r["tau_fit"]["tau"] for r in results if np.isfinite(r["tau_fit"]["tau"])]
    asymptotes = [r["tau_fit"]["asymptote"] for r in results if np.isfinite(r["tau_fit"]["asymptote"])]
    r2s = [r["tau_fit"]["r_squared"] for r in results if np.isfinite(r["tau_fit"]["r_squared"])]
    baseline_matches = sum(1 for r in results if r.get("matches_baseline", False))
    
    summary = {
        "condition": "passive_F_tracking",
        "n_prompts": len(results),
        "n_generate": n_generate,
        "store_vectors": store_vectors,
        "baseline_match_count": baseline_matches,
        "baseline_match_pct": 100 * baseline_matches / max(1, len(results)),
    }
    if taus:
        summary["tau_mean"] = float(np.mean(taus))
        summary["tau_std"] = float(np.std(taus))
        summary["tau_median"] = float(np.median(taus))
    if asymptotes:
        summary["asymptote_mean"] = float(np.mean(asymptotes))
        summary["asymptote_std"] = float(np.std(asymptotes))
    if r2s:
        summary["r_squared_mean"] = float(np.mean(r2s))
    
    if verbose:
        print(f"\n  Passive F-tracking: {len(results)} prompts, "
              f"τ = {summary.get('tau_mean', float('nan')):.2f} ± {summary.get('tau_std', float('nan')):.2f}, "
              f"asymptote = {summary.get('asymptote_mean', float('nan')):.3f}")
        # Not expected to be 100%. The hook decomposes the hidden state to P+D+S+F
        # in float32 and sums it back, which is not bitwise identical to the original
        # bf16/fp16 state, so greedy decoding flips a fraction of prompts. The
        # published runs span 0-100% on this statistic depending on the model. The
        # meaningful read-only check is the step-offset k=0 control, where no prompt
        # should diverge within the generation window.
        print(f"  Baseline exact-match rate: {summary['baseline_match_pct']:.1f}% "
              f"(model-dependent; exact match is precision-sensitive, see k=0 control)")
    
    return results, summary


def compute_F_topK_basis(
    H: np.ndarray,
    P_bases: Dict[int, np.ndarray],
    D_basis: np.ndarray,
    S_basis: np.ndarray,
    k_F: Union[int, str] = "adaptive",
    k_min: int = 2,
    k_max: int = 64,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute F-topK basis: top PCA of the P-D-S residual across prompts.

    Fourth step in the sequential orthogonal decomposition:
    P -> D -> S -> F-topK -> F-residual. Methodologically identical to
    compute_D_basis_global and compute_S_basis_global: project out prior
    subspaces, center, SVD, PR-adaptive rank.

    F-topK captures directions of maximal cross-prompt variance within
    the frame subspace.

    Args:
        H: Hidden states (n_prompts, d_model) at a single layer
        P_bases: Dict[int, ndarray] -- per-prompt P basis
        D_basis: D basis (d_model, k_D)
        S_basis: S basis (d_model, k_S)
        k_F: Rank selection ("adaptive" or int)
        k_min, k_max: Bounds for PR-adaptive rank

    Returns:
        (F_topK_basis, info) where basis is (d_model, k) and info
        contains k, pr, n, singular_values
    """
    n, d = H.shape

    # Sequential projection: remove P, D, S to get F
    F_vecs = np.zeros_like(H)
    for i in range(n):
        h = H[i]
        Pb = P_bases.get(i)
        hp = project_out_basis(h, Pb) if Pb is not None and Pb.size > 0 else h.copy()
        hpd = project_out_basis(hp, D_basis) if D_basis is not None and D_basis.size > 0 else hp
        F_vecs[i] = project_out_basis(hpd, S_basis) if S_basis is not None and S_basis.size > 0 else hpd

    # PR-adaptive rank selection (same logic as D and S)
    pr = participation_ratio(F_vecs, center=True)
    if not np.isfinite(pr) or pr < 1.0:
        F_centered = F_vecs - F_vecs.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(F_centered, compute_uv=False)
            eigs = s ** 2
            pr = (eigs.sum() ** 2) / (eigs ** 2).sum() if (eigs ** 2).sum() > 1e-12 else float(k_min)
        except Exception:
            pr = float(k_min)

    if k_F == "adaptive":
        k = max(k_min, min(k_max, round(pr), n - 1))
    else:
        k = max(1, min(int(k_F), k_max, n - 1))

    basis, svs = pca_basis(F_vecs, k=k, center=True)

    return basis, {"k": int(k), "pr": float(pr), "n": n,
        "singular_values": svs.tolist() if hasattr(svs, 'tolist') else list(svs)}


def compute_F_dynamic_temporal_basis(
    tracking_results: List[Dict[str, Any]],
    k_min: int = 2,
    k_max: int = 64,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute F-dynamic-temporal basis from passive tracking vectors.

    For each prompt, the passive tracking stored F vectors at every
    generation step. This function:
      1. Stacks each prompt's F trajectory into (T, d_model)
      2. Centers temporally (removes temporal mean)
      3. Computes temporal covariance
      4. Averages covariance across prompts
      5. Eigendecomposes with PR-adaptive rank

    The resulting basis captures directions along which F changes most
    during generation -- the "active computational workspace" within F.

    Args:
        tracking_results: Output of run_passive_F_tracking with
            store_vectors=True. Each entry must have _F_vectors.
        k_min, k_max: Bounds for PR-adaptive rank

    Returns:
        (basis, info) where basis is (d_model, k_dynamic) and info
        contains k, pr, n_prompts
    """
    cov_sum = None
    n_valid = 0

    for r in tracking_results:
        vecs = r.get("_F_vectors")
        if vecs is None or len(vecs) < 3:
            continue
        F_mat = np.array(vecs, dtype=np.float64)
        F_centered = F_mat - F_mat.mean(axis=0, keepdims=True)
        cov = (F_centered.T @ F_centered) / (F_mat.shape[0] - 1)
        if cov_sum is None:
            cov_sum = cov
        else:
            cov_sum += cov
        n_valid += 1

    if n_valid == 0:
        raise ValueError("No tracking results with stored F vectors found. "
                         "Run passive_F_tracking with store_vectors=True.")

    cov_avg = cov_sum / n_valid
    eigvals, eigvecs = np.linalg.eigh(cov_avg)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    # PR-adaptive rank
    pr = float((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-12))
    k = max(k_min, min(k_max, round(pr)))

    basis = eigvecs[:, :k].astype(np.float32)

    return basis, {"k": int(k), "pr": float(pr), "n_prompts": n_valid}


def make_F_subspace_rotation_fn(
    basis: np.ndarray,
    seed: int = 42,
):
    """Create a modify_F_fn that rotates F within a sub-subspace.

    The returned function is suitable for generate_with_every_token_F_hook.
    It projects F onto the given basis, applies a random orthogonal rotation
    to the coefficients, and reconstructs. Components of F outside the basis
    are preserved unchanged.

    Same operation as D/S rotation (scramble_coefficients_rotation), but
    applied to an arbitrary sub-subspace of F rather than the full D or S.

    Args:
        basis: Orthonormal basis (d_model, k) defining the F sub-subspace
        seed: Random seed for the orthogonal matrix

    Returns:
        Callable(h_F_natural, dc, token_idx) -> h_F_modified
    """
    k = basis.shape[1]
    rng = np.random.RandomState(seed)
    R, _ = np.linalg.qr(rng.randn(k, k).astype(np.float32))

    def modify_fn(h_F_natural, dc, token_idx):
        coeffs = h_F_natural @ basis          # (k,) project onto sub-subspace
        coeffs_rot = R @ coeffs               # (k,) rotate within sub-subspace
        h_in_basis = coeffs @ basis.T          # back to model space (original)
        h_rot_basis = coeffs_rot @ basis.T     # back to model space (rotated)
        return h_F_natural - h_in_basis + h_rot_basis  # swap in rotated, keep rest

    return modify_fn


def run_F_subspace_rotation_experiment(
    model, tokenizer,
    rows: List[PromptRow],
    bases: SpecB2Bases,
    baselines: Dict[str, BaselineResult],
    f_sub_basis: np.ndarray,
    scramble_layer: int,
    condition_name: str = "F_topK_rotation",
    n_generate: int = 64,
    seed: int = 42,
    model_id: str = "",
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run rotational intervention on an F sub-subspace during generation.

    Uses generate_with_every_token_F_hook with a modify_fn that rotates
    F within the given sub-subspace at every generation step. Same
    intervention mechanics as D-rotation and S-rotation experiments.

    The predicted intervention hierarchy (Paper 2, Section 5.4):
      D-rotation > F-dynamic-rotation > F-topK-rotation > random
    where > means "more disruptive".

    Args:
        model, tokenizer: HuggingFace model
        rows: Prompt data (SpecB)
        bases: PDSF bases at intervention layer (SpecB2Bases)
        baselines: Cached baseline generations
        f_sub_basis: (d_model, k) orthonormal basis of the F sub-subspace
        scramble_layer: Intervention layer
        condition_name: Label for this condition in output
        n_generate: Tokens to generate (64 = standard for continuation exps)
        seed: Random seed
        model_id: For chat template formatting

    Returns:
        (results, summary) -- per-prompt results and aggregate divergence stats
    """
    from tqdm import tqdm

    modify_fn = make_F_subspace_rotation_fn(f_sub_basis, seed=seed)
    results = []
    row_iter = tqdm(rows, desc=f"  {condition_name}") if verbose else rows

    for i, row in enumerate(row_iter):
        key = f"{row.group_id}_{row.variant_id}"
        bl = baselines.get(key)
        if bl is None:
            continue

        F_T0 = _get_F_T0(bl, bases)
        _hdim = bases.D_basis.shape[0]
        bl_fid = _bl_get(bl, "family_id", "")
        Pb = bases.P_bases.get(bl_fid, np.zeros((_hdim, 0), dtype=np.float32))

        scr = generate_with_every_token_F_hook(
            model, tokenizer, row.prompt, scramble_layer,
            modify_fn, F_T0, Pb, bases.D_basis, bases.S_basis,
            n_generate, seed + i, model_id, store_vectors=False,
        )

        # Compare to baseline
        bl_ids = _bl_get(bl, "generated_ids", [])
        gen_ids = scr.get("generated_ids", [])
        comp = compare_generations(
            {"generated_ids": bl_ids},
            {"generated_ids": gen_ids},
        )

        results.append({
            "prompt_key": key,
            "group_id": row.group_id,
            "variant_id": row.variant_id,
            "category": getattr(row, "category", ""),
            "regime": getattr(row, "regime", ""),
            "condition": condition_name,
            "f_sub_rank": f_sub_basis.shape[1],
            **comp,
            "generated_text": scr.get("generated_text", ""),
            "baseline_text": _bl_get(bl, "generated_text", ""),
        })

    # Aggregate summary
    summary = compute_divergence_summary(results)
    summary["condition"] = condition_name
    summary["f_sub_rank"] = f_sub_basis.shape[1]

    if verbose:
        # compute_divergence_summary returns None for these when nothing diverged,
        # so .get(key, default) is not enough — the key exists and holds None, and
        # formatting None raises. This print used to kill the whole phase after the
        # experiment had already finished computing.
        def _num(key, default=float("nan")):
            v = summary.get(key, default)
            return default if v is None else v
        fd = _num("mean_divergence_token")
        idr = _num("immediate_divergence_pct", 0.0)
        ir = _num("identical_pct", 0.0)
        print(f"\n  {condition_name} (rank={f_sub_basis.shape[1]}): "
              f"first_div={fd:.1f}, imm_div={idr:.1f}%, "
              f"identical={ir:.1f}%")

    return results, summary


def run_step_offset_F_injection(
    model, tokenizer, rows: List[PromptRow], bases: SpecB2Bases,
    baselines: Dict[str, BaselineResult], scramble_layer: int,
    k_values: List[int] = None,
    n_generate: int = 32, seed: int = 0, model_id: str = "",
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Step-offset F injection: at each generation step, inject F from t-k steps ago.
    
    k=0 is the baseline control (identity — validates hook overhead is negligible).
    k=1 injects the previous step's F — a minimal lag.
    k→∞ is the persistent freeze (retired experiment).
    
    The transition from coherent to degenerate output as k increases directly
    measures the timescale of F's functional relevance. This is the most
    informative single experiment in the F-dynamics suite — it yields a full
    dose-response curve with a continuous control parameter.
    
    MOTIVATION:
    The step-offset naturally decomposes F into:
      - Static component: invariant to lag (F directions that don't change
        between steps remain correct even with offset)
      - Dynamic component: sensitive to lag (F directions that change rapidly
        become stale and cause divergence)
    
    The hook maintains a growing list of F vectors from all previous steps.
    At step t, it returns the F vector from step max(0, t-k). For steps
    where t < k, it returns F_T0 (the prompt boundary F).
    
    Args:
        model, tokenizer: HuggingFace model and tokenizer.
        rows: Prompt data.
        bases: PDS bases at the target layer.
        baselines: Cached baselines.
        scramble_layer: Layer for intervention.
        k_values: List of lag values to test. Default [0, 1, 2, 3, 5].
            k=0 is the identity control (should produce 100% identical rate).
        n_generate: Tokens to generate per condition. 32 is sufficient given
            τ ≈ 2-5 tokens — output will degenerate within ~10 tokens for
            large k. This is 4× cheaper than the persistent hook's 128 tokens.
        seed: Random seed.
        model_id: Model identifier.
        verbose: Print progress.
    
    Returns:
        results: Per-prompt × per-k dicts with generated text, divergence
            metrics, and F cosine trajectory.
        summary: Per-k aggregate divergence statistics.
    """
    if k_values is None:
        k_values = [0, 1, 2, 3, 5]
    
    from tqdm import tqdm
    results: List[Dict[str, Any]] = []
    
    for k in k_values:
        if verbose:
            print(f"\n  Step-offset injection k={k} ({'identity control' if k == 0 else f'{k}-step lag'})...")
        
        row_iter = tqdm(rows, desc=f"k={k}") if verbose else rows
        k_results: List[Dict[str, Any]] = []
        
        for i, row in enumerate(row_iter):
            key = f"{row.group_id}_{row.variant_id}"
            bl = baselines.get(key)
            if bl is None:
                continue
            
            F_T0 = _get_F_T0(bl, bases)
            _hdim = bases.D_basis.shape[0]
            bl_fid = _bl_get(bl, "family_id", "")
            Pb = bases.P_bases.get(bl_fid, np.zeros((_hdim, 0), dtype=np.float32))
            
            # Build the offset modify_fn with a growing buffer
            # buffer[0] = F_T0 (prompt boundary)
            # buffer[t+1] = F_natural at step t
            # At step t, we return buffer[max(0, t+1-k)] (the F from k steps ago)
            _buffer = [F_T0.copy()]
            _k = k
            
            def modify_fn(h_F_natural, dc, token_idx, _buf=_buffer, _lag=_k):
                _buf.append(h_F_natural.copy())
                if _lag == 0:
                    return h_F_natural  # Identity control: no modification
                lag_idx = max(0, len(_buf) - 1 - _lag)
                return _buf[lag_idx].copy()
            
            scr = generate_with_every_token_F_hook(
                model, tokenizer, row.prompt, scramble_layer,
                modify_fn, F_T0, Pb, bases.D_basis, bases.S_basis,
                n_generate, seed, model_id, store_vectors=False,
            )
            
            # Compare to baseline
            ef = compare_generations(
                {"generated_ids": _bl_get(bl, "generated_ids", []),
                 "generated_text": _bl_get(bl, "generated_text", "")}, scr)
            
            telem = scr.get("telemetry", {})
            
            k_results.append({
                "prompt_key": key, "group_id": row.group_id, "variant_id": row.variant_id,
                "category": row.category, "regime": getattr(row, 'regime', ''),
                "condition": f"step_offset_k{k}",
                "k": k,
                "n_generate": n_generate,
                "baseline_text": _bl_get(bl, "generated_text", ""),
                "scrambled_text": scr.get("generated_text"),
                "effect": ef,
                "hook_fired": bool(scr.get("hook_fired")),
                "cosine_trajectory": telem.get("pre_hook_F_cosine_vs_T0", []),
                "F_T0_norm": float(np.linalg.norm(F_T0)),
            })
        
        results.extend(k_results)
        
        # Per-k summary
        sm = compute_divergence_summary(k_results)
        if verbose:
            ident_pct = sm.get('identical_pct', 0)
            mean_div = sm.get('mean_divergence_token')
            mean_div_str = f"{mean_div:.1f}" if mean_div is not None else "N/A"
            print(f"    k={k}: {sm['n_prompts']} prompts, "
                  f"{ident_pct:.1f}% identical, "
                  f"mean divergence token = {mean_div_str}")
    
    # Aggregate summary: one row per k value
    summary = {
        "condition": "step_offset_F_injection",
        "k_values": k_values,
        "n_generate": n_generate,
        "per_k": {},
    }
    for k in k_values:
        k_res = [r for r in results if r["k"] == k]
        sm = compute_divergence_summary(k_res)
        summary["per_k"][str(k)] = sm
    
    return results, summary

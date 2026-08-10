"""pds_geometry.py

Model management, hidden state extraction, and Part G causal intervention experiments.

Part of: PDSF — Prediction-Anchored Decomposition into Functional Subspaces

Paper:   "Geometric and Behavioral Stratification in Transformer Residual Streams"
         Nelson Guda, 2026
Repo:    https://github.com/nelsonguda/pdsf-residual-geometry

License: MIT (code), CC-BY 4.0 (data)


Purpose:
    Infrastructure layer for PDSF experiments:
    1. Model registry and loading (single-GPU and multi-GPU with quantization)
    2. Hidden state extraction from transformer residual streams
    3. Part G causal intervention experiments (attenuate/rotate/mix/transplant
       on P, D, S, and F subspaces)

    The PDSF basis computation itself lives in pds_continuation.py; this module
    imports and uses those functions.

Paper references:
    §5.1–5.2 (single-pass intervention results), Figure 6, Table A.4-1
    Appendix A.4 (intervention protocols)
    Companion paper (rotation hierarchy and F sub-subspace analyses)

Key components:
    MODEL_REGISTRY          — Maps short names to HuggingFace model IDs
    ModelBundle             — Container for loaded model + tokenizer + config
    load_model_bundle_multigpu() — Load with multi-GPU / 4-bit quantization
    extract_hidden_states() — Run forward pass, collect residual stream per layer
    run_scramble_experiment() — Part G: intervene on subspace, measure divergence
    run_invariance_control() — Null experiment: verify deterministic forward pass

Inputs:
    HuggingFace model IDs, prompt lists, cached hidden states.

Outputs:
    Hidden state tensors, Part G intervention results (JSON).

Dependencies:
    torch, numpy, transformers, bitsandbytes (optional, for quantization),
    tqdm, pds_continuation, pds_spirality (optional, for Part M)
"""

from __future__ import annotations


import json
import os
import re
import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# Optional: suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Script version for result tracking
__version__ = "1.0"


# ============================================================
# MODEL REGISTRY AND CONFIGURATION
# ============================================================

MODEL_REGISTRY = {
    # Open AI
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "gpt_oss_20b": "openai/gpt-oss-20b",
    
    # === LLAMA FAMILY ===
    "llama_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama_405b": "meta-llama/Llama-3.1-405B-Instruct",
    "llama_405b_fp8": "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
    
    # === QWEN FAMILY ===
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    
    # === MISTRAL FAMILY ===
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    
    # === GEMMA FAMILY ===
    "gemma_27b": "google/gemma-2-27b-it",
    "gemma_9b": "google/gemma-2-9b-it",
    "gemma_2b": "google/gemma-2-2b-it",
    
    # === PHI FAMILY ===
    "phi3_medium": "microsoft/Phi-3-medium-4k-instruct",
    "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
}

# Approximate sizes in GB (BF16, for quantization decisions)
MODEL_SIZES_GB = {
    "gpt_oss_120b": 65,
    "gpt_oss_20b": 40,    
    "llama_70b": 140,
    "llama_405b": 810,
    "llama_405b_fp8": 405,
    "llama_8b": 16,
    "llama_3b": 6,
    "llama_1b": 2,
    "qwen_72b": 145,
    "qwen_32b": 65,
    "qwen_14b": 28,
    "qwen_7b": 14,
    "mistral_7b": 14,
    "mixtral_8x7b": 90,
    "gemma_27b": 54,
    "gemma_9b": 18,
    "gemma_2b": 5,
    "phi3_medium": 28,
    "phi3_mini": 8,
}

# Model families for grouping in analysis
MODEL_FAMILIES = {
    "llama": ["llama_70b", "llama_405b", "llama_405b_fp8", "llama_8b", "llama_3b", "llama_1b"],
    "qwen": ["qwen_72b", "qwen_32b", "qwen_14b", "qwen_7b"],
    "mistral": ["mistral_7b", "mixtral_8x7b"],
    "gemma": ["gemma_27b", "gemma_9b", "gemma_2b"],
    "phi": ["phi3_medium", "phi3_mini"],
    "gpt": ["gpt_oss_120b", "gpt_oss_20b"],
}


# ============================================================
# RESULT METADATA GENERATION 
# ============================================================
# This utility generates standardized metadata headers for all result JSON files,
# ensuring traceability and reproducibility of experiments.

def generate_result_metadata(
    experiment_name: str,
    experiment_version: str,
    model_id: str,
    model_key: str,
    n_samples: int,
    n_layers: int,
    hidden_dim: int,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a standardized metadata dictionary for result files.
    
    This metadata header is included at the top level of all JSON result files
    to provide complete information about the experiment context.
    
    Args:
        experiment_name: Name of the experiment (e.g., "SpecA_Geometry", "SpecB_Scramble")
        experiment_version: Version string for the experiment protocol
        model_id: Full HuggingFace model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        model_key: Sanitized model key used in filenames (e.g., "meta-llama__Llama-3.3-70B-Instruct")
        n_samples: Number of samples/prompts processed
        n_layers: Number of transformer layers in the model
        hidden_dim: Dimension of the hidden/residual stream
        extra_info: Optional dict of additional experiment-specific metadata
        
    Returns:
        Dict containing:
        - experiment_name: Identifies the experiment type
        - experiment_version: Protocol version for reproducibility
        - model_id: Full model identifier
        - model_key: Filename-safe model identifier  
        - run_timestamp: ISO format timestamp of when results were generated
        - n_samples: Number of data points
        - n_layers: Model architecture info
        - hidden_dim: Model architecture info
        - script_version: Version of residual_utils.py used
        - Any additional fields from extra_info
        
    Example:
        >>> metadata = generate_result_metadata(
        ...     experiment_name="SpecA_Geometry",
        ...     experiment_version="2.0",
        ...     model_id="meta-llama/Llama-3.1-8B-Instruct",
        ...     model_key="meta-llama__Llama-3.1-8B-Instruct",
        ...     n_samples=256,
        ...     n_layers=32,
        ...     hidden_dim=4096,
        ... )
        >>> # Include in results:
        >>> results = {"metadata": metadata, "by_layer": {...}}
    """
    from datetime import datetime, timezone
    
    metadata = {
        "experiment_name": experiment_name,
        "experiment_version": experiment_version,
        "model_id": model_id,
        "model_key": model_key,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "script_version": __version__,
    }
    
    if extra_info:
        metadata.update(extra_info)
    
    return metadata


# ============================================================
# MODEL REGISTRY 
# ============================================================
# 
# NAMING CONVENTION:
# -----------------
# Keys follow the pattern: {family}_{size} where:
#   - family: Model family (llama, qwen, gemma, phi, mistral)
#   - size: Parameter count (1b, 3b, 8b, 70b, 405b)
#
# VERSION TRACKING:
# ----------------
# - llama_70b: Llama-3.3 series (Jan 2025) - latest at time of writing
# - llama_405b: Llama-3.1 series - largest open-weight model
# - llama_8b/3b/1b: Llama-3.1/3.2 series
# - qwen_*: Qwen 2.5 series (Oct 2024)
# - gemma_*: Gemma 2 series (mid-2024)
# - phi3_*: Phi-3 series (Apr 2024)
#
# When new model versions are released, add new entries rather than
# overwriting existing ones to maintain reproducibility.
#


# ============================================================
# GPU PROFILES 
# ============================================================
#
# MEMORY CALCULATIONS:
# -------------------
# Each profile specifies max_memory per GPU. This is set ~5GB below
# the physical VRAM to leave room for:
#   - CUDA context overhead (~1-2GB)
#   - Activation memory during forward pass
#   - Safety margin for batch processing
#
# QUANTIZATION THRESHOLDS:
# -----------------------
# The quantize_4bit_threshold determines when to use BitsAndBytes 4-bit
# quantization instead of full precision (bfloat16).
#
# Formula: threshold = total_vram * 0.6 (approximate)
#   - 4xA100-80GB (320GB) → threshold ~200GB
#   - 3xRTX-6000 (288GB)  → threshold ~150GB
#   - 1xA100-80GB (80GB)  → threshold ~45GB
#
# Models exceeding the threshold require 4-bit quantization to fit.
# 70B models (~140GB bf16) need 2+ A100s or quantization
# 405B models (~810GB bf16) always need quantization + multi-GPU
#
GPU_PROFILES = {
    "4xA100_PCIe": {
        "description": "4× A100 PCIe (320GB total) - RECOMMENDED",
        "device": "auto",  # Let Accelerate handle layer distribution
        "max_memory": {0: "75GB", 1: "75GB", 2: "75GB", 3: "75GB"},  # 300GB usable
        "quantize_4bit_threshold": 200,  # Use 4-bit if model > this GB
    },
    "3xRTX_PRO_6000": {
        "description": "3× RTX PRO 6000 (288GB total) - FALLBACK",
        "device": "auto",
        "max_memory": {0: "90GB", 1: "90GB", 2: "90GB"},  # 270GB usable
        "quantize_4bit_threshold": 150,  # More conservative for RTX (different memory architecture)
    },
    "4xH100_SXM": {
        "description": "4× H100 SXM (320GB total) - PREMIUM",
        "device": "auto",
        "max_memory": {0: "75GB", 1: "75GB", 2: "75GB", 3: "75GB"},
        "quantize_4bit_threshold": 200,
    },
    "2xH200_SXM": {
        "description": "2× H200 SXM (282GB total)",
        "device": "auto",
        "max_memory": {0: "135GB", 1: "135GB"},  # 270GB usable
        "quantize_4bit_threshold": 150,
    },
    "1xA100_80GB": {
        "description": "1× A100 80GB - Single GPU",
        "device": "cuda",  # Single device, no distribution
        "max_memory": None,  # Use all available
        "quantize_4bit_threshold": 45,  # Must quantize 70B+ (140GB → ~35GB at 4-bit)
    },
}


# ============================================================
# Configuration
# ============================================================
@dataclass
class ExtractConfig:
    """Configuration for hidden state extraction.
    
    Attributes:
        pos_mode: Position selection mode
            - "last": Extract from the last token position (default)
            - "index": Extract from a specific position
        pos_index: Index to use when pos_mode="index" (-1 = last)
    """
    pos_mode: str = "last"  # "last" or "index"
    pos_index: int = -1     # Used when pos_mode="index"


@dataclass 
class ModelBundle:
    """Container for loaded model and tokenizer.
    
    Bundles all model-related objects to simplify passing between functions.
    
    Attributes:
        model: The loaded HuggingFace model
        tokenizer: The loaded tokenizer
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        device: Primary device (may be distributed across multiple GPUs)
        n_layers: Number of transformer layers
        hidden_dim: Dimension of hidden states (residual stream width)
        vocab_size: Vocabulary size
        is_quantized: Whether model is loaded with quantization 
    """
    model: Any
    tokenizer: Any
    model_id: str
    device: torch.device
    n_layers: int
    hidden_dim: int
    vocab_size: int
    is_quantized: bool = False  # Track quantization status


# ============================================================
# Device Utilities 
# ============================================================
#
# MULTI-GPU INPUT PLACEMENT LOGIC:
# -------------------------------
# When using device_map="auto", HuggingFace Accelerate distributes model
# layers across multiple GPUs. The challenge is determining where to place
# input tensors - they must go to the same device as the model's first layer.
#
# The hf_device_map attribute (if present) maps layer names to device indices.
# We search for the embedding layer using common naming conventions:
#   - Llama/Qwen: 'model.embed_tokens'
#   - GPT-2/GPT-J: 'transformer.wte'  
#   - Some encoder-decoder: 'model.decoder.embed_tokens'
#
# If the embedding isn't found in the map, we fall back to the first device
# in the map, which is typically where initial layers are placed.

def get_model_input_device(model: Any) -> torch.device:
    """Get the device where model inputs should be placed.
    
    For multi-GPU models with device_map="auto", inputs should go to
    the device of the first layer (usually cuda:0), not necessarily
    where the embedding layer is.
    
    This function inspects the model's device map to find where inputs
    should be placed, falling back to parameter inspection for single-GPU models.
    
    Args:
        model: HuggingFace model (potentially distributed across GPUs)
        
    Returns:
        torch.device for input tensors (e.g., torch.device("cuda:0"))
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
        >>> input_device = get_model_input_device(model)
        >>> input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(input_device)
    """
    # Check if model has device_map (distributed across GPUs)
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Find the device of the input embeddings
        device_map = model.hf_device_map
        # Common keys for input embeddings across different architectures
        for key in ['model.embed_tokens', 'embed_tokens', 'transformer.wte', 'model.decoder.embed_tokens']:
            if key in device_map:
                device = device_map[key]
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                return torch.device(device)
        # Fallback: use first device in map (typically where embedding is)
        first_device = next(iter(device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    
    # Single device model: get device from first parameter
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_gpu_memory():
    """Force GPU memory cleanup between operations.
    
    Call this after model unloading or between large operations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# Token Matching Utilities (DIAGNOSTIC ONLY)
# ============================================================
# IMPORTANT: These functions are used ONLY for verification/logging.
# They do NOT affect geometry analysis in any way.

def normalize_token_str(s: str) -> str:
    """Normalize a token string for comparison.
    
    Handles: leading/trailing whitespace, case differences, newlines.
    Used for diagnostic matching only - does not affect geometry.
    """
    normalized = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return normalized.strip().lower()


def is_whitespace_only(s: str) -> bool:
    """Check if string is only whitespace (including newlines)."""
    return len(s.strip()) == 0


def normalize_soft_answer(s: str) -> str:
    """Normalize for soft answer matching."""
    s2 = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    s2 = s2.strip().lower()
    s2 = re.sub(r"[\s\.\,\:\;\)\]\}]+$", "", s2)
    return s2


# ============================================================
# Chat Template Support
# ============================================================
def apply_chat_template_if_needed(model_id: str, prompt: str) -> str:
    """Wrap prompt in chat template for models that require it.
    
    Different model families expect different prompt formats.
    """
    model_lower = model_id.lower()
    
    # Llama 3.x models
    if "llama-3" in model_lower or "llama-3" in model_lower.replace(".", "-"):
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Qwen 2.5 models
    if "qwen" in model_lower:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Mistral/Mixtral models
    if "mistral" in model_lower or "mixtral" in model_lower:
        return f"[INST] {prompt} [/INST]"
    
    # Gemma models
    if "gemma" in model_lower:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Phi-3 models (all variants)
    if "phi-3" in model_lower:
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    # Fallback: return unchanged
    return prompt


def coerce_expected_to_list(expected_next_token):
    """Convert expected_next_token to a list of strings."""
    if expected_next_token is None:
        return []
    if isinstance(expected_next_token, list):
        return [str(x) for x in expected_next_token]
    return [str(expected_next_token)]


def parse_choice_tokens(answer_options):
    """Parse answer option tokens from `answer_options` field."""
    if answer_options is None:
        return []
    if isinstance(answer_options, list):
        return [str(x).strip() for x in answer_options if str(x).strip()]
    s = str(answer_options)
    letters = re.findall(r"\b([A-D])\b", s)
    if letters:
        seen = set()
        out = []
        for L in letters:
            if L not in seen:
                out.append(L)
                seen.add(L)
        return out
    parts = re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            left = p.split("=", 1)[0].strip()
            if left:
                out.append(left)
        else:
            out.append(p.split()[0])
    seen = set()
    uniq = []
    for o in out:
        if o and o not in seen:
            uniq.append(o)
            seen.add(o)
    return uniq


def is_minor_token_mismatch(got_str: str, expected_str: str) -> bool:
    """Check if a mismatch is minor (same semantic answer, different tokenization)."""
    if is_whitespace_only(got_str):
        return False
    return normalize_token_str(got_str) == normalize_token_str(expected_str)


def classify_mismatch(got_str: str, expected_str: str) -> str:
    """Classify a mismatch into categories for diagnostic purposes."""
    if got_str == expected_str:
        return 'match'
    if is_whitespace_only(got_str):
        return 'error_newline'
    got_norm = normalize_token_str(got_str)
    exp_norm = normalize_token_str(expected_str)
    if got_norm != exp_norm:
        return 'error_wrong_answer'
    got_stripped = got_str.strip()
    exp_stripped = expected_str.strip()
    if got_stripped == exp_stripped:
        return 'minor_whitespace'
    elif got_stripped.lower() == exp_stripped.lower():
        has_ws_diff = got_str != got_stripped or expected_str != exp_stripped
        return 'minor_case_whitespace' if has_ws_diff else 'minor_case'
    else:
        return 'minor_other'


def find_target_token_position(
    tokenizer, 
    output_ids: List[int], 
    target_str: str,
    search_last_n: int = 10,
) -> Optional[int]:
    """Find the position of a target token in generated output."""
    target_norm = normalize_token_str(target_str)
    start_pos = max(0, len(output_ids) - search_last_n)
    for pos in range(len(output_ids) - 1, start_pos - 1, -1):
        token_id = output_ids[pos]
        token_str = tokenizer.decode([token_id])
        if normalize_token_str(token_str) == target_norm:
            return pos
    return None


# ============================================================
# Token Encoding Utilities
# ============================================================
def encode_token_sequence(tokenizer, text: str, add_special_tokens: bool = False) -> List[int]:
    """Encode text to token IDs, returning empty list on failure."""
    if not text:
        return []
    try:
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return ids if ids else []
    except Exception:
        return []


def get_token_id(tokenizer, text: str) -> Optional[int]:
    """Get the first token ID for text, or None if encoding fails."""
    ids = encode_token_sequence(tokenizer, text, add_special_tokens=False)
    return ids[0] if ids else None


# ============================================================
# Model Loading
# ============================================================
def load_model_bundle(
    model_id: str,
    device: str = "cuda",
    torch_dtype: Optional[str] = None,
    trust_remote_code: bool = True,
) -> ModelBundle:
    """Load model and tokenizer into a ModelBundle.
    
    Automatically handles multi-GPU distribution for large models when
    device_map="auto" is appropriate.
    
    Args:
        model_id: HuggingFace model identifier
        device: Target device ("cuda", "cpu", "auto" for multi-GPU)
        torch_dtype: Data type ("bfloat16", "float16", "float32", or None for auto)
        trust_remote_code: Whether to trust remote code
        
    Returns:
        ModelBundle with loaded model and tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Map string dtype to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        None: "auto",
        "auto": "auto",
    }
    dtype = dtype_map.get(torch_dtype, torch_dtype)
    
    print(f"Loading model: {model_id}")
    
    # Handle device placement
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        device_obj = torch.device("cuda:0")
    elif device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        model = model.to("cpu")
        device_obj = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        device_obj = torch.device(device)
    
    model.eval()
    
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden_dim = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=device_obj,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        is_quantized=False,
    )


def load_model_bundle_multigpu(
    model_id: str,
    torch_dtype: Optional[str] = "bfloat16",
    trust_remote_code: bool = True,
    quantize_4bit: bool = False,
    max_memory: Optional[Dict[int, str]] = None,
) -> ModelBundle:
    """Load a large model distributed across multiple GPUs.
    
    Specifically designed for 70B and 405B models on multi-GPU setups.
    
    Args:
        model_id: HuggingFace model identifier
        torch_dtype: Data type (default bfloat16)
        trust_remote_code: Whether to trust remote code
        quantize_4bit: Whether to use 4-bit quantization (required for 405B)
        max_memory: Dict mapping GPU index -> max memory string
                    e.g., {0: "70GB", 1: "70GB", 2: "70GB"}
                    
    Returns:
        ModelBundle with distributed model
        
    Example:
        # For Llama-70B on 4xA100-80GB:
        bundle = load_model_bundle_multigpu(
            "meta-llama/Llama-3.3-70B-Instruct",
            torch_dtype="bfloat16",
        )
        
        # For Llama-405B (requires quantization):
        bundle = load_model_bundle_multigpu(
            "meta-llama/Llama-3.1-405B-Instruct",
            quantize_4bit=True,
            max_memory={0: "75GB", 1: "75GB", 2: "75GB", 3: "75GB"},
        )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Map string dtype to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        None: torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    print(f"Loading model (multi-GPU): {model_id}")
    print(f"  Quantization: {'4-bit' if quantize_4bit else 'None'}")
    print(f"  Data type: {dtype}")
    
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }
    
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = dtype
    
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    
    # Get the actual input device for this distributed model
    input_device = get_model_input_device(model)
    
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden_dim = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"  Input device: {input_device}")
    
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=input_device,  # Use correct input device
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        is_quantized=quantize_4bit,
    )


# ============================================================
# Hidden State Extraction
# ============================================================
@torch.no_grad()
def extract_hidden_states(
    bundle: ModelBundle,
    prompts: List[str],
    pos_cfg: ExtractConfig = None,
    max_length: int = 4096,
    batch_size: int = 1,
    move_to_cpu: bool = True,
    show_progress: bool = True,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[int], List[int]]:
    """Extract hidden states from all layers for a list of prompts.
    
    This is the core extraction function for geometry analysis. It runs each
    prompt through the model and captures the residual stream (hidden states)
    at a specified position for all layers.
    
    Args:
        bundle: Model bundle
        prompts: List of prompt strings
        pos_cfg: Position configuration (default: last token)
        max_length: Maximum sequence length
        batch_size: Batch size for processing (default 1 for large models)
        move_to_cpu: Whether to move tensors to CPU immediately (saves VRAM)
        show_progress: Whether to show tqdm progress bar 
        
    Returns:
        Tuple of:
        - layer_to_H: Dict mapping layer index to hidden states [n_prompts, hidden_dim]
        - logits_last: Logits at extraction position [n_prompts, vocab_size]
        - pred_ids: Predicted token IDs at extraction position
        - seq_lens: Sequence lengths for each prompt
    """
    if pos_cfg is None:
        pos_cfg = ExtractConfig()
    
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.n_layers
    
    # Get correct input device for multi-GPU models
    input_device = get_model_input_device(model)
    
    # Initialize storage
    layer_to_H = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer
    all_logits = []
    all_pred_ids = []
    all_seq_lens = []
    
    # Optional progress bar
    prompt_iter = prompts
    if show_progress:
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(prompts, desc="Extracting hidden states", dynamic_ncols=True)
        except ImportError:
            pass
    
    for prompt in prompt_iter:
        # Apply chat template if model requires it
        prompt_for_model = apply_chat_template_if_needed(bundle.model_id, prompt)
        
        tok_out = tokenizer(
            prompt_for_model,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        # Place inputs on correct device
        input_ids = tok_out["input_ids"].to(input_device)
        seq_len = input_ids.shape[1]
        all_seq_lens.append(seq_len)
        
        # Forward pass with hidden states
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # Determine extraction position
        if pos_cfg.pos_mode == "last":
            pos = -1
        else:
            pos = pos_cfg.pos_index
            if pos < 0:
                pos = seq_len + pos
        
        # Extract hidden states at position
        hidden_states = outputs.hidden_states
        for layer_idx, h in enumerate(hidden_states):
            h_pos = h[0, pos, :]
            # Ensure float conversion for quantized models
            if move_to_cpu:
                h_pos = h_pos.float().cpu()
            layer_to_H[layer_idx].append(h_pos)
        
        # Extract logits and prediction
        logits = outputs.logits[0, pos, :]
        if move_to_cpu:
            logits = logits.float().cpu()
        all_logits.append(logits)
        all_pred_ids.append(int(torch.argmax(logits).item()))
    
    # Stack into tensors
    layer_to_H = {k: torch.stack(v) for k, v in layer_to_H.items()}
    logits_last = torch.stack(all_logits)
    
    return layer_to_H, logits_last, all_pred_ids, all_seq_lens


# ============================================================
# Verification (DIAGNOSTIC ONLY)
# ============================================================
def verify_expected_tokens(
    bundle: ModelBundle,
    prompts: List[str],
    expected_strs: List[str],
    max_length: int = 4096,
) -> List[dict]:
    """Verify that model predicts expected tokens.
    
    THIS IS DIAGNOSTIC ONLY - does not affect geometry calculations.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    input_device = get_model_input_device(model)
    
    mismatches = []
    
    for i, (prompt, expected_str) in enumerate(zip(prompts, expected_strs)):
        prompt_for_model = apply_chat_template_if_needed(bundle.model_id, prompt)
        
        tok_out = tokenizer(prompt_for_model, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tok_out["input_ids"].to(input_device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=False, use_cache=False)
            logits = outputs.logits
        
        got_id = int(torch.argmax(logits[0, -1]).item())
        got_str = tokenizer.decode([got_id])
        
        exp_ids = encode_token_sequence(tokenizer, expected_str)
        exp_id = exp_ids[0] if exp_ids else -1
        
        is_match = is_minor_token_mismatch(got_str, expected_str) or (got_id == exp_id)
        
        if not is_match:
            mismatch_type = classify_mismatch(got_str, expected_str)
            mismatches.append({
                "index": i,
                "got_id": got_id,
                "got_str": got_str,
                "expected_id": exp_id,
                "expected_str": expected_str,
                "mismatch_type": mismatch_type,
            })
    
    return mismatches


def verify_expected_tokens_detailed(
    bundle: ModelBundle,
    prompts: List[str],
    meta: List[dict],
    max_length: int = 4096,
    show_progress: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Detailed verification that separates minor mismatches from real errors.
    
    THIS IS DIAGNOSTIC ONLY - it does not affect geometry calculations.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    input_device = get_model_input_device(model)
    
    exact_matches = []
    minor_mismatches = []
    real_errors = []
    
    # Optional progress bar
    prompt_iter = enumerate(prompts)
    if show_progress:
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(list(enumerate(prompts)), desc="Verifying predictions")
        except ImportError:
            pass

    for i, prompt in prompt_iter:
        expected_list = coerce_expected_to_list(meta[i].get('expected_next_token'))
        _ = parse_choice_tokens(meta[i].get('answer_options'))

        expected_ids_list = []
        for exp in expected_list:
            ids = encode_token_sequence(tokenizer, exp)
            if ids:
                expected_ids_list.append(int(ids[0]))
        expected_ids_list = list(dict.fromkeys(expected_ids_list))

        expected_str = expected_list[0] if expected_list else ''
        expected_ids = encode_token_sequence(tokenizer, expected_str)

        prompt_for_model = apply_chat_template_if_needed(
            bundle.model_id if hasattr(bundle, 'model_id') else str(bundle.model.config._name_or_path), 
            prompt
        )
        
        tok_out = tokenizer(
            prompt_for_model, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        )
        p_ids = tok_out["input_ids"].to(input_device)

        with torch.no_grad():
            out = model(input_ids=p_ids, output_hidden_states=False, use_cache=False)
            logits = out.logits

        got_id = int(torch.argmax(logits[0, -1]).item())
        exp_id = int(expected_ids[0]) if expected_ids else -1
        got_str = tokenizer.decode([got_id])

        record = {
            "index": i,
            "group_id": meta[i]["group_id"],
            "variant_id": meta[i]["variant_id"],
            "got_id": got_id,
            "expected_id": exp_id,
            "got_str": got_str,
            "expected_str": expected_str,
        }
        
        strict_id_match = (len(expected_ids_list) > 0) and (got_id in expected_ids_list)

        got_soft = normalize_soft_answer(got_str)
        expected_soft = set(normalize_soft_answer(s) for s in expected_list)
        soft_match = False
        if expected_soft:
            if got_soft in expected_soft:
                soft_match = True
            else:
                m_choice = re.match(r'^([a-d])\b', got_soft)
                if m_choice and m_choice.group(1) in expected_soft:
                    soft_match = True

        record['strict_id_match'] = bool(strict_id_match)
        record['soft_match'] = bool(soft_match)

        if strict_id_match:
            record['mismatch_type'] = 'match'
            exact_matches.append(record)
        elif soft_match:
            record['mismatch_type'] = 'minor_softmatch'
            minor_mismatches.append(record)
        else:
            mismatch_type = classify_mismatch(got_str, expected_str)
            record['mismatch_type'] = mismatch_type
            if mismatch_type.startswith('error'):
                real_errors.append(record)
            else:
                minor_mismatches.append(record)

    return exact_matches, minor_mismatches, real_errors


# ============================================================
# Generation Utilities
# ============================================================
@torch.no_grad()
def greedy_generate(
    bundle: ModelBundle,
    prompt: str,
    n_tokens: int,
    max_length: int = 4096,
) -> List[int]:
    """Generate tokens greedily from a prompt."""
    model = bundle.model
    tokenizer = bundle.tokenizer
    input_device = get_model_input_device(model)
    
    tok_out = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = tok_out["input_ids"].to(input_device)
    
    generated = []
    for _ in range(n_tokens):
        outputs = model(input_ids=input_ids, output_hidden_states=False, use_cache=False)
        next_id = int(torch.argmax(outputs.logits[0, -1]).item())
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_device)], dim=1)
        
        if next_id == tokenizer.eos_token_id:
            break
    
    return generated


# ============================================================
# Linear Algebra Utilities
# ============================================================
#
# These functions implement the core mathematical operations for the
# Intersection Geometry Experiment. The key insight is that transformer
# residual streams live in high-dimensional vector spaces, and we can
# decompose representations using linear algebra.
#
# KEY CONCEPTS:
# ------------
# - Participation Ratio (PR): Measures effective dimensionality of a point cloud.
#   PR = (Σλ)² / Σλ² where λ are covariance eigenvalues.
#   PR=1 means all variance in one dimension; PR=d means uniform across d dims.
#
# - Subspace Projection: Given orthonormal basis B (shape d×r), projection of
#   vector v onto span(B) is: proj = B @ (B.T @ v) = B @ B.T @ v
#
# - Principal Angles: Measure alignment between two subspaces. Computed as
#   arccos of singular values of U.T @ V where U, V are orthonormal bases.
#   Angle = 0° means perfectly aligned; 90° means orthogonal.
#
# NUMERICAL STABILITY:
# -------------------
# - Always use SVD-based methods rather than eigendecomposition of X.T @ X
# - Clip singular values to [-1, 1] before arccos to handle numerical error
# - Use eps thresholds to avoid division by zero

def compute_participation_ratio(vectors: np.ndarray, center: bool = True) -> float:
    """Compute participation ratio (PR) of a set of vectors.
    
    PR is a measure of effective dimensionality based on covariance eigenvalues:
        PR = (Σλ)² / Σλ²
    
    Interpretation:
    - PR = 1: All variance along one direction (maximally anisotropic)
    - PR = d: Variance uniform across d dimensions (isotropic)
    - PR typically << d for neural network representations
    
    Mathematical background:
    The participation ratio is equivalent to the inverse of the "inverse participation
    ratio" from statistical mechanics. It provides a soft measure of the number of
    "active" dimensions, robust to the presence of small eigenvalues.
    
    Reference: Gao et al. (2017) "On the dimensionality of word embedding"
    
    Args:
        vectors: Data matrix of shape (n_samples, d_dimensions)
        center: Whether to center data by subtracting mean (recommended)
        
    Returns:
        Participation ratio (float between 1 and min(n-1, d))
    """
    if center:
        vectors = vectors - vectors.mean(axis=0, keepdims=True)
    
    # Compute covariance matrix: Σ = (1/n) X.T @ X
    cov = vectors.T @ vectors / len(vectors)
    # Use eigvalsh for symmetric matrices (faster, more stable)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative eigenvalues (numerical error)
    
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    
    if sum_eig_sq < 1e-12:
        return 1.0  # Degenerate case: no variance
    
    return (sum_eig ** 2) / sum_eig_sq


def project_onto_subspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vectors onto a subspace defined by an orthonormal basis.
    
    For orthonormal basis B (columns), projection matrix is P = B @ B.T
    The projection of v is: P @ v = B @ (B.T @ v)
    
    This computes the component of each vector that lies within the subspace.
    
    Args:
        vectors: Data matrix of shape (n, d)
        basis: Orthonormal basis of shape (d, r) where r is subspace dimension
        
    Returns:
        Projected vectors of shape (n, d)
    """
    # coefficients[i] = basis.T @ vectors[i] gives coordinates in subspace
    coefficients = vectors @ basis.T  # (n, r)
    # Reconstruct in original space
    projected = coefficients @ basis  # (n, d)
    return projected


def project_out_subspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vectors out of a subspace (onto orthogonal complement).
    
    This removes the component of each vector that lies within the subspace,
    leaving only the orthogonal complement.
    
    v_perp = v - proj_B(v) = v - B @ B.T @ v
    
    Args:
        vectors: Data matrix of shape (n, d)
        basis: Orthonormal basis of shape (d, r) to project out
        
    Returns:
        Residual vectors of shape (n, d), orthogonal to span(basis)
    """
    projected = project_onto_subspace(vectors, basis)
    return vectors - projected


def orthonormalize(vectors: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Orthonormalize vectors using SVD (more stable than Gram-Schmidt).
    
    Given vectors V, returns an orthonormal basis for span(V) via SVD:
    V = U @ S @ Vt => the columns of U corresponding to non-zero S are orthonormal
    
    Args:
        vectors: Input vectors of shape (d, m) or (m, d)
        eps: Threshold for determining numerical rank (singular values < eps*max are dropped)
        
    Returns:
        Orthonormal basis of shape (d, r) where r is numerical rank
    """
    U, S, Vt = np.linalg.svd(vectors, full_matrices=False)
    # Numerical rank: count singular values > eps * max(S)
    rank = np.sum(S > eps)
    return Vt[:rank]


def compute_principal_angles(basis1: np.ndarray, basis2: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.
    
    Principal angles θ₁ ≤ θ₂ ≤ ... ≤ θᵣ are defined via singular values of U.T @ V:
        σᵢ = cos(θᵢ)
        
    where U, V are orthonormal bases and r = min(rank(U), rank(V)).
    
    Interpretation:
    - θ = 0°: Subspaces share a direction (aligned)
    - θ = 90°: Subspaces are orthogonal in that dimension
    - All angles = 0°: Subspaces are identical
    - All angles = 90°: Subspaces are completely orthogonal
    
    Reference: Golub & Van Loan, "Matrix Computations", Section 6.4.3
    
    Args:
        basis1: First orthonormal basis, shape (d, r1)
        basis2: Second orthonormal basis, shape (d, r2)
        
    Returns:
        Array of principal angles in radians, length = min(r1, r2)
    """
    # M = U.T @ V has shape (r1, r2); its singular values are cos(θᵢ)
    M = basis1 @ basis2.T
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    # Clip to handle numerical error (cos should be in [-1, 1])
    S = np.clip(S, -1.0, 1.0)
    angles = np.arccos(S)
    return angles


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    cos(θ) = (a · b) / (||a|| ||b||)
    
    Returns value in [-1, 1]:
    - 1: Vectors point in same direction
    - 0: Vectors are orthogonal  
    - -1: Vectors point in opposite directions
    
    Args:
        a: First vector
        b: Second vector (same dimension as a)
        
    Returns:
        Cosine similarity (float)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0  # Handle zero vectors
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute cosine similarity between two tensors (GPU-compatible).
    
    Same as compute_cosine_similarity but for PyTorch tensors.
    Uses F.normalize for numerical stability.
    
    Args:
        a: First tensor
        b: Second tensor
        eps: Small value to avoid division by zero
        
    Returns:
        Cosine similarity as 0-dimensional tensor
    """
    a_norm = F.normalize(a.unsqueeze(0), dim=1, eps=eps).squeeze(0)
    b_norm = F.normalize(b.unsqueeze(0), dim=1, eps=eps).squeeze(0)
    return torch.dot(a_norm, b_norm)


# ============================================================
# CKA (Centered Kernel Alignment)
# ============================================================
def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)
    
    if hsic_xx < 1e-12 or hsic_yy < 1e-12:
        return 0.0
    
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


# ============================================================
# Model Layer Access 
# ============================================================
def get_unembedding_matrix(model: Any) -> torch.Tensor:
    """Get the unembedding (lm_head) matrix from a model.
    
    Handles meta tensors from device_map='auto' with MXFP4/quantized models
    by scanning named_parameters for the actual materialized weight.
    
    Returns:
        Tensor of shape [vocab_size, hidden_dim]
    """
    # Try standard access first
    w = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        w = model.lm_head.weight
    elif hasattr(model, 'get_output_embeddings'):
        emb = model.get_output_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            w = emb.weight
    
    if w is None:
        raise AttributeError("Could not locate unembedding matrix in model.")
    
    # Handle meta tensors (no actual data — just shape placeholders)
    if w.device.type == 'meta':
        for name, param in model.named_parameters():
            if ('lm_head' in name or 'output' in name) and 'weight' in name:
                if param.device.type != 'meta':
                    return param.detach()
        # Fallback: try embed_tokens (some models tie weights)
        for name, param in model.named_parameters():
            if 'embed_tokens' in name and 'weight' in name:
                if param.device.type != 'meta':
                    return param.detach()
        raise RuntimeError(
            "Unembedding weight is on meta device and no materialized copy found. "
            "Model may not have loaded correctly — check device_map and quantization."
        )
    
    return w.detach()


# =============================================================================
# CACHE AND MEMORY MANAGEMENT UTILITIES
# =============================================================================

def clear_model_from_memory(bundle: Any = None, model: Any = None, verbose: bool = True) -> None:
    """
    Clear a model from GPU memory.
    
    Args:
        bundle: ModelBundle object (will delete bundle.model)
        model: Direct model reference (alternative to bundle)
        verbose: Print status messages
    """
    import gc
    
    if bundle is not None and hasattr(bundle, 'model'):
        if verbose:
            print("  Clearing model from bundle...")
        try:
            del bundle.model
        except:
            pass
        try:
            del bundle.tokenizer
        except:
            pass
    
    if model is not None:
        if verbose:
            print("  Clearing model...")
        try:
            del model
        except:
            pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose:
        print("  ✓ Model cleared from memory")


def get_cache_usage() -> Dict[str, Any]:
    """
    Get information about HuggingFace cache usage.
    
    Returns:
        Dict with cache locations, models found, and sizes
    """
    
    # Cache locations: check HF_HOME env var, then standard defaults
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_locations = [
        Path(hf_home) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    
    results = {
        "locations": [],
        "total_size_gb": 0,
        "models": []
    }
    
    for cache_path in cache_locations:
        if not cache_path.exists():
            continue
            
        location_info = {
            "path": str(cache_path),
            "models": [],
            "size_gb": 0
        }
        
        for item in sorted(cache_path.iterdir()):
            if item.is_dir() and (item.name.startswith("models--") or not item.name.startswith(".")):
                try:
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / 1e9
                except:
                    size = 0
                    
                if size < 0.001:  # Skip if < 1MB
                    continue
                    
                model_name = item.name.replace("models--", "").replace("--", "/")
                location_info["models"].append({
                    "name": model_name,
                    "path": str(item),
                    "size_gb": round(size, 2)
                })
                location_info["size_gb"] += size
                results["models"].append(model_name)
        
        if location_info["models"]:
            results["locations"].append(location_info)
            results["total_size_gb"] += location_info["size_gb"]
    
    results["total_size_gb"] = round(results["total_size_gb"], 2)
    return results


def print_cache_status(verbose: bool = True) -> None:
    """
    Print a summary of HuggingFace cache usage.
    """
    import shutil
    
    cache_info = get_cache_usage()
    
    if not cache_info["locations"]:
        print("  No cached models found")
        return
    
    for loc in cache_info["locations"]:
        print(f"\n  {loc['path']}:")
        for model in loc["models"]:
            print(f"    {model['name']}: {model['size_gb']:.1f} GB")
    
    print(f"\n  Total cache size: {cache_info['total_size_gb']:.1f} GB")
    
    # Show disk space if possible
    try:
        for path in ["/workspace", Path.home()]:
            if Path(path).exists():
                total, used, free = shutil.disk_usage(path)
                print(f"  Disk ({path}): {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
                break
    except:
        pass


def clear_hf_cache_for_model(model_name: str, verbose: bool = True) -> bool:
    """
    Clear HuggingFace cache for a specific model.
    
    Args:
        model_name: Model name (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        verbose: Print status messages
    
    Returns:
        True if cache was cleared, False if model not found
    """
    import shutil
    
    # Cache locations: check HF_HOME env var, then standard defaults
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_locations = [
        Path(hf_home) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    
    # Convert model name to cache directory format
    cache_name = "models--" + model_name.replace("/", "--")
    
    found = False
    for cache_path in cache_locations:
        if not cache_path.exists():
            continue
            
        # Check for exact match
        model_path = cache_path / cache_name
        if model_path.exists():
            if verbose:
                size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / 1e9
                print(f"  Removing {model_name} ({size:.1f} GB) from {cache_path}...")
            shutil.rmtree(model_path)
            found = True
            
        # Also check for direct model directory (some setups)
        direct_path = cache_path / model_name.split("/")[-1]
        if direct_path.exists() and direct_path.is_dir():
            if verbose:
                size = sum(f.stat().st_size for f in direct_path.rglob('*') if f.is_file()) / 1e9
                print(f"  Removing {direct_path.name} ({size:.1f} GB) from {cache_path}...")
            shutil.rmtree(direct_path)
            found = True
    
    if verbose:
        if found:
            print(f"  ✓ Cache cleared for {model_name}")
        else:
            print(f"  Model not found in cache: {model_name}")
    
    return found


def clear_all_hf_cache(confirm: bool = True, verbose: bool = True) -> int:
    """
    Clear all models from HuggingFace cache.
    
    Args:
        confirm: If True, require user confirmation
        verbose: Print status messages
    
    Returns:
        Number of models cleared
    """
    import shutil
    
    cache_info = get_cache_usage()
    
    if not cache_info["models"]:
        if verbose:
            print("  No cached models to clear")
        return 0
    
    if verbose:
        print(f"  Found {len(cache_info['models'])} models ({cache_info['total_size_gb']:.1f} GB)")
        for model in cache_info["models"]:
            print(f"    - {model}")
    
    if confirm:
        response = input("\n  Clear all cached models? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("  Cancelled")
            return 0
    
    count = 0
    for loc in cache_info["locations"]:
        for model in loc["models"]:
            try:
                shutil.rmtree(model["path"])
                count += 1
                if verbose:
                    print(f"  ✓ Cleared {model['name']}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to clear {model['name']}: {e}")
    
    if verbose:
        print(f"\n  Cleared {count} models")
    
    return count


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# JSON SERIALIZATION HELPERS
# ============================================================

def to_jsonable(x):
    """Recursively convert common scientific/python objects into JSON-safe primitives."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # Paths
    if isinstance(x, Path):
        return str(x)

    # Numpy
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()

    # Torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().tolist()
    if isinstance(x, (torch.dtype, torch.device)):
        return str(x)

    # Dict-like
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if not (k is None or isinstance(k, (str, int, float, bool))):
                k = str(k)
            out[k] = to_jsonable(v)
        return out

    # Iterables
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    # Dataclasses
    try:
        if is_dataclass(x):
            return to_jsonable(_asdict(x))
    except Exception:
        pass

    # Fallback
    return str(x)


def save_json(obj: Any, path: Path):
    """Save object to JSON with automatic type conversion."""
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

@dataclass
class ScrambleConfig:
    """Configuration for the S-scramble experiment."""
    model_id: str
    analysis_dir: Path
    samples_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Scramble parameters
    scramble_layers: List[str] = field(default_factory=lambda: ["early"])
    n_scrambles: int = 10
    s_rank: int = 13
    seed: int = 42
    
    # Runtime
    verbose: bool = True
    
    def __post_init__(self):
        self.analysis_dir = Path(self.analysis_dir)
        if self.samples_path:
            self.samples_path = Path(self.samples_path)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)


# PDSDecomposition is imported from pds_continuation.py (canonical source)
from pds_continuation import PDSDecomposition


# ============================================================
# CACHED DATA LOADING
# ============================================================

def load_cached_extractions(
    analysis_dir: Path,
    model_id: str,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Load cached hidden state extractions from SpecA geometry run.
    
    Args:
        analysis_dir: Path to analysis directory containing cache/
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        
    Returns:
        H_by_layer: Dict mapping layer index to hidden states [n_prompts, d_model]
        metadata: Dict with group_ids, variant_ids, prompts, expected_tokens
    """
    model_key = model_id.replace("/", "__")
    cache_dir = analysis_dir / "cache"
    
    npz_path = cache_dir / f"extraction_{model_key}.npz"
    meta_path = cache_dir / f"extraction_{model_key}.meta.json"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Cached extraction not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    # Load hidden states
    data = np.load(npz_path)
    H_by_layer = {}
    for key in data.keys():
        if key.startswith("H_"):
            layer = int(key.split("_")[1])
            H_by_layer[layer] = data[key]
    
    # Load metadata
    with open(meta_path) as f:
        metadata = json.load(f)
    
    return H_by_layer, metadata


def load_specA_results(analysis_dir: Path, model_id: str) -> Dict[str, Any]:
    """Load existing SpecA analysis results for reference."""
    model_key = model_id.replace("/", "__")
    model_dir = analysis_dir / model_key
    
    results = {}
    for part in ["part_a_pr_summary", "part_d_trajectory", "part_e_mu_landscape", "part_f_rotation"]:
        path = model_dir / f"{part}.json"
        if path.exists():
            with open(path) as f:
                results[part] = json.load(f)
    
    return results


# ============================================================
# BASIS COMPUTATION
# ============================================================

# NOTE: P basis computation has been consolidated into pds_continuation.py
# compute_P_bases_from_predictions(). All experiments use the same function.
# The old compute_P_basis() function (unembedding per token string) has been removed.


# NOTE: S basis computation has been consolidated into pds_continuation.py
# compute_S_basis_global(). S = top-k PCA of H after projecting out P and D.
# The old compute_S_basis_from_cached() (PCA of group means of raw H) has been removed.


# NOTE: D basis computation has been consolidated into pds_continuation.py
# compute_D_basis_global(). D = top-k PCA of H after projecting out P.
# The old compute_D_basis() (projected out both P and S) has been removed.
# ============================================================
# HIDDEN STATE DECOMPOSITION
# ============================================================
# 
# decompose_hidden_state is imported from pds_continuation.py.
# It uses SEQUENTIAL orthogonal projection (P → D → S → F), which is the 
# mathematically correct approach. The old parallel projection version has been removed.
#
# Import at module level:
from pds_continuation import decompose_hidden_state as _decompose_sequential

def decompose_hidden_state(h, P_basis, D_basis, S_basis):
    """Decompose hidden state using sequential orthogonal projection.
    
    Delegates to the canonical implementation in pds_continuation.py.
    Args: h (d_model,), P_basis (d_model, r_P), D_basis (d_model, r_D), S_basis (d_model, r_S).
    Returns: PDSDecomposition with h_P, h_D, h_S, h_F and energy fields.
    See §3 of the paper (and Appendix A.3) for the PDSF decomposition methodology.
    """
    return _decompose_sequential(h, P_basis, D_basis, S_basis)


# ============================================================
# MODEL LAYER ACCESS
# ============================================================

def get_model_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Get the list of transformer layers from a model.
    
    Returns the actual transformer blocks, NOT including embedding layer.
    For a model with N layers, returns list of length N (indices 0 to N-1).
    
    Note on indexing:
        - model.layers[i] = transformer block i
        - hidden_states[i+1] = output after block i
        - hidden_states[0] = embedding output (before any blocks)
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)  # Llama, Qwen, Mistral
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)  # GPT-2 style
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        if hasattr(model.model.decoder, 'layers'):
            return list(model.model.decoder.layers)
    raise ValueError(f"Could not find layers in model: {type(model)}")


# ============================================================
# FORWARD PASS WITH HIDDEN STATE COLLECTION
# ============================================================

def run_forward_with_collection(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers_to_collect: List[int],
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    """
    Run forward pass and collect hidden states at specified layers.
    
    Uses HuggingFace outputs.hidden_states for collection (no hooks needed).
    
    Indexing convention:
      - hidden_states[0] = embeddings output
      - hidden_states[1] = after block 0
      - ...
      - hidden_states[n_layers] = after last block (final hidden state)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        layers_to_collect: List of layer indices to collect (0 = embedding, 1+ = after blocks)
        device: Device for computation
        
    Returns:
        collected: Dict mapping layer_idx -> hidden state at last token [d_model]
        logits: Logits at last position [vocab_size]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    last_pos = inputs["input_ids"].shape[1] - 1

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = outputs.hidden_states  # tuple of length (n_layers + 1)
        logits = outputs.logits[0, -1, :].detach().cpu().float()

    collected: Dict[int, np.ndarray] = {}
    for layer_idx in layers_to_collect:
        # Bounds check - skip invalid indices
        if layer_idx < 0 or layer_idx >= len(hs):
            continue
        collected[layer_idx] = hs[layer_idx][0, last_pos, :].detach().cpu().float().numpy()

    return collected, logits


# ============================================================
# GEOMETRY INTERVENTION TYPES
# ============================================================
# Part G tests all four subspaces (P, D, S, F) with subspace-appropriate interventions:
#   attenuate:  scale coefficients by (1-alpha), reducing energy (preserves direction)
#   rotation:   random orthogonal rotation within subspace (preserves magnitude) [P/D/S only]
#   mix:        permute + sign-flip coefficients, destroying structure (preserves magnitude) [F only]
#   transplant: replace with component from a different prompt (energy-matched)
#
# P, D, S (low-rank ≤ ~20 dims): attenuate, rotation (QR), transplant
# F (rank ~4000):                 attenuate, mix (QR intractable), transplant

GEOMETRY_SUBSPACES = ["P", "D", "S", "F"]
GEOMETRY_INTERVENTION_TYPES_PDS = ["attenuate", "rotation", "transplant"]
GEOMETRY_INTERVENTION_TYPES_F = ["attenuate", "mix", "transplant_within", "transplant_cross", "transplant_null"]
# NOTE: "transplant" is retained in PDS for backward compat (single cross-group donor).
# F uses three typed transplant conditions instead:
#   transplant_within : same regime/domain, different group  (within-regime baseline)
#   transplant_cross  : different regime/domain (+1 index)   (tests regime-specificity)
#   transplant_null   : single-space " " near-prior donor    (upper-anchor; see NULL_DONOR_IDX)
# 50% energy reduction; see Appendix A.4 of the paper for justification
DEFAULT_ATTENUATE_ALPHA = 0.5

# Single-space " " donor serves as near-prior upper anchor for the transplant conditions.
# No transplant result is reported in Paper 1; the conditions are retained so that the
# published Part G artefact reproduces exactly.
# Stored in transplant_donors at this key; never a real prompt index.
NULL_DONOR_IDX: int = -1

# ── Domain mappings for prompt sets without a regime field ───────────────────
# Principle: group by TYPE OF PROCESSING REQUIRED, matching SpecA's domain logic.

SPECA_GROUP_TO_DOMAIN: Dict[str, str] = {
    # Arithmetic: numerical computation, symbolic manipulation
    'G01_arith_yes_2plus2': 'arithmetic', 'G03_arith_yes_10gt5': 'arithmetic',
    'G04_arith_no_2gt7':    'arithmetic', 'G05_arith_yes_7prime': 'arithmetic',
    # Logic: formal inference, truth-value evaluation
    'G07_logic_true_tautology': 'logic',  'G09_logic_true_modustollens': 'logic',
    'N03_logic_always':         'logic',
    # Factual: world-knowledge retrieval
    'G13_fact_yes_paris': 'factual', 'G14_fact_no_london': 'factual',
    'G15_fact_yes_h2o':   'factual', 'N05_fact_geo_mc':    'factual',
    # Linguistic: form/structure analysis
    'G19_ling_a_vowel': 'linguistic', 'G21_ling_a_verb': 'linguistic',
    'N07_ling_pos3':    'linguistic',
}

SPECB_GROUP_TO_DOMAIN: Dict[str, str] = {
    # internal_states: agent internal-model tracking (Emotional, Evaluative, Epistemic)
    'group_01': 'internal_states', 'group_09': 'internal_states', 'group_11': 'internal_states',
    # agent_action: action/consequence schemas (Decision, Goal-Driven, Social Role)
    # NOTE: Social Role sits here (not narrative_frame) because its what_it_tests
    # emphasises role-appropriate action schemas and knowledge constraints, not narrator voice.
    'group_02': 'agent_action',    'group_03': 'agent_action',    'group_08': 'agent_action',
    # narrative_frame: discourse framing apparatus (Perspective/Voice, Genre/Register, Relationship)
    'group_04': 'narrative_frame', 'group_07': 'narrative_frame', 'group_12': 'narrative_frame',
    # context_setting: world-model grounding (Causal Setup, Temporal Anchor, Physical Setting)
    'group_05': 'context_setting', 'group_06': 'context_setting', 'group_10': 'context_setting',
}


def run_forward_with_intervention(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    scramble_layer: int,
    basis: np.ndarray,
    intervention: Dict[str, Any],
    layers_to_collect: List[int],
    device: torch.device,
    decompose_fn=None,
    subspace_name: str = "S",
) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    """
    Run forward pass with a generalized intervention at specified layer.
    
    The intervention is applied to the specified subspace component of the hidden
    state at the last token position of scramble_layer.
    
    For P/D/S subspaces: project onto basis, apply intervention to coefficients,
    reconstruct via delta = basis @ (c_new - c).
    
    For F subspace: compute h_F = h - h_P - h_D - h_S, apply intervention directly
    in model space (no explicit basis needed for attenuate/mix).
    
    Args:
        basis: Orthonormal basis [d_model, rank] for the target subspace.
               For F interventions, this should contain the P∪D∪S bases
               as a dict {"P": Bp, "D": Bd, "S": Bs}.
        intervention: Dict with keys:
            "type": "rotation" | "attenuate" | "mix" | "transplant"
            "R": rotation matrix [rank, rank] (for rotation, P/D/S only)
            "alpha": float (for attenuate, default 0.5)
            "perm": permutation array (for mix, F only)
            "signs": sign array ±1 (for mix, F only)
            "donor_coefficients": np.ndarray (for transplant, P/D/S)
            "donor_h_F": np.ndarray (for transplant, F)
        subspace_name: "P", "D", "S", or "F"
    """
    if scramble_layer <= 0:
        raise ValueError("scramble_layer must be >= 1")

    # Apply chat template for consistency with extract_hidden_states
    from pds_continuation import apply_chat_template as _apply_template
    prompt_formatted = _apply_template(model.config._name_or_path if hasattr(model, 'config') else "", prompt)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(device)
    last_pos = inputs["input_ids"].shape[1] - 1
    model_layers = get_model_layers(model)
    block_to_hook = scramble_layer - 1

    scramble_applied = {"done": False}
    dtype = next(model.parameters()).dtype
    
    int_type = intervention["type"]
    
    if subspace_name in ("P", "D", "S"):
        # Standard basis-based intervention
        Bs_t = torch.as_tensor(basis, device=device, dtype=dtype)
        
        if int_type == "rotation":
            R_t = torch.as_tensor(intervention["R"], device=device, dtype=dtype)
        elif int_type == "attenuate":
            alpha = float(intervention.get("alpha", DEFAULT_ATTENUATE_ALPHA))
        elif int_type == "transplant":
            donor_c_t = torch.as_tensor(intervention["donor_coefficients"], device=device, dtype=dtype)
        
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not scramble_applied["done"]:
                h_t = h[0, last_pos, :].clone()
                c = Bs_t.transpose(0, 1) @ h_t
                
                if int_type == "rotation":
                    c_new = R_t @ c
                elif int_type == "attenuate":
                    c_new = (1.0 - alpha) * c
                elif int_type == "transplant":
                    c_new = donor_c_t
                else:
                    c_new = c
                
                delta = Bs_t @ (c_new - c)
                h[0, last_pos, :] = h_t + delta
                scramble_applied["done"] = True
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
    
    else:  # F subspace
        # F = h - h_P - h_D - h_S. Interventions applied directly in model space.
        bases_dict = basis  # For F, 'basis' is {"P": Bp, "D": Bd, "S": Bs}
        Bp_t = torch.as_tensor(bases_dict["P"], device=device, dtype=dtype)
        Bd_t = torch.as_tensor(bases_dict["D"], device=device, dtype=dtype)
        Bs_t = torch.as_tensor(bases_dict["S"], device=device, dtype=dtype)
        
        if int_type == "attenuate":
            alpha = float(intervention.get("alpha", DEFAULT_ATTENUATE_ALPHA))
        elif int_type == "mix":
            perm_t = torch.as_tensor(intervention["perm"], device=device, dtype=torch.long)
            signs_t = torch.as_tensor(intervention["signs"], device=device, dtype=dtype)
        elif int_type in ("transplant", "transplant_within", "transplant_cross", "transplant_null"):
            donor_h_F_t = torch.as_tensor(intervention["donor_h_F"], device=device, dtype=dtype)
        
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not scramble_applied["done"]:
                h_t = h[0, last_pos, :].clone()
                
                # Compute F = h - proj_P(h) - proj_D(h) - proj_S(h)
                h_P = Bp_t @ (Bp_t.transpose(0, 1) @ h_t) if Bp_t.shape[1] > 0 else torch.zeros_like(h_t)
                h_D = Bd_t @ (Bd_t.transpose(0, 1) @ h_t) if Bd_t.shape[1] > 0 else torch.zeros_like(h_t)
                h_S = Bs_t @ (Bs_t.transpose(0, 1) @ h_t) if Bs_t.shape[1] > 0 else torch.zeros_like(h_t)
                h_F = h_t - h_P - h_D - h_S
                
                if int_type == "attenuate":
                    h_F_new = (1.0 - alpha) * h_F
                elif int_type == "mix":
                    # Mix: permute + sign-flip in model space
                    h_F_new = h_F[perm_t] * signs_t
                elif int_type in ("transplant", "transplant_within", "transplant_cross", "transplant_null"):
                    # All transplant variants: replace h_F with energy-matched donor h_F.
                    # Direction and magnitude were fixed at spec-build time.
                    h_F_new = donor_h_F_t
                else:
                    h_F_new = h_F
                
                h[0, last_pos, :] = h_P + h_D + h_S + h_F_new
                scramble_applied["done"] = True
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

    handle = model_layers[block_to_hook].register_forward_hook(hook_fn)

    collected: Dict[int, np.ndarray] = {}
    hooks = []
    for layer_idx in layers_to_collect:
        block_idx = layer_idx - 1
        if 0 <= block_idx < len(model_layers):
            def make_hook(lidx):
                def h(module, inp, out):
                    hid = out[0] if isinstance(out, tuple) else out
                    collected[lidx] = hid[0, last_pos, :].detach().cpu().float().numpy()
                return h
            hooks.append(model_layers[block_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
    logits = outputs.logits[0, -1, :].detach().cpu()

    handle.remove()
    for hk in hooks:
        hk.remove()

    return collected, logits


# ============================================================
# COMPONENT DIVERGENCE MEASUREMENT
# ============================================================

def compute_component_divergence(
    baseline: np.ndarray,
    scrambled: np.ndarray,
) -> Dict[str, float]:
    """
    Compute divergence metrics between baseline and scrambled component vectors.
    
    Generalized from compute_D_divergence to work for any PDSF component.
    Used by Part G to measure how each component (P, D, S, F) is affected
    by an intervention on any subspace.
    
    Args:
        baseline: Component vector from baseline forward pass [d_model]
        scrambled: Component vector from scrambled forward pass [d_model]
        
    Returns:
        Dict with divergence metrics:
        - cosine: cosine similarity (-1 to 1)
        - angle_deg: angle in degrees (0 to 180)
        - l2_distance: Euclidean distance
        - relative_change: ||scrambled - baseline|| / ||baseline||
    """
    norm_base = np.linalg.norm(baseline)
    norm_scram = np.linalg.norm(scrambled)
    
    # Handle zero vectors (e.g. P component when token not in vocab)
    if norm_base < 1e-10 or norm_scram < 1e-10:
        return {
            "cosine": 0.0,
            "angle_deg": 90.0,
            "l2_distance": float(np.linalg.norm(scrambled - baseline)),
            "relative_change": float('inf') if norm_base < 1e-10 else 0.0,
        }
    
    # Cosine similarity
    cosine = float(np.dot(baseline, scrambled) / (norm_base * norm_scram))
    cosine = np.clip(cosine, -1.0, 1.0)  # Numerical stability
    
    # Angle in degrees
    angle_deg = float(np.degrees(np.arccos(cosine)))
    
    # L2 distance
    l2_dist = float(np.linalg.norm(scrambled - baseline))
    
    # Relative change
    relative_change = l2_dist / norm_base
    
    return {
        "cosine": cosine,
        "angle_deg": angle_deg,
        "l2_distance": l2_dist,
        "relative_change": relative_change,
    }

compute_D_divergence = compute_component_divergence  # Backward-compatible alias


# ============================================================
# INVARIANCE CONTROL EXPERIMENT
# ============================================================

def run_invariance_control(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    layers: List[int],
    device: torch.device,
    verbose: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Verify that D trajectory is deterministic (invariant under repeated runs).
    
    This establishes the baseline: if D changes without intervention,
    something is wrong. D should be identical for the same prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to test
        layers: Layer indices to collect
        device: Device for computation
        verbose: Print summary statistics
        show_progress: Show progress bar
        
    Returns:
        Dict with invariance check results
    """
    results = {
        "n_prompts": len(prompts),
        "layers": layers,
        "max_D_diff": 0.0,
        "mean_D_diff": 0.0,
        "all_identical": True,
        "per_prompt": [],
    }
    
    if verbose:
        print("Running invariance control (verifying determinism)...")
    
    # Setup progress bar
    prompt_iter = enumerate(prompts)
    if show_progress:
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(enumerate(prompts), total=len(prompts), desc="Invariance check")
        except ImportError:
            pass

    for idx, prompt in prompt_iter:
        # Run twice with identical inputs
        H1, logits1 = run_forward_with_collection(model, tokenizer, prompt, layers, device)
        H2, logits2 = run_forward_with_collection(model, tokenizer, prompt, layers, device)
        
        # Compare hidden states at each layer
        prompt_diffs = []
        for layer in layers:
            if layer in H1 and layer in H2:
                diff = np.linalg.norm(H1[layer] - H2[layer])
                prompt_diffs.append(diff)
                if diff > 1e-6:
                    results["all_identical"] = False
        
        max_diff = max(prompt_diffs) if prompt_diffs else 0.0
        results["per_prompt"].append({"prompt_idx": idx, "max_diff": max_diff})
        results["max_D_diff"] = max(results["max_D_diff"], max_diff)
    
    # Compute mean
    all_diffs = [p["max_diff"] for p in results["per_prompt"]]
    results["mean_D_diff"] = float(np.mean(all_diffs)) if all_diffs else 0.0
    
    if verbose:
        status = "PASSED" if results["all_identical"] else "FAILED"
        print(f"Invariance control: {status}")
        print(f"  Max difference: {results['max_D_diff']:.2e}")
        print(f"  Mean difference: {results['mean_D_diff']:.2e}")
    
    return results


# ============================================================
# MAIN GEOMETRY INTERVENTION EXPERIMENT
# ============================================================

def run_scramble_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    expected_tokens: List[str],
    group_ids: List[str],
    H_cached: Dict[int, np.ndarray],
    layers: List[int],
    scramble_layer: int,
    device: torch.device,
    regime_ids: Optional[List[str]] = None,
    n_scrambles: int = 3,
    s_rank: int = 13,
    verbose: bool = True,
    show_progress: bool = True,
    seed: int = 42,
    compute_kl: bool = False,
    kl_topk: int = 2048,
    subspaces: List[str] = None,
    intervention_types: List[str] = None,
    attenuate_alpha: float = 0.5,
    compute_spirality: bool = False,
    spirality_n_pc_pairs: int = 3,
) -> Dict[str, Any]:
    """
    Run the Part G geometry intervention experiment.

    Tests all specified subspaces with subspace-appropriate intervention types
    at an early layer, measuring D divergence at subsequent layers and logit impact.

    Intervention dispatch:
      P, D, S (low-rank): attenuate, rotation, transplant
      F (high-rank):      attenuate, mix, transplant

    Args:
        subspaces: List of subspaces to test (default: ["P", "D", "S", "F"])
        intervention_types: List of requested intervention types. The function
            filters to valid types per subspace automatically.
        attenuate_alpha: Scaling factor for attenuate intervention (default: 0.5)
        n_scrambles: Number of random instances per condition per prompt (for rotation/mix)
        compute_kl: If True, compute KL divergence (expensive)
        kl_topk: Top-k for approximate KL
        compute_spirality: If True, compute Part M spirality measures on baseline
            and intervened trajectories (piggybacks on existing forward passes).
        spirality_n_pc_pairs: Number of PC pairs for phase/winding analysis.

    Returns:
        Dict with results nested by subspace × intervention type
    """
    if subspaces is None:
        subspaces = list(GEOMETRY_SUBSPACES)
    if intervention_types is None:
        # Union of both sets as default
        intervention_types = sorted(set(GEOMETRY_INTERVENTION_TYPES_PDS) | set(GEOMETRY_INTERVENTION_TYPES_F))

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print("\n" + "=" * 60)
        print(f"GEOMETRY INTERVENTION EXPERIMENT (layer {scramble_layer})")
        print(f"  Subspaces: {subspaces}")
        print(f"  Interventions: {intervention_types}")
        print(f"  n_scrambles: {n_scrambles}, attenuate_alpha: {attenuate_alpha}")
        print("=" * 60)

    from datetime import datetime, timezone

    # Part M spirality measures (piggyback on Part G forward passes)
    _spirality_available = False
    if compute_spirality:
        try:
            from pds_spirality import (
                compute_spirality_profile as _compute_spirality_profile,
                compute_spirality_disruption as _compute_spirality_disruption,
                aggregate_spirality_disruptions as _aggregate_spirality_disruptions,
            )
            _spirality_available = True
            if verbose:
                print("  Part M spirality measures: enabled")
        except ImportError:
            if verbose:
                print("  Part M spirality measures: pds_spirality not available, skipping")

    metadata = {
        "experiment_name": "Geometry_PDSF_Intervention",
        "experiment_version": "5.2",  # spirality measures
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "scramble_layer": int(scramble_layer),
        "n_scrambles": int(n_scrambles),
        "s_rank": int(s_rank),
        "seed": int(seed),
        "n_prompts": len(prompts),
        "n_layers_measured": len(layers),
        "subspaces": subspaces,
        "intervention_types": intervention_types,
        "attenuate_alpha": attenuate_alpha,
        "compute_kl": compute_kl,
        "compute_spirality": compute_spirality and _spirality_available,
        "spirality_n_pc_pairs": spirality_n_pc_pairs if _spirality_available else None,
        # True only if a regime-aware F transplant condition was actually requested.
        # Previously this reported whether regime ids were *supplied*, so a run that
        # dropped the transplant conditions still advertised transplant data.
        "has_regime_transplant": bool(
            regime_ids and any(regime_ids)
            and any(str(t).startswith("transplant") for t in (intervention_types or []))
        ),
        "regime_ids_provided": regime_ids is not None,
    }

    # ----------------------------
    # Bases at scramble layer (use pre-computed unified bases)
    # ----------------------------
    # P/D/S bases are computed once by the pipeline using the canonical
    # sequential decomposition (P→D→S) from pds_continuation.py.
    # Part G receives them as parameters rather than computing its own.
    
    from pds_continuation import compute_P_bases_from_predictions, compute_D_basis_global as _compute_D, compute_S_basis_global as _compute_S
    
    H_scramble = H_cached[scramble_layer]
    
    # Build per-prompt P bases from predicted tokens
    if verbose:
        print("\nUsing unified PDS bases...")
    
    # Get predicted token IDs — use expected_tokens if they are token IDs (int),
    # otherwise look them up
    pred_token_ids = []
    token_id_by_prompt: Dict[int, Optional[int]] = {}
    for i, tok in enumerate(expected_tokens):
        if isinstance(tok, int):
            # Already a token ID
            pred_token_ids.append(tok)
            token_id_by_prompt[i] = tok
        elif tok and tok.strip():
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(" " + tok.strip(), add_special_tokens=False)
            tid = ids[0] if ids else None
            pred_token_ids.append(tid if tid is not None else 0)
            token_id_by_prompt[i] = tid
        else:
            pred_token_ids.append(0)
            token_id_by_prompt[i] = None
    
    # Get unembedding matrix (meta-safe)
    _unembed_w = get_unembedding_matrix(model)
    unembed_np = _unembed_w.cpu().float().numpy()
    
    # Compute unified P bases
    P_bases, P_info = compute_P_bases_from_predictions(pred_token_ids, unembed_np)
    
    # Compute unified D basis (at scramble layer, using fixed P)
    D_basis, D_info = _compute_D(H_scramble, P_bases, k_D=min(s_rank, 12))
    
    # Compute unified S basis (at scramble layer, after P and D)
    S_basis, S_info = _compute_S(H_scramble, P_bases, D_basis, k_S=s_rank)
    
    if verbose:
        print(f"  P: rank-1 per prompt ({P_info['n_unique_tokens']} unique tokens)")
        print(f"  D: rank={D_info['k']}, PR={D_info['pr']:.2f}")
        print(f"  S: rank={S_info['k']}, PR={S_info['pr']:.2f}")
    
    # Use fixed scramble-layer bases for all subsequent layers
    # (Part G measures how perturbations propagate, not how bases change)
    Bs_scramble = S_basis
    
    layers_after_scramble = [l for l in sorted(layers) if l >= scramble_layer]

    # KL helper
    def approx_kl_topk(logits_base, logits_scram, topk):
        if topk is None or topk <= 0:
            return float("nan")
        k = min(int(topk), logits_base.numel())
        idx_b = torch.topk(logits_base, k=k).indices
        idx_s = torch.topk(logits_scram, k=k).indices
        idx = torch.unique(torch.cat([idx_b, idx_s], dim=0))
        lb, ls = logits_base[idx], logits_scram[idx]
        logp = torch.log_softmax(lb, dim=0)
        logq = torch.log_softmax(ls, dim=0)
        p = torch.exp(logp)
        return float(torch.sum(p * (logp - logq)).item())

    # Entropy helper
    def topk_entropy(logits, topk):
        """Compute entropy of the top-k softmax distribution.
        Low entropy = peaked/confident prediction. High entropy = flat/gibberish.
        Max entropy for k tokens = log(k) ~ 7.6 for k=2048."""
        if topk is None or topk <= 0:
            return float("nan")
        k = min(int(topk), logits.numel())
        topk_vals = torch.topk(logits, k=k).values
        logp = torch.log_softmax(topk_vals, dim=0)
        p = torch.exp(logp)
        entropy = -float(torch.sum(p * logp).item())
        return entropy

    # ----------------------------
    # Pre-generate interventions
    # ----------------------------
    if verbose:
        print("Generating intervention specs...")

    rng = np.random.RandomState(seed)
    intervention_specs: Dict[str, Dict[str, List[Dict]]] = {}

    # --- Build transplant donor mappings (regime/domain-aware) ---
    # Strategy:
    #   transplant_donor_idx        : original round-robin cross-group (used by P/D/S)
    #   transplant_within_donor_idx : same regime/domain, different group  (F transplant_within)
    #   transplant_cross_donor_idx  : different regime/domain, near (+1)    (F transplant_cross)
    #   transplant_null_donor_idx   : NULL_DONOR_IDX for every prompt        (F transplant_null)

    unique_groups = sorted(set(group_ids))
    group_to_indices: Dict[str, List[int]] = {}
    for i, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(i)

    # ── Legacy round-robin donor (P/D/S, unchanged behaviour) ────────────────
    transplant_donor_idx: Dict[int, int] = {}
    for i, gid in enumerate(group_ids):
        other_groups = [g for g in unique_groups if g != gid]
        if not other_groups:
            same = [j for j in group_to_indices[gid] if j != i]
            transplant_donor_idx[i] = same[i % len(same)] if same else i
        else:
            donor_group = other_groups[i % len(other_groups)]
            donor_candidates = group_to_indices[donor_group]
            transplant_donor_idx[i] = donor_candidates[i % len(donor_candidates)]

    # ── Auto-detect regime/domain source ─────────────────────────────────────
    has_regime = regime_ids is not None and any(r for r in regime_ids)
    if has_regime:
        effective_domain_ids = regime_ids
        transplant_type_within = "within_regime"
        transplant_type_cross  = "cross_regime"
    else:
        # Infer domains from group IDs using hardcoded prompt-set maps.
        # Priority: SpecA -> SpecB -> fall back to group_id as its own domain.
        effective_domain_ids = [
            SPECA_GROUP_TO_DOMAIN.get(gid) or SPECB_GROUP_TO_DOMAIN.get(gid) or gid
            for gid in group_ids
        ]
        transplant_type_within = "within_domain"
        transplant_type_cross  = "cross_domain"

    # Build domain -> sorted group_ids and domain -> sorted indices
    domain_to_groups: Dict[str, List[str]] = {}
    for gid, dom in zip(group_ids, effective_domain_ids):
        domain_to_groups.setdefault(dom, [])
        if gid not in domain_to_groups[dom]:
            domain_to_groups[dom].append(gid)
    domain_list = sorted(domain_to_groups.keys())
    for dom in domain_list:
        domain_to_groups[dom] = sorted(domain_to_groups[dom])

    group_to_domain: Dict[str, str] = {gid: dom for gid, dom in zip(group_ids, effective_domain_ids)}

    # ── Within-regime/domain donor ────────────────────────────────────────────
    # Donor: first other group in same domain (alphabetically), position-matched.
    # If only 1 group in domain (fallback): use cross-domain donor.
    transplant_within_donor_idx: Dict[int, int] = {}
    for i, gid in enumerate(group_ids):
        dom = group_to_domain[gid]
        same_dom_groups = [g for g in domain_to_groups[dom] if g != gid]
        if same_dom_groups:
            donor_gid = same_dom_groups[0]
            donor_cands = group_to_indices[donor_gid]
            local_j = group_to_indices[gid].index(i) if i in group_to_indices[gid] else 0
            transplant_within_donor_idx[i] = donor_cands[min(local_j, len(donor_cands) - 1)]
        else:
            # Fallback: will be filled in after cross-domain map is built
            transplant_within_donor_idx[i] = None  # type: ignore

    # ── Cross-regime/domain donor ─────────────────────────────────────────────
    # Donor: domain at index (di+1) % n_domains, first group in that domain, position-matched.
    n_domains = len(domain_list)
    transplant_cross_donor_idx: Dict[int, int] = {}
    for i, gid in enumerate(group_ids):
        dom = group_to_domain[gid]
        di = domain_list.index(dom)
        cross_dom = domain_list[(di + 1) % n_domains]
        cross_groups = domain_to_groups[cross_dom]
        donor_gid = cross_groups[0]
        donor_cands = group_to_indices[donor_gid]
        local_j = group_to_indices[gid].index(i) if i in group_to_indices[gid] else 0
        transplant_cross_donor_idx[i] = donor_cands[min(local_j, len(donor_cands) - 1)]

    # Back-fill within fallbacks with cross donor
    for i in range(len(prompts)):
        if transplant_within_donor_idx[i] is None:
            transplant_within_donor_idx[i] = transplant_cross_donor_idx[i]

    # ── Null-state donor: same sentinel for every recipient ───────────────────
    transplant_null_donor_idx: Dict[int, int] = {i: NULL_DONOR_IDX for i in range(len(prompts))}

    # ── Pre-decompose null donor (single space " ") ───────────────────────────
    # Extracted once; reused for every recipient via energy-matching.
    if verbose:
        print("Pre-computing transplant donor components (including null-state donor)...")
    transplant_donors: Dict[int, Dict[str, Any]] = {}

    _null_prompt = " "
    try:
        H_null_map, _ = run_forward_with_collection(
            model, tokenizer, _null_prompt, [scramble_layer], device
        )
        h_null = H_null_map[scramble_layer]
        # h_null may be shape (1, d_model) or (d_model,) depending on implementation
        if h_null.ndim == 2:
            h_null = h_null[0]
        h_null = h_null.astype(np.float32) if isinstance(h_null, np.ndarray) else h_null.cpu().float().numpy()
        # Derive a P basis for the null prompt from its predicted token
        _null_toks = tokenizer.encode(_null_prompt, add_special_tokens=True)
        _null_pred_id = _null_toks[-1] if _null_toks else 0
        _null_p_vec = unembed_np[_null_pred_id].astype(np.float32)
        _null_p_norm = float(np.linalg.norm(_null_p_vec))
        if _null_p_norm > 1e-12:
            _null_p_vec = _null_p_vec / _null_p_norm
        Bp_null = _null_p_vec.reshape(-1, 1)  # (d_model, 1)
        dc_null = decompose_hidden_state(h_null, Bp_null, D_basis, S_basis)
        transplant_donors[NULL_DONOR_IDX] = {
            "P_coeffs": (Bp_null.T @ dc_null.h_P).flatten(),
            "D_coeffs": (D_basis.T @ dc_null.h_D).flatten() if D_basis.shape[1] > 0 else np.array([], dtype=np.float32),
            "S_coeffs": (Bs_scramble.T @ dc_null.h_S).flatten(),
            "h_F": dc_null.h_F,
            "energy_P": float(np.linalg.norm(dc_null.h_P)),
            "energy_D": float(np.linalg.norm(dc_null.h_D)),
            "energy_S": float(np.linalg.norm(dc_null.h_S)),
            "energy_F": float(np.linalg.norm(dc_null.h_F)),
            "group_id": "__null__",
            "regime":   "__null__",
        }
        if verbose:
            nf = transplant_donors[NULL_DONOR_IDX]["energy_F"]
            print(f"  Null donor extracted — F energy: {nf:.4f}")
    except Exception as _e:
        if verbose:
            print(f"  WARNING: null donor extraction failed ({_e}); transplant_null will be skipped")
        transplant_donors[NULL_DONOR_IDX] = None  # type: ignore

    # ── Pre-decompose all substantive donors (union of all three maps) ────────
    all_donor_indices = (
        set(transplant_donor_idx.values()) |
        set(transplant_within_donor_idx.values()) |
        set(transplant_cross_donor_idx.values())
    )
    all_donor_indices.discard(NULL_DONOR_IDX)

    for donor_idx in sorted(all_donor_indices):
        if donor_idx in transplant_donors:
            continue
        h_donor = H_scramble[donor_idx]
        Bp_donor = P_bases.get(donor_idx)
        if Bp_donor is None:
            transplant_donors[donor_idx] = None
            continue
        dc_donor = decompose_hidden_state(h_donor, Bp_donor, D_basis, S_basis)
        transplant_donors[donor_idx] = {
            "P_coeffs": (Bp_donor.T @ dc_donor.h_P).flatten(),
            "D_coeffs": (D_basis.T @ dc_donor.h_D).flatten() if D_basis.shape[1] > 0 else np.array([], dtype=np.float32),
            "S_coeffs": (Bs_scramble.T @ dc_donor.h_S).flatten(),
            "h_F": dc_donor.h_F,
            "energy_P": float(np.linalg.norm(dc_donor.h_P)),
            "energy_D": float(np.linalg.norm(dc_donor.h_D)),
            "energy_S": float(np.linalg.norm(dc_donor.h_S)),
            "energy_F": float(np.linalg.norm(dc_donor.h_F)),
            "group_id": group_ids[donor_idx],
            "regime":   regime_ids[donor_idx] if regime_ids else None,
        }


    for subspace in subspaces:
        # Filter intervention types to valid set for this subspace
        if subspace == "F":
            sub_int_types = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_F]
        else:
            sub_int_types = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_PDS]

        intervention_specs[subspace] = {}
        if subspace == "F":
            rank_for_intervention = H_scramble.shape[1]  # d_model for F
        elif subspace == "S":
            rank_for_intervention = Bs_scramble.shape[1]
        else:
            # P and D ranks vary per prompt; generate per-prompt later
            rank_for_intervention = None

        for int_type in sub_int_types:
            specs = []
            if rank_for_intervention is None:
                # P/D: rank varies per prompt — leave specs empty,
                # per-prompt generation will handle it in the main loop
                pass
            else:
                if int_type == "attenuate":
                    specs.append({"type": "attenuate", "alpha": attenuate_alpha})
                    # Attenuate is deterministic — one instance
                elif int_type in ("transplant", "transplant_within", "transplant_cross", "transplant_null"):
                    pass  # All transplant variants are per-prompt; handled in main loop
                else:
                    for si in range(n_scrambles):
                        if int_type == "rotation":
                            R, _ = np.linalg.qr(rng.randn(rank_for_intervention, rank_for_intervention).astype(np.float32))
                            specs.append({"type": "rotation", "R": R})
                        elif int_type == "mix":
                            perm = rng.permutation(rank_for_intervention).astype(np.int64)
                            signs = rng.choice([-1.0, 1.0], size=rank_for_intervention).astype(np.float32)
                            specs.append({"type": "mix", "perm": perm, "signs": signs})
            intervention_specs[subspace][int_type] = specs

    # ----------------------------
    # Results structure
    # ----------------------------
    results: Dict[str, Any] = {
        "metadata": metadata,
        "scramble_layer": scramble_layer,
        "layers": layers,
        "layers_after_scramble": layers_after_scramble,
        "by_prompt": [],
        "aggregate": {},
    }

    # Measured components — divergence of each PDSF component is tracked
    # after every intervention, not just D. This enables cross-subspace coherence
    # analysis (do interventions disrupt all subspaces proportionally?).
    measured_components = ["P", "D", "S", "F"]

    # Aggregate accumulators — only for valid intervention types per subspace
    # Structure: agg[intervened_subspace][intervention_type][layer]["{component}_{metric}"] = [float, ...]
    agg: Dict[str, Dict[str, Dict[int, Dict[str, List[float]]]]] = {}
    for sub in subspaces:
        agg[sub] = {}
        if sub == "F":
            sub_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_F]
        else:
            sub_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_PDS]
        for it in sub_its:
            agg[sub][it] = {}
            for l in layers_after_scramble:
                agg[sub][it][l] = {}
                for mc in measured_components:
                    agg[sub][it][l][f"{mc}_angle_deg"] = []
                    agg[sub][it][l][f"{mc}_relative_change"] = []

    # ----------------------------
    # Iterate prompts
    # ----------------------------
    prompt_iter = list(enumerate(zip(prompts, expected_tokens)))
    if show_progress:
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(prompt_iter, total=len(prompts), desc="Part G interventions")
        except ImportError:
            pass

    for prompt_idx, (prompt, expected_token) in prompt_iter:
        Bp = P_bases.get(prompt_idx)
        if Bp is None:
            continue

        expected_token_id = token_id_by_prompt.get(prompt_idx)

        # Baseline
        H_baseline, logits_baseline = run_forward_with_collection(
            model, tokenizer, prompt, layers_after_scramble, device
        )

        # Store baseline decomposition for ALL four components at every layer.
        # Previously only h_D was stored. We need P, D, S, F baselines to measure
        # how each component is affected by interventions on any subspace.
        baseline_components_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
        for layer in layers_after_scramble:
            h = H_baseline[layer]
            # Use fixed scramble-layer bases (Part G tracks perturbation propagation)
            decomp = decompose_hidden_state(h, Bp, D_basis, S_basis)
            baseline_components_by_layer[layer] = {
                "P": decomp.h_P,
                "D": decomp.h_D,
                "S": decomp.h_S,
                "F": decomp.h_F,
            }

        prompt_result: Dict[str, Any] = {
            "prompt_idx": prompt_idx,
            "expected_token": expected_token,
            "baseline_logit": float(logits_baseline[expected_token_id]) if expected_token_id is not None else None,
            "by_subspace": {},
        }

        # Part M: Compute baseline spirality profile
        baseline_spirality = None
        if _spirality_available:
            baseline_spirality = _compute_spirality_profile(
                H_baseline, layers_after_scramble,
                n_pc_pairs=spirality_n_pc_pairs,
                include_full_spectrum=False,
            )
            prompt_result["spirality_baseline"] = baseline_spirality.get("summary")

        # Compute baseline entropy (once per prompt)
        baseline_entropy = None
        if compute_kl and logits_baseline is not None:
            baseline_entropy = topk_entropy(logits_baseline.float(), kl_topk)
            prompt_result["baseline_entropy_topk"] = baseline_entropy

        # For each subspace × intervention type
        for subspace in subspaces:
            prompt_result["by_subspace"][subspace] = {"by_intervention": {}}

            # Determine basis for this subspace
            if subspace == "P":
                sub_basis = Bp
            elif subspace == "D":
                sub_basis = D_basis
            elif subspace == "S":
                sub_basis = Bs_scramble
            elif subspace == "F":
                # For F, pass dict of P/D/S bases
                sub_basis = {"P": Bp, "D": D_basis, "S": Bs_scramble}

            # Filter to valid intervention types for this subspace
            if subspace == "F":
                sub_int_types = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_F]
            else:
                sub_int_types = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_PDS]

            for int_type in sub_int_types:
                specs = intervention_specs[subspace].get(int_type, [])
                scramble_results = []

                # Generate per-prompt specs for P/D if needed (rank varies per prompt)
                if subspace in ("P", "D") and not specs:
                    rank = sub_basis.shape[1]
                    if rank == 0:
                        continue
                    per_prompt_specs = []
                    for si in range(n_scrambles if int_type not in ("attenuate", "transplant") else 1):
                        if int_type == "rotation":
                            R, _ = np.linalg.qr(rng.randn(rank, rank).astype(np.float32))
                            per_prompt_specs.append({"type": "rotation", "R": R})
                        elif int_type == "attenuate":
                            per_prompt_specs.append({"type": "attenuate", "alpha": attenuate_alpha})
                        elif int_type == "transplant":
                            pass  # Handled below
                    specs = per_prompt_specs

                # Generate transplant specs for this prompt (per-type donor selection)
                # All four cases share the same energy-matched construction;
                # they differ only in donor map and output metadata.
                _transplant_map = {
                    "transplant":       (transplant_donor_idx,       "cross_group"),
                    "transplant_within":(transplant_within_donor_idx, transplant_type_within),
                    "transplant_cross": (transplant_cross_donor_idx,  transplant_type_cross),
                    "transplant_null":  (transplant_null_donor_idx,   "null_state"),
                }
                if int_type in _transplant_map and not specs:
                    _donor_map, _txp_type = _transplant_map[int_type]
                    donor_idx = _donor_map.get(prompt_idx)
                    donor_data = transplant_donors.get(donor_idx) if donor_idx is not None else None
                    if donor_data is not None:
                        h_recip = H_scramble[prompt_idx]
                        if subspace == "F":
                            dc_recip = decompose_hidden_state(h_recip, Bp, D_basis, S_basis)
                            recip_norm = float(np.linalg.norm(dc_recip.h_F))
                            donor_h_F = donor_data["h_F"].copy()
                            donor_norm = donor_data["energy_F"]
                            if donor_norm > 1e-12 and recip_norm > 1e-12:
                                donor_h_F = donor_h_F * (recip_norm / donor_norm)
                            specs = [{
                                "type": int_type,
                                "donor_h_F": donor_h_F.astype(np.float32),
                                "donor_idx": int(donor_idx) if donor_idx != NULL_DONOR_IDX else None,
                                "donor_group_id": donor_data.get("group_id"),
                                "donor_regime":   donor_data.get("regime"),
                                "transplant_type": _txp_type,
                            }]
                        elif int_type == "transplant":
                            # P/D/S: only the legacy "transplant" type applies
                            sub_key    = {"P": "P_coeffs", "D": "D_coeffs", "S": "S_coeffs"}[subspace]
                            energy_key = {"P": "energy_P",  "D": "energy_D",  "S": "energy_S"}[subspace]
                            donor_coeffs = donor_data[sub_key].copy()
                            donor_norm   = donor_data[energy_key]
                            c_recip    = (sub_basis.T @ h_recip).flatten()
                            recip_norm = float(np.linalg.norm(sub_basis @ c_recip))
                            if donor_norm > 1e-12 and recip_norm > 1e-12:
                                donor_coeffs = donor_coeffs * (recip_norm / donor_norm)
                            if len(donor_coeffs) != sub_basis.shape[1]:
                                specs = []
                            else:
                                specs = [{
                                    "type": "transplant",
                                    "donor_coefficients": donor_coeffs.astype(np.float32),
                                    "donor_idx": int(donor_idx),
                                    "donor_group_id": donor_data.get("group_id"),
                                    "donor_regime":   donor_data.get("regime"),
                                    "transplant_type": _txp_type,
                                }]

                for spec in specs:
                    H_scrambled, logits_scrambled = run_forward_with_intervention(
                        model, tokenizer, prompt,
                        scramble_layer, sub_basis, spec,
                        layers_after_scramble, device,
                        subspace_name=subspace,
                    )

                    scr_result: Dict[str, Any] = {"by_layer": {}}

                    for layer in layers_after_scramble:
                        if layer not in H_scrambled:
                            continue
                        h_scram = H_scrambled[layer]
                        # Use fixed scramble-layer bases at all layers
                        decomp_scram = decompose_hidden_state(h_scram, Bp, D_basis, S_basis)

                        # Compare ALL four components to their baselines
                        baselines = baseline_components_by_layer[layer]
                        scrambled_comps = {
                            "P": decomp_scram.h_P,
                            "D": decomp_scram.h_D,
                            "S": decomp_scram.h_S,
                            "F": decomp_scram.h_F,
                        }

                        layer_divergence = {}
                        for mc in measured_components:
                            div = compute_component_divergence(baselines[mc], scrambled_comps[mc])
                            layer_divergence[mc] = div
                            agg[subspace][int_type][layer][f"{mc}_angle_deg"].append(div["angle_deg"])
                            agg[subspace][int_type][layer][f"{mc}_relative_change"].append(div["relative_change"])

                        # backward compat: keep top-level angle_deg/relative_change as D values
                        # Backward compat: keep top-level angle_deg/relative_change as D values
                        # continues to work without modification.
                        layer_divergence["angle_deg"] = layer_divergence["D"]["angle_deg"]
                        layer_divergence["relative_change"] = layer_divergence["D"]["relative_change"]
                        scr_result["by_layer"][layer] = layer_divergence

                    # Part M: Compute spirality on intervened trajectory
                    if _spirality_available and baseline_spirality is not None:
                        intervened_spirality = _compute_spirality_profile(
                            H_scrambled, layers_after_scramble,
                            n_pc_pairs=spirality_n_pc_pairs,
                            include_full_spectrum=False,
                        )
                        spirality_disruption = _compute_spirality_disruption(
                            baseline_spirality, intervened_spirality,
                        )
                        scr_result["spirality_intervened"] = intervened_spirality.get("summary")
                        scr_result["spirality_disruption"] = spirality_disruption

                    # Logit metrics
                    if expected_token_id is not None:
                        scrambled_logit = float(logits_scrambled[expected_token_id])
                        scr_result["scrambled_logit"] = scrambled_logit
                        scr_result["logit_diff"] = scrambled_logit - prompt_result["baseline_logit"]

                        if compute_kl:
                            kl = approx_kl_topk(logits_baseline.float(), logits_scrambled.float(), kl_topk)
                            scr_result["kl_divergence_topk"] = kl
                            scr_entropy = topk_entropy(logits_scrambled.float(), kl_topk)
                            scr_result["entropy_topk"] = scr_entropy
                            scr_result["entropy_ratio"] = scr_entropy / baseline_entropy if baseline_entropy and baseline_entropy > 0 else float("nan")

                    # Copy transplant donor metadata into the result record
                    if int_type in ("transplant", "transplant_within", "transplant_cross", "transplant_null"):
                        for _field in ("donor_group_id", "donor_regime", "transplant_type"):
                            if _field in spec:
                                scr_result[_field] = spec[_field]
                        if "donor_idx" in spec:
                            scr_result["donor_prompt_idx"] = spec["donor_idx"]

                    scramble_results.append(scr_result)

                prompt_result["by_subspace"][subspace]["by_intervention"][int_type] = {
                    "scrambles": scramble_results
                }

        results["by_prompt"].append(prompt_result)

    # ----------------------------
    # Aggregate statistics
    # ----------------------------
    for sub in subspaces:
        results["aggregate"][sub] = {}
        if sub == "F":
            sub_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_F]
        else:
            sub_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_PDS]
        for it in sub_its:
            if it not in agg[sub]:
                continue
            results["aggregate"][sub][it] = {}
            for layer in layers_after_scramble:
                layer_agg = {}
                # Per-component aggregate stats (P, D, S, F)
                for mc in measured_components:
                    angles = agg[sub][it][layer].get(f"{mc}_angle_deg", [])
                    rel_changes = agg[sub][it][layer].get(f"{mc}_relative_change", [])
                    layer_agg[f"{mc}_angle_deg_mean"] = float(np.mean(angles)) if angles else None
                    layer_agg[f"{mc}_angle_deg_std"] = float(np.std(angles)) if angles else None
                    layer_agg[f"{mc}_relative_change_mean"] = float(np.mean(rel_changes)) if rel_changes else None
                    layer_agg[f"{mc}_relative_change_std"] = float(np.std(rel_changes)) if rel_changes else None
                layer_agg["n_samples"] = len(agg[sub][it][layer].get("D_angle_deg", []))

                # Cross-subspace coherence — Pearson correlation of disruption
                # angles across prompts between each pair of measured components.
                # Tests whether F-scramble disrupts all subspaces proportionally
                # (unified framework → high correlation) or independently
                # (compartmentalized → low correlation).
                coherence = {}
                for i_mc, mc1 in enumerate(measured_components):
                    for mc2 in measured_components[i_mc + 1:]:
                        a1 = agg[sub][it][layer].get(f"{mc1}_angle_deg", [])
                        a2 = agg[sub][it][layer].get(f"{mc2}_angle_deg", [])
                        if len(a1) >= 3 and len(a2) >= 3:
                            # Guard against near-constant distributions
                            # (e.g. transplant_null injects same F vector to all prompts,
                            # making one distribution constant → corrcoef returns nan).
                            std1, std2 = float(np.std(a1)), float(np.std(a2))
                            if std1 < 1e-6 or std2 < 1e-6:
                                coherence[f"{mc1}_{mc2}_corr"] = None
                                coherence[f"{mc1}_{mc2}_corr_note"] = "near_constant"
                            else:
                                corr = float(np.corrcoef(a1, a2)[0, 1])
                                coherence[f"{mc1}_{mc2}_corr"] = round(corr, 4) if np.isfinite(corr) else None
                layer_agg["cross_subspace_coherence"] = coherence
                # Hoist priority pairs to top level for analysis ergonomics
                # (avoids nested key navigation in downstream scripts)
                for _pair in ("D_F_corr", "S_F_corr", "D_S_corr"):
                    layer_agg[_pair] = coherence.get(_pair)

                results["aggregate"][sub][it][layer] = layer_agg

            # Part M: Aggregate spirality disruptions per (sub × intervention)
            if _spirality_available:
                spirality_disruptions = []
                for pr in results["by_prompt"]:
                    by_int = pr.get("by_subspace", {}).get(sub, {}).get("by_intervention", {}).get(it, {})
                    for scr in by_int.get("scrambles", []):
                        sd = scr.get("spirality_disruption")
                        if sd is not None:
                            spirality_disruptions.append(sd)
                if spirality_disruptions:
                    results["aggregate"][sub][it]["spirality_disruption_agg"] = (
                        _aggregate_spirality_disruptions(spirality_disruptions)
                    )

    if verbose:
        # Show all four component angles + cross-subspace coherence
        print("\nAggregate PDSF divergence after intervention:")
        for sub in subspaces:
            for it in results["aggregate"].get(sub, {}):
                last_layer = layers_after_scramble[-1] if layers_after_scramble else None
                if last_layer and last_layer in results["aggregate"][sub][it]:
                    a = results["aggregate"][sub][it][last_layer]
                    # Per-component angle summary
                    parts = []
                    for mc in measured_components:
                        val = a.get(f"{mc}_angle_deg_mean")
                        if val is not None:
                            parts.append(f"{mc}={val:5.1f}°")
                    print(f"  {sub}-{it:18s} final: {', '.join(parts)}")
                    # Cross-subspace coherence summary (key correlations)
                    coh = a.get("cross_subspace_coherence")
                    if coh is None:
                        print(f"    {'':14s} coherence: MISSING")
                    elif not any(v is not None for v in coh.values()
                                 if not isinstance(v, str)):
                        n = a.get("n_samples", 0)
                        near_const = any(
                            str(v) == "near_constant"
                            for v in coh.values() if isinstance(v, str)
                        )
                        reason = "near_constant distribution" if near_const else f"n={n} < 3"
                        print(f"    {'':14s} coherence: N/A ({reason})")
                    else:
                        coh_str = ", ".join(
                            f"{k}={v:.3f}" for k, v in sorted(coh.items())
                            if v is not None and not isinstance(v, str)
                        )
                        print(f"    {'':14s} coherence: {coh_str}")

    # Coherence completeness assertion
    # Warns if any expected sub×it×layer entry has no coherence despite n>=3 samples.
    # Guards against future accumulator bugs (early continue, wrong list key, etc.)
    _exp_f_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_F]
    _exp_pds_its = [t for t in intervention_types if t in GEOMETRY_INTERVENTION_TYPES_PDS]
    _missing_coh = []
    for _sub in subspaces:
        _sub_its = _exp_f_its if _sub == "F" else _exp_pds_its
        for _it in _sub_its:
            for _layer in layers_after_scramble:
                _lagg = results["aggregate"].get(_sub, {}).get(_it, {}).get(_layer, {})
                if _lagg.get("cross_subspace_coherence") is None and _lagg.get("n_samples", 0) >= 3:
                    _missing_coh.append(f"{_sub}-{_it}@L{_layer}(n={_lagg.get('n_samples')})")
    if _missing_coh:
        print(f"\n  WARNING: cross_subspace_coherence MISSING for: {_missing_coh}")
    elif verbose:
        _n_coh_entries = sum(
            1
            for _s in subspaces
            for _i in results["aggregate"].get(_s, {})
            for _l in layers_after_scramble
            if results["aggregate"][_s][_i].get(_l, {}).get("cross_subspace_coherence") is not None
        )
        print(f"  \u2713 cross_subspace_coherence populated for {_n_coh_entries} sub\u00d7it\u00d7layer entries")

    # Aggregate KL and entropy across prompts (per subspace x intervention type)
    if compute_kl:
        for sub in subspaces:
            if sub not in results["aggregate"]:
                continue
            for it in results["aggregate"][sub]:
                kl_vals = []
                entropy_vals = []
                entropy_ratio_vals = []
                for pr in results["by_prompt"]:
                    by_int = pr.get("by_subspace", {}).get(sub, {}).get("by_intervention", {}).get(it, {})
                    for scr in by_int.get("scrambles", []):
                        if "kl_divergence_topk" in scr:
                            kl_vals.append(scr["kl_divergence_topk"])
                        if "entropy_topk" in scr:
                            entropy_vals.append(scr["entropy_topk"])
                        if "entropy_ratio" in scr and not (isinstance(scr["entropy_ratio"], float) and np.isnan(scr["entropy_ratio"])):
                            entropy_ratio_vals.append(scr["entropy_ratio"])
                # Store in a top-level aggregate key (not per-layer since KL/entropy are single-layer)
                agg_entry = results["aggregate"][sub][it]
                agg_entry["kl_topk_mean"] = float(np.mean(kl_vals)) if kl_vals else None
                agg_entry["kl_topk_std"] = float(np.std(kl_vals)) if kl_vals else None
                agg_entry["entropy_topk_mean"] = float(np.mean(entropy_vals)) if entropy_vals else None
                agg_entry["entropy_topk_std"] = float(np.std(entropy_vals)) if entropy_vals else None
                agg_entry["entropy_ratio_mean"] = float(np.mean(entropy_ratio_vals)) if entropy_ratio_vals else None
                agg_entry["entropy_ratio_std"] = float(np.std(entropy_ratio_vals)) if entropy_ratio_vals else None

    return results


# "early" ≈ 12.5% depth; see Appendix A.4 and Appendix B.5 for layer selection rationale
def resolve_scramble_layers(n_layers: int, scramble_spec: List[str]) -> List[int]:
    """
    Resolve scramble layer specifications to actual layer indices.
    
    Args:
        n_layers: Total number of transformer blocks in model
        scramble_spec: List of specs like ["early", "mid", "late"] or ["5", "10"]
        
    Returns:
        List of layer indices (in hidden_states indexing, so +1 from block index)
        
    Layer selection (returns hidden_states index, not block index):
        - "early": ~15% depth (layer 12 for 80-block model → hidden_states[12])
        - "mid": ~50% depth
        - "late": ~85% depth
        
    Note: The returned indices are for hidden_states, not model.layers.
    hidden_states[i] for i>=1 comes from block[i-1].
    """
    resolved = []
    
    for spec in scramble_spec:
        if spec == "early":
            # ~15% depth, minimum layer 1 (can't scramble embedding at 0)
            layer = max(1, int(n_layers * 0.15))
            resolved.append(layer)
        elif spec == "mid":
            layer = n_layers // 2
            resolved.append(layer)
        elif spec == "late":
            # ~85% depth
            layer = int(n_layers * 0.85)
            resolved.append(layer)
        elif spec.isdigit():
            layer = int(spec)
            if 1 <= layer <= n_layers:  # Must be >= 1 (can't scramble embedding)
                resolved.append(layer)
            else:
                print(f"[WARN] Layer {layer} out of valid range [1, {n_layers}]")
        else:
            print(f"[WARN] Unknown scramble spec: {spec}")
    
    return sorted(set(resolved))

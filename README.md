# PDSF: Prediction-Anchored Decomposition into Functional Subspaces

Code and prompt sets for:

**"Scale-Invariant Prediction-Proximal Structure in Transformer Residual Streams"**
Nelson Guda, 2026. arXiv: XXXX.XXXXX

## Overview

PDSF decomposes the residual stream of transformer language models into four strictly orthogonal subspaces ordered by proximity to the model's own prediction direction:

- **P (Predictive)** — rank-1 projection onto the unembedding vector for the model's predicted next token.
- **D (Discriminative)** — top-k PCA of the P-residual across prompts. Captures the high-variance content most related to prediction. Typically 4–12 dimensions.
- **S (Situational)** — top-k PCA of the double residual after removing P and D. Captures slower-varying contextual structure. Typically 10–16 dimensions.
- **F (Framework)** — everything that remains. Spans thousands of dimensions (>99% of the residual stream).

The decomposition guarantees `h = h_P + h_D + h_S + h_F` with strict orthogonality and lossless reconstruction at every layer.

The paper uses this coordinate system to measure a set of geometric and causal properties that are consistent across 18 models from 6 architecture families (7B–120B parameters):
prediction-proximal dimensionality does not scale with model width; a monotonic manifold complexity gradient is anchored to prediction proximity; the prediction-distal complement is freely transferable across semantic domains; and the two prediction-proximate subspaces produce dissociable behavioral effects when disrupted.

## Repository Structure

```
pdsf-residual-geometry/
├── pdsf_pipeline.ipynb          # Main experiment notebook
├── pds_geometry.py              # Model loading, hidden state extraction, Part G interventions
├── pds_continuation.py          # PDSF basis computation, D/S scramble, F interventions
├── specA_analysis.py            # Geometry analysis (Parts A–H)
├── pds_prompt_analysis.py       # Token distribution analysis utilities
├── pds_spirality.py             # Spirality measures (spectral concentration, phase linearity)
├── prompts/
│   ├── SpecA_prompts_v4_full_factorial.json   # 224 factual prompts, 14 groups × 16 variants
│   ├── SpecB_experimental_prompts.json        # 80 open-ended continuation prompts
│   └── specB_diverse_prompts.json             # 84 prompts across 8 linguistic regimes
├── LICENSE
├── .gitignore
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU with ≥8 GB VRAM (test mode), ≥16 GB (7B models), ≥80 GB or multi-GPU (70B+ models)
- HuggingFace account with access to gated models (Llama, Gemma)

### Installation

```bash
pip install torch transformers accelerate bitsandbytes tqdm scipy matplotlib huggingface_hub
```

### Test Mode (Verify Installation)

The notebook includes a `TEST_MODE` toggle that runs the full pipeline end-to-end on
`gemma-2-2b-it` (~5 GB) with minimal prompts in approximately 10–15 minutes on a single GPU:

1. Open `pdsf_pipeline.ipynb`
2. In Cell 9, set `TEST_MODE = True`
3. Run all cells

Test mode automatically selects the smallest model, reduces prompt counts to 4 per set,
limits generation to 16 tokens, and disables expensive experiment parts. Use it to verify
that the code runs correctly in your environment before committing to full experimental runs.

### Full Experimental Run

1. Open `pdsf_pipeline.ipynb`
2. Set `TEST_MODE = False`
3. In Cell 9, edit `MODELS_TO_RUN` to select your model(s)
4. In Cell 10, configure which experiment parts to enable
5. In Cell 10 Section 3A, update `BASE_DIR` to your working directory
6. Run all cells

A full run on a single 7B model (all geometry + continuation experiments) takes
approximately 2–4 hours on an A100-80GB.

## Prompt Sets

**SpecA** (224 prompts): 14 groups × 16 variants in a 2⁴ factorial design crossing
paraphrase, constraint emphasis, preamble, and output format. Each group poses a
factual or logical question with a known single-token answer. Used for geometry experiments.

**SpecB** (80 prompts): 10 groups × 8 variants. Open-ended narrative starters that the
model continues for 64 tokens. Used for continuation/intervention experiments.

**Diverse** (84 prompts): 21 groups × 4 variants spanning 8 linguistic regimes — code
completion, East Asian languages, English analytical, English narrative, formal mathematics,
Romance languages, structured instructions, and unusual register.

## Module Guide

### pds_continuation.py
The canonical implementation of the PDSF decomposition. Contains:
- `compute_P_bases_from_predictions()` — P basis from unembedding vectors
- `compute_D_basis_global()` — D basis via PCA of P-residual with adaptive rank
- `compute_S_basis_global()` — S basis via PCA of P+D residual with adaptive rank
- `decompose_hidden_state()` — Full P→D→S→F sequential projection
- `generate_with_D_scramble()`, `generate_with_S_scramble()` — Persistent rotational interventions
- `run_F_mix_experiment()`, `run_F_transplant_experiment()` — F intervention protocols

### pds_geometry.py
Infrastructure layer. Contains:
- `MODEL_REGISTRY` — Maps short names to HuggingFace model IDs
- `load_model_bundle_multigpu()` — Multi-GPU loading with 4-bit quantization
- `extract_hidden_states()` — Residual stream extraction at every layer
- `run_scramble_experiment()` — Part G geometry-level interventions
- `run_invariance_control()` — Null experiment (decompose-recompose identity check)

### specA_analysis.py
Geometry analysis pipeline. Implements Parts A–H of the SpecA experiment:
effective dimensionality (participation ratio), family separation, factorial effects,
trajectory analysis, energy landscape, rotation analysis, interventions, and injectivity.

### pds_spirality.py
Measures helical structure in residual stream trajectories: spectral concentration,
phase linearity (R²), and winding number.

## Supported Models

The 18-model testbed from the paper:

| Model | Key | Parameters | Hidden Dim |
|-------|-----|-----------|------------|
| Llama 8B Instruct | `llama_8b` | 8.0B | 4,096 |
| Llama 70B Instruct | `llama_70b` | 70.6B | 8,192 |
| Gemma 9B IT | `gemma_9b` | 9.2B | 3,584 |
| Gemma 27B IT | `gemma_27b` | 27.2B | 4,608 |
| Mistral 7B Instruct | `mistral_7b` | 7.2B | 4,096 |
| Mixtral 8×7B Instruct | `mixtral_8x7b` | 46.7B | 4,096 |
| Qwen 14B Instruct | `qwen_14b` | 14.2B | 5,120 |
| Qwen 72B Instruct | `qwen_72b` | 72.7B | 8,192 |
| GPT-OSS 20B | `gpt_oss_20b` | 20.0B | 6,144 |
| GPT-OSS 120B | `gpt_oss_120b` | 120B | 13,312 |

Additional models in the registry (Gemma 2B, Llama 1B/3B, Qwen 7B/32B, Phi-3)
can be used for exploration but are not part of the paper's testbed.

## Output Format

All results are saved as JSON files with standardized metadata headers.
File naming convention: `{model_key}-{Experiment}-{PromptSet}-{part}.json`

Example: `llama_8b-Geometry-SpecA-part_g_v11_scramble.json`

## Citation

```bibtex
@article{guda2026scale,
  title={Scale-Invariant Prediction-Proximal Structure in Transformer Residual Streams},
  author={Guda, Nelson},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

Code: MIT License. See [LICENSE](LICENSE).
Prompt sets and data: CC-BY 4.0.

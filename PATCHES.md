# PATCHES — `pdsf_pipeline.ipynb` (GitHub release, pristine)

Patches against the **unmodified** GitHub release notebook
`releases/paper_1-pdsf-residual-geometry-v1.0-2026-03-22/pdsf_pipeline.ipynb`
(V11.4), as needed to run **Mixtral 8×7B Instruct, Diverse prompt set, `part_j`
(resolution geometry)** in **bf16** on a **single H200 (141 GB)**.

All four fixes live in the notebook; `pds_geometry.py`, `pds_continuation.py`,
and `specA_analysis.py` need **no changes** (their tensor→numpy conversions already
force `.float()` before `.numpy()`, so they are bf16-safe).

Because `.ipynb` is JSON, these are given as exact **find → replace** blocks keyed
by cell, not as a unified diff. Cell numbers are 0-indexed code-cell positions in
the pristine notebook.

Context: two of these bugs are only exposed once you (a) run on a card that can
hold the model in bf16 and (b) actually reach the MoE forward pass on a recent
`transformers` build whose Mixtral path uses the bf16-only grouped-GEMM kernel.
On smaller GPUs the auto-4-bit path masked them.

---

## Patch 1 — module-level scipy import (fixes silent "skipped: squareform not defined")

**Why:** the only `pdist`/`squareform` import in the notebook is indented *inside*
one function (cell 23), so it is not visible to the `_subspace_geometry` helper the
resolution-geometry functions rely on. The result is a `NameError` that the
stratified block swallows per-regime (writing `"skipped": true, "reason": "name
'squareform' is not defined"`), while the group and within-regime blocks raise and
write nothing. Failure is **silent** — a partial file is produced, so it looks like
it ran.

**Cell 15** (the main imports cell — the one containing `import numpy as np`).

Locate:
```python
import numpy as np
from tqdm import tqdm
```
Replace with:
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
```

(The stray indented `from scipy.spatial.distance import pdist, squareform` deeper in
cell 23 can be left in place — harmless — or removed.)

---

## Patch 2 — load weights in bf16, not fp16 (fixes `Expected mat_a to be BFloat16 matrix got Half`)

**Why:** the loader's general branch hard-codes `torch.float16` for every
non-Gemma-2, non-pre-quantized model. Recent `transformers` routes Mixtral's MoE
through `torch._grouped_mm`, which is **bf16-only**, so an fp16 load dies in the
first expert matmul. bf16 is also the safer default for the other MoE models in the
registry (Qwen, GPT-OSS) and avoids fp16 activation overflow generally.

**Cell 24**, inside `load_model_for_pipeline`.

Locate:
```python
        elif not pre_quantized:
            load_kwargs["torch_dtype"] = torch.float16
```
Replace with:
```python
        elif not pre_quantized:
            load_kwargs["torch_dtype"] = torch.bfloat16
```

---

## Patch 3 — keep the whole model on one GPU when there's only one (fixes CPU/meta-device offload)

**Why:** `device_map="auto"` decided to offload part of the model to CPU even with
94 GB of weights and 141 GB of VRAM ("Some parameters are on the meta device
because they were offloaded to the cpu"), which is both unnecessary and a source of
fragility. Forcing everything onto GPU 0 when `device_count() == 1` removes the
offload; multi-GPU boxes keep `"auto"` sharding.

**Cell 24**, inside `load_model_for_pipeline`.

Locate:
```python
        load_kwargs = {
            "device_map": "auto", "cache_dir": cache_dir,
            "trust_remote_code": True, "low_cpu_mem_usage": True,
        }
```
Replace with:
```python
        _n_gpus = torch.cuda.device_count()
        load_kwargs = {
            "device_map": ({"": 0} if _n_gpus == 1 else "auto"),
            "cache_dir": cache_dir,
            "trust_remote_code": True, "low_cpu_mem_usage": True,
        }
```

---

## Patch 4 — decide 4-bit by VRAM fit, not by coarse profile bucket (fixes H200 wrongly forced to 4-bit)

**Why:** quantization is chosen by comparing model size to a per-bucket
`quantize_threshold`. The H200 (141 GB) lands in the `total_vram >= 80` →
`single_h100` bucket, whose threshold is 80, so any model larger than 80 GB is
4-bit-quantized even though bf16 fits comfortably in 141 GB. The fix: quantize only
when the bf16 weights genuinely do not fit in detected VRAM (with headroom), and
add an explicit override for edge cases.

### 4a — add a precision override knob

**Cell 12**, in SECTION 1 ("WHAT TO RUN"), near the other top-level toggles. Add:
```python
PRECISION_OVERRIDE = "auto"   # "auto" | "bf16" | "4bit" — "auto" fits bf16 into detected VRAM
```

### 4b — replace the quant decision

**Cell 12**, inside `setup_model_run`.

Locate:
```python
    # Determine if 4-bit quantization is needed
    _sizes = _NOTEBOOK_MODEL_SIZES_GB if '_NOTEBOOK_MODEL_SIZES_GB' in dir() or '_NOTEBOOK_MODEL_SIZES_GB' in globals() else MODEL_SIZES_GB
    USE_4BIT_QUANTIZATION = _sizes.get(model_key, 50) > profile.get("quantize_threshold", 40)
```
Replace with:
```python
    # Determine if 4-bit quantization is needed.
    # Decide by whether bf16 weights fit in detected VRAM (with headroom), not by a
    # coarse per-bucket threshold — the bucket approach wrongly quantizes cards like
    # the H200 (141 GB) that sit in the 80–160 GB band but can hold >80 GB in bf16.
    _sizes = _NOTEBOOK_MODEL_SIZES_GB if '_NOTEBOOK_MODEL_SIZES_GB' in dir() or '_NOTEBOOK_MODEL_SIZES_GB' in globals() else MODEL_SIZES_GB
    _bf16_gb = _sizes.get(model_key, 50)
    _vram_gb = globals().get("total_vram", 0.0)   # set in the GPU-detection cell
    _headroom = 1.15                               # activations / KV cache / fragmentation
    _prec = globals().get("PRECISION_OVERRIDE", "auto")
    if _prec == "bf16":
        USE_4BIT_QUANTIZATION = False
    elif _prec == "4bit":
        USE_4BIT_QUANTIZATION = True
    else:  # "auto"
        USE_4BIT_QUANTIZATION = (_bf16_gb * _headroom) > _vram_gb
```

Correctness across configs: H200 141 GB → `94×1.15 = 108 < 141` → bf16; single
H100 80 GB → `108 > 80` → 4-bit (bf16 genuinely won't fit); 2×H100 160 GB → bf16
with `"auto"` sharding. The `quantize_threshold` field in the profile buckets is no
longer read for this decision; the buckets still drive `batch_size`.

---

## Apply order and re-run

1. Apply Patches 1–4.
2. Re-run **cell 15** (imports), **cell 12** (config + `setup_model_run`), and
   **cell 24** (loader + pipeline) so the new definitions take effect, then run the
   pipeline cell.
3. Expected: Phase 1 prints `4-bit quantization: NO`; no meta-device/offload
   warning; extraction runs 84 forward passes; the Diverse **group** block prints a
   real `P-PCA: angle=…° var=… rank=…` line; three files are written —
   `-Diverse-group-resolution_geometry_v11.json`, `…_stratified.json`,
   `…_within_regime.json`.

## Audit — paths checked and confirmed safe (no patch needed)

- `extract_hidden_states` (pds_geometry.py): `h_pos.float().cpu()` and
  `logits.float().cpu()` with `move_to_cpu=True` default → bf16→float32 before any
  `.numpy()`. Safe.
- `_safe_unembed_to_numpy` (cell 24) and `get_unembedding_matrix` (pds_geometry.py):
  `.detach().cpu().float().numpy()` on every return path, incl. meta-tensor
  fallback. Safe.
- `_to_np` (specA_analysis.py): `.detach().cpu().float().numpy()`. Safe.
- No stray `.half()` / `float16` casts in the geometry compute path; the only
  `float16` references are in the unused 4-bit branch.
- `compute_D_basis_global` / `compute_S_basis_global` exist in pds_continuation.py
  with matching signatures and the expected adaptive-rank clamps (D `[2,12]`,
  S `[4,16]`).
- `RESOLUTION_GEOMETRY_LAYERS = "all"` → full-depth (33-layer) trajectory, as
  required.

## Caveat worth a methods footnote

The pristine loader default was fp16, so the nine peer Diverse files were most
likely extracted in fp16 (not recorded in the JSON — inferred from this same
default). Mixtral is now forced to bf16 (fp16 can't run this transformers' MoE
kernel). For the reported quantities (principal angles, participation ratios,
curvature ratios) fp16↔bf16 differs only in ~the 3rd significant figure, well below
the effect sizes — comparability holds — but it is a real precision inconsistency
worth one line in the methods and a `verification_incidents.md` note.

---

## Patch 5 — MISSING TESTS: the release does not produce four Paper-1 results

A completeness audit (2026-07-02, `revision_documents/github_release_paper1_completeness_audit.md`) found the release lacks four test families whose results appear in Paper 1:

1. **part_n scaling rank** (six rank estimators) — §4.1, Table 1, Figure 2C. Port `compute_part_N_scaling_rank_suite` + helpers (`stable_rank`, `spectral_entropy_rank`, `explained_variance_rank`) from `scripts/V11.6 (post-persistent-hook)/specA_analysis.py` (lines 2622–2757, 2541–2598); add a `part_n` toggle to `RUN_PARTS` and a call in the geometry phase.
2. **F-topK rotation** (persistent behavioral) — §6.4, Figure 8. Port from `scripts/V11.8 (Paper 2)/pds_continuation.py` (+ `compute_layerwise_F_topK.py`).
3. **F-dynamic rotation** — §7.3, Figure 9. Port from `scripts/V11.8 (Paper 2)/pds_continuation.py`.
4. **step-offset F injection** — §7.2, Table 5. Port `run_step_offset_F_injection` from `scripts/V11.8 (Paper 2)/pds_continuation.py`.

Also confirm the Part G phase invokes `run_scramble_experiment` (produces `part_g_v11_scramble`, source for §5.1) — the function is present in `pds_geometry.py` but the July run only wrote `regime_transplant`.

These are the substance of the forthcoming GitHub-update document; Patches 1–4 above are the environment/loader fixes already applied.

---

## Patch 5 — Part J metadata references undefined `spectrum_layers` when Part I is off

**Why:** the Part J (Resolution Geometry) `_metadata` block in Cell 25
(`run_consolidated_specA_analysis`) was copied from the Part I (Energy Spectrum)
block and left referencing `spectrum_layers`. That variable is only bound inside the
Part I block, which is gated by `run_parts["run_energy_spectrum"]`. Running with
Part I disabled (`run_energy_spectrum: False`) raises
`UnboundLocalError: cannot access local variable 'spectrum_layers'` — and crucially it
raises at the metadata step **after** the (multi-minute) resolution-geometry compute
but **before** `save_json`, so the whole Part J result is computed and then discarded.
Symptom in the log: `Resolution geometry: FAILED - ... 'spectrum_layers' ...` followed
by `✓ Analysis: 0 parts in 0.0s`. First hit on the `gemma_27b_base` large-base rerun,
2026-07-04. The same consolidated function serves SpecA / SpecB / Diverse, so this one
block was the only site.

**Cell 25** (`run_consolidated_specA_analysis`), the Part J `_metadata` dict.

Locate:
```python
                "k_neighbors": 8,
                "n_layers_sampled": len(spectrum_layers),
                "layers_sampled": spectrum_layers,
```
Replace with:
```python
                "k_neighbors": 8,
                "n_layers_sampled": (len(rg_layers) if rg_layers is not None else len(layers_sorted)),
                "layers_sampled": (rg_layers if rg_layers is not None else layers_sorted),
```

`rg_layers` (the Part J layer set; `None` in full-depth mode) and `layers_sorted`
(all layers) are both in scope at that point. The fix binds correctly regardless of
whether Part I ran, and reports Part J's true layer coverage (all layers in
full-depth mode) rather than Part I's 5 sampled layers. Leave the identical-looking
Part I metadata lines (which legitimately use `spectrum_layers`) untouched.

Pre-fix snapshot: `versions/v2_partJ_spectrum_layers_bug_2026-07-04/` (see its `NOTES.md`).

---

## Patch 6 — Part L (and Part M Phase C) gated behind Part G; skipped when interventions are off

**Why:** in `run_full_pipeline` (Cell 25), Part L (Complexity Depth Profiles) and Part M
Phase C (spirality↔rotation correlation) were nested *inside* the Phase 5A block
`if run_part_g and layer_to_H is not None:`. Both are baseline, cache-based measures that
use the already-extracted `layer_to_H` + `P_bases` and need no forward passes (the code's
own comment says so), yet with `part_g=False` (geometry-only runs) the entire Part G block
is skipped and Part L never runs. First observed on the `gemma_27b_base` geometry run
(2026-07-05): `resolution_geometry_v11` (Part J) and `part_k` landed, but
`part_l_complexity_depth_profile`, `part_m_correlation`, and `part_m_spirality` were absent.
This also explains the earlier "only one result" symptom of the first `part_g=False` run.

**Fix:** moved the Part L block and the Part M Phase C block out of the Part G
`if/try`, to run immediately after the Phase 5A `if/else`, each gated by its own toggle.
Part M Phase B (`extract_part_m_from_part_g`) legitimately needs `part_g_results` and stays
inside the Part G block. The moved Part L guard gains `and layer_to_H is not None` (it was
previously implied by the enclosing Part G condition). Applied programmatically with a
compile-gate and count-preservation checks (`run_part_L_for_pipeline`,
`run_part_M_correlation_for_pipeline`, `extract_part_m_from_part_g`, and the Part G call
each retain their original occurrence counts across the SpecA/SpecB/Diverse phases; only
the SpecA instances were relocated).

Net effect: with `part_g=False`, Part L and Part M Phase C now run from the cached
hidden states. The SpecB/Diverse Part L calls (in their own phase drivers) were not
affected. No pre-fix snapshot was needed beyond `versions/v2_...` since Patch 5 and this
patch are documented here; snapshot the fully-patched notebook to `versions/` once a
`part_g=False` run confirms Part L saves.

---

## Patch 7 — TEST_MODE rebuilt into a real smoke test

**Why:** the shipped `TEST_MODE` was misconfigured for its stated purpose ("verify the
pipeline runs end-to-end"): (1) it selected `mixtral_8x7b` (94 GB) as the "small model"
despite the on-screen message saying `gemma_2b`; (2) its `RUN_PARTS` ran only A/E/G/K —
**skipping Part J and Part L**, the exact parts that carried the Patch 5 / Patch 6 bugs, so
a test run could not catch them; (3) it updated `RUN_PARTS` *after* `PIPELINE_OPTIONS` was
derived but only rebuilt some option keys, so enabling J/L in test mode wouldn't have
propagated to the option-gated code paths; and (4) it did not reduce the SpecA **geometry**
prompt count (only intervention counts), so extraction still ran all 224 prompts.

**Fix (three cells):**
- **Cell 12** — `TEST_MODE` model `mixtral_8x7b` → `gemma_2b` (smallest, ~5 GB).
- **Cell 13** `if TEST_MODE:` — `RUN_PARTS` now mirrors a real geometry run: all parts True
  except `part_g` and `part_m_intervention` (so it exercises A–F, H, I, **J**, **K**, **L**,
  M, N). Added re-derivation of the geometry option keys from the overridden `RUN_PARTS`
  (`run_geometry_part_g`, `run_geometry_part_l`, `run_resolution_geometry`,
  `run_energy_spectrum`) and a `TEST_N_GROUPS = 3` knob.
- **Cell 18** — after SpecA parsing, `TEST_MODE` slices to the first `TEST_N_GROUPS` **whole
  groups** (group-aware, order-preserving), keeping group structure intact for parts B/C/K
  while cutting ~224 prompts to ~48.

Net: `TEST_MODE = True` runs gemma_2b on ~48 prompts through the full geometry battery in a
couple of minutes, exercising the previously-broken Part J/L paths and surfacing the still-open
Part H exception. Pair with a post-run manifest check (enabled parts vs produced files) for a
complete "does the script work" verification.

---

## Patch 8 — rank-matched PCA control for Part J (2026-08-03)

**Why:** the Part J PCA control is described as partitioning components "into bins matching the
PDSF subspace sizes", but it allocated a **fixed budget of 50** components in the order
P -> D -> S -> F, giving F whatever was left. PDSF's own F is separately truncated to 50
(`res_rank`, Step 5), so the control's F bin was `50 - (P+D+S)` and was **never** rank-matched
on any prompt set: 13-19 on Diverse, 17-20 on SpecA, 6-19 on SpecB. It degrades to nothing
precisely where the PR-adaptive P rank is large - GPT-OSS 120B's Diverse `bin_F` fell to rank 2
(PCA/PDSF = -0.752) and GPT-OSS 20B's was **absent** (P 27 + D 12 + S 13 = 52 > 50).

Two literals were involved, and they are easy to confuse:
- **line ~324** `res_rank = min(50, ...)` truncates **PDSF F**.
- **line ~405** `pca_basis(H, k=min(50, n_prompts - 1))` was the entire **control budget**.

**Constraint.** Sizing the control's F bin to PDSF F's rank of 50 needs `P + D + S + 50`
components, but only `n_prompts - 1` exist. On Diverse (84 prompts) that is 83, and seven of
ten models need 84-102. Exact matching at 50 is unreachable there.

**Fix (match downward, both sides).** Per model and layer,
`r_F = max(0, min(50, (n_prompts - 1) - (p_rank + k_D + k_S)))`, with **both** the PCA `bin_F`
and a matched PDSF F computed at exactly `r_F`. The headline PDSF F stays at rank 50, so
Section 4.3's primary numbers, Figure 3's bands and the D < S < F ordering statistics are
untouched; only the control ratio is redefined. A secondary common-rank pass at `r_F = 31`
(the Diverse floor) is emitted alongside, so the cross-model mean can be checked for
`r_F`-sensitivity.

**Six edits, all inside `compute_resolution_geometry` in the analysis cell** (scoped strictly to
that function - `compute_resolution_geometry_within_regime` and
`compute_resolution_geometry_stratified` were not touched; the latter inherits the fix by
delegation):

1. **Step 4b (new, before Step 5)** - compute `F_TRUNC_RANK`, `n_pca_max`, `r_F`, `r_F_common`.
2. **Step 5** - add `_F_at(rank)` helper; emit `Res_coords_matched` / `Res_coords_common` and
   their realised ranks alongside the unchanged rank-50 `Res_coords`.
3. **Step 6** - extend the subspace loop with `residual_rank_matched` and
   `residual_rank_common31`.
4. **Step 8** - budget becomes `min(n_pca_max, p_rank + k_d + Bs.shape[1] + r_F)`. PCA bases are
   nested, so the top `p_rank` columns are unchanged and the P-vs-PCA principal angles are
   bit-identical to the pre-patch run (verified).
5. **Step 9** - add the `bin_F_common31` contiguous bin at the same offset.
6. **`basis_ranks`** - record `residual_rank_matched`, `residual_rank_common31`,
   `pca_control_budget` and a `pca_control_rule` string, so this never has to be traced again.

`Step 10` (the 20-shuffle control) needed no edit; it reads `total_pca_rank` from
`H_pca_basis.shape[1]` and follows the new budget automatically. Its values do change - expected,
and more correct.

**Deliberately NOT changed:** `compute_part_K_cosine_discrimination` has its own `bin_sizes` with
an **uncapped** `bin_F = total_pca_rank - p - k_D - k_S`. It has no 50 and no starvation, shares
no code with Part J, and v62 quotes no PCA control in Section 4.4 / B.4. Harmonising it would
move Figure 4. `compute_part_L_complexity_depth_profile` has its own `res_rank = min(50, ...)`
but no PCA control at all.

**Three configuration changes** (also applied, all reversible):

- **Model-selection cell** - `MODELS_TO_RUN` set to the 10 instruct models, smallest to largest,
  so the two GPT-OSS sanity-gate models land at positions 5 and 7.
- **Config cell (the paper-aligned one - the later of the two, and the one that takes effect)** -
  a clearly delimited `JOB 1 OVERRIDE` block inserted immediately before the `Apply RUN_PRESET`
  section: `RUN_PRESET = "custom"`, `RUN_CONTINUATION = False`, all three geometry prompt sets on
  except `Diverse_regime`, and `RUN_PARTS` all False except `part_j`. `OUTPUT_DIR` redirected to
  `results_rankmatched/` so a run cannot overwrite prior results. Delete the block to restore
  defaults.
- **`PRECISION_OVERRIDE` set to `"bf16"` at its Section-2 assignment, not in the override block.**
  This matters: the Section-2 assignment runs *after* the override block and silently clobbers
  anything set earlier - the first attempt at this patch put it in the block and it read back as
  `"auto"`. Under `"auto"`, `setup_model_run` computes
  `USE_4BIT_QUANTIZATION = (size_gb * 1.15) > vram_gb`, so any model that does not fit is loaded
  in nf4/fp16 with only a one-line notice. Different weights mean different hidden states and a
  run comparable neither to canonical nor across its own models. Verified: on a single 80 GB card
  under `"auto"`, Mixtral 8x7B, LLaMA 70B and Qwen 72B all silently 4-bit; under `"bf16"` none do,
  and an undersized card fails loudly on OOM instead.

**Verification performed before shipping:**

- Both edited cells compile; the patched function compiles in isolation.
- The three config cells were **executed** with a stubbed `torch`; the resulting
  `PIPELINE_OPTIONS` has `run_resolution_geometry = True` with every Part G / L / M and
  continuation key False, and `USE_4BIT_QUANTIZATION` is False for all ten models on 2xH200.
- The pre-patch and post-patch functions were run side by side on synthetic data across five
  rank profiles. In every case `bin_F` rank now equals `residual_rank_matched`, `basis_ranks.residual`
  is still 50, and `P_vs_PCA_mean_angle_deg` is **bit-identical** between the two versions.
- The `r_F` formula was evaluated against the ten real final-layer `(n, P, D, S)` profiles:
  Diverse 31-50 (GPT-OSS 20B 31, previously absent; GPT-OSS 120B 35, previously rank 2),
  SpecA 50 on all ten, SpecB 41-50.

**Snapshot:** pre-patch notebook preserved at
`versions/v3_pre_patch8_rankmatched_pca_2026-08-03/`.

**Note for the release build.** The two configuration cells still both exist, and the later one
is not the value-identical duplicate its own comment claims (see README, "Which notebook is
authoritative"). Before the GitHub release, delete the earlier cell or reconcile the two -
a reproducer who edits the earlier one will see no effect.

---

## Patch 9 — release preparation (2026-08-04)

Applied after the Patch 8 rerun completed, to put the notebook in a state a reviewer can read
and run. Snapshot of the as-run state: `versions/v4_post_rankmatched_run_2026-08-04/`.

**1. Removed the duplicate configuration cell.** The notebook shipped with two configuration
cells; the second silently overrode the first (see README, "Which notebook is authoritative"),
so a reviewer editing the first would see no effect. Verified safe before deleting: the second
cell is a strict superset — same `setup_model_run` definition, same imports, no assignment
present only in the first. Confirmed by executing cells `[model-selection, first, second]` and
`[model-selection, second]` in a stubbed environment and diffing the resulting globals (no
differences) and `PIPELINE_OPTIONS` (identical, 72 keys). **There is now one configuration cell.**

**2. Restored initial run settings.** The Job 1 override block was deleted, `OUTPUT_DIR` returned
from `results_rankmatched/` to `results/`, and `MODELS_TO_RUN` returned to the single-model
default `["llama_8b"]`. `RUN_PRESET` is back at `"all"`, `RUN_PARTS` all True, `TEST_MODE` False.

**3. Cleared all stored output and execution counts.** Three cells carried output from the
environment checks.

**4. `PRECISION_OVERRIDE` left at `"bf16"` (changed from the original `"auto"` in Patch 8).**
This is deliberate and is documented in the header cell. Under `"auto"`, `setup_model_run`
computes `USE_4BIT_QUANTIZATION = (size_gb * 1.15) > vram_gb`, silently loading any oversized
model in nf4/fp16 with a single line of notice — different hidden states, results comparable
neither to the published run nor across models. `"bf16"` forces the flag False so an undersized
card fails loudly instead. Reviewers who want the old behaviour can set it back.

**5. Filtered `snapshot_download` (`ignore_patterns`).** The unfiltered call pulled every weight
format in each repo: Mixtral 8×7B ships consolidated `.pt` (96 GB) beside safetensors (95 GB),
and gpt-oss-120b ships three complete copies (`original/` 72 GB, `metal/` 65 GB, safetensors
58 GB). Across the ten-model testbed that is ~695 GB rather than ~400 GB, and it exhausted the
volume during the 2026-08-04 run. Every model in `MODEL_REGISTRY` loads from the standard
sharded safetensors, so the filter skips nothing that is used.

**6. Reviewer-facing notes added to the header cell** — disk requirements, the precision default
and its trade-off, the Part G condition scope, the Part J control-bin rule, how `RUN_PRESET`
relates to the individual switches, and the fact that only SpecA is enabled by default.

**7. Part G condition scope commented in place.** The battery computes 14 conditions and the
paper reports 8. The six transplant conditions were **kept**, not removed: the published Part G
artefact (`part_g_v11_regime_transplant.json`) contains them and the paper's KL table derives
from that file, so removing them would break reproduction of a published result. The transplant
machinery also lives in `pds_geometry.py` (78 references) and `pds_continuation.py` (26), not in
the notebook, so removing it from the notebook alone would be cosmetic. A comment at the
condition list states the scope instead.

**Verification:** every code cell compiles (notebook magics excluded); the configuration chain
executes to the defaults listed above; no cell carries stored output or an execution count.

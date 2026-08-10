# MANIFEST — paper_1-pdsf-residual-geometry v1.0 (first public release) · **STAGING, NOT FROZEN**

**Status:** release staging, prepared 2026-08-06; **code finalized 2026-08-06** after seven GPU verification runs. **Do not cite this path as a release** — it is mutable and the upload has not happened yet. The remaining gates are the upload itself and the supplementary/data tiers, not the code.
**Version numbering reset.** This tree was previously staged as "v1.1" against a "v1.0" in `releases/paper_1-pdsf-residual-geometry-v1.0-2026-03-22/`. That earlier tree was **never made public**, so it is not a release in any sense a reader can act on. This tree is therefore **v1.0, the first public version**, and `CHANGELOG.md` starts here. The 2026-03-22 directory stays in `releases/` as vault provenance only.
**Renamed** from `github_patched/` on 2026-08-04; all vault citations were retargeted the same day.

## What ships, and what does not

`.gitignore` is the authoritative answer, and it is written so that `git add .` from this directory produces exactly the intended repository.

**Ships:**

| File | Note |
|---|---|
| `pdsf_pipeline_github.ipynb` | the only notebook in the release |
| `pds_geometry.py`, `pds_continuation.py`, `specA_analysis.py`, `pds_prompt_analysis.py`, `pds_spirality.py` | |
| `prompts/SpecA_prompts_v4_full_factorial.json` | 224 prompts, 14 groups × 16 |
| `prompts/SpecB_experimental_prompts.json` | 96 prompts, 12 groups × 8; the paper's 80 is the `"standard"` tier |
| `prompts/Diverse_prompts.json` | 84 prompts, 8 regimes. **Renamed 2026-08-06** from `specB_diverse_prompts.json`, which read as a SpecB file |
| `README.md`, `CHANGELOG.md`, `LICENSE`, `.gitignore` | |
| `data/` — 132 files, 14.5 MB | **added 2026-08-07**; see the section below |

**Does not ship (gitignored, vault-only):** `MANIFEST.md` (this file), `PATCHES.md`, `versions/` (which now also holds the pre-Patch-8 `pdsf_pipeline.ipynb`, moved there 2026-08-06), `.DS_Store`, `_to_delete_session_artifacts/`. The `.gitignore` still carries a bare `pdsf_pipeline.ipynb` line, which matches at any depth, so the exclusion holds wherever that file sits.

`PATCHES.md` is retained in the vault but is not public: every patch in it is written against the 2026-03-22 tree, which no reader ever saw, so publicly it documents a diff between two states that were never both visible. Post-release changes go in `CHANGELOG.md` instead.

## Release-preparation pass, 2026-08-06

Applied after the audit in `../revision_documents/github_release_code_readiness_audit_2026-08-06.md`. Two behavioral fixes, the rest documentation and hygiene.

**Behavioral:**

1. **The Diverse continuation block now receives the F-dynamics switches.** `diverse_options` passed none of `run_step_offset_injection` / `run_passive_F_tracking` / `run_F_topK_rotation` / `run_F_dynamic_rotation`, and `pds_continuation.py` reads them with a `False` default — so the Diverse path could never run them. All 12 canonical `step_offset_F_injection` files are Diverse, which made **Table 5 / §7.2 unreproducible from the release**. Step-offset and passive tracking are now wired on Diverse; F-topK and F-dynamic are pinned `False` there with a comment, because the paper runs both on SpecB only.
2. **The `F_DYNAMICS_*` knobs are now live.** The config cell defined four of them; the module reads lowercase `f_dynamics_*` keys that appeared nowhere in the notebook, so all four were dead and passive tracking silently ran at n_generate=32 rather than the advertised 64. Both `specb2_config` and `diverse_config` now carry the keys.

**Configuration:**

3. `TEST_MODE` now ships **`True`**. A reader who runs all cells as distributed gets the ~10–15 minute `gemma-2-2b-it` smoke test rather than an accidental multi-hour run.
4. The four `RUN_F_*` switches moved from an orphaned block at the bottom of the config cell into Section 1B, where `RUN_PRESET` and `TEST_MODE` can act on them. `"all"` and `"behavioral"` now enable them; `"geometry"` disables them; `TEST_MODE` forces them off. Previously `RUN_PRESET = "all"` left all four off, so the documented "everything" preset produced no Figure 8, Figure 9 or Table 5 data.
5. `BASE_DIR` defaults to `Path(".")` rather than `Path("/workspace")`, so a fresh clone finds `prompts/`. The module-import cell no longer hard-codes `/workspace` on `sys.path`.
6. **Scratch root is no longer hard-coded.** The first cell uses `/workspace` when writable, falls back to `./.pdsf_scratch`, and honours `PDSF_SCRATCH`. Previously a non-pod environment failed in the first cell.
7. **`CLEAR_ALL` in the cache-clearing cell now defaults `False`.** It shipped `True`, and because `clear_all_hf_cache` is not yet imported at that point the standalone branch runs and calls `input()` — a "Run All" would block on a hidden prompt, or delete the model cache. The blocking `input()` is gone as well, replaced by a second `CONFIRM_DELETE` switch, so nothing in the notebook can stall a Run All.

**Cell removal pass, same day.** Four cells removed, 31 → 27, after checking every name each one bound:

- **CUDA availability check** (a `!python -c` shell-out) — bound nothing, and duplicated both the `CUDA available` line the version-check cell already prints and the hard `raise` in GPU detection.
- **"Manual prompt analysis (Optional)"** and **"Manual SpecA-style on SpecB (Optional)"** — no-ops. Each set one flag used nowhere else, and each body was a comment saying the real code "would go here if needed" plus an else-branch printing "disabled". Neither branch did anything.
- A trailing empty code cell.

**Kept deliberately, because they are load-bearing despite reading as diagnostics:**

- **GPU environment detection** sets `total_vram`, which `setup_model_run` reads at call time via `globals().get("total_vram", 0.0)`. Under `PRECISION_OVERRIDE = "auto"` that value is what decides 4-bit quantization — remove the cell and every model silently quantizes because VRAM reads as 0. It also carries the CUDA gate. Its `profile` dict *was* dead (Patch 4 replaced bucket-threshold quantization with VRAM-fit logic, and nothing has read `profile` since), so that block was trimmed and the rest kept.
- **Prompt-analysis utilities** sets `PROMPT_ANALYSIS_AVAILABLE`, read inside `run_full_pipeline`. It short-circuits today only because `run_prompt_analysis` defaults `False`; enabling it with that cell gone would be a `NameError`.
- **Cache status** (read-only disk report) and **cache clearing** (now double-gated) both stay: on a ~400 GB run they are the tools a reader needs, and the hazard in the second one was the `input()`, not the cell.
- **The MXFP4 `kernels`/`triton` check** stays, because GPT-OSS needs it. It was a `!python -c` shell-out that dumped an `ImportError` traceback on any machine without `kernels` — alarming during a smoke test. Rewritten as a `try/except` that prints the install line and says plainly that every non-GPT-OSS model is fine without it.

**Documentation:**

8. Paper title corrected to **"Geometric and Behavioral Stratification in Transformer Residual Streams"** in the README (twice, including the BibTeX) and in four module headers, all of which carried the pre-v30 title.
9. Section cross-references in the modules rewritten against v63 numbering. They used an obsolete scheme throughout (decomposition as §7.1, Part G as §7.3, Part A as §3.1) while the notebook's own part table already used the current one.
10. `arXiv: XXXX.XXXXX` removed from six places; the BibTeX entry is a `@misc` pointing at the repository rather than a placeholder arXiv ID.
11. README rewritten against the authoritative notebook: correct filename throughout, no stale cell numbers, the `RUN_PRESET` × F-dynamics matrix, the 96-vs-80 SpecB tier distinction, the CUDA requirement stated plainly, the disk section, and the precision table with `"bf16"` marked as the actual default.
12. `pds_continuation.py` gained the standard header block the other four modules carry. In all five, `from __future__ import annotations` moved below the docstring — above it, the string is an expression statement and `__doc__` is `None`.
13. Two references to an internal vault document were stripped from `pds_continuation.py`.
14. `.gitignore` added (listed in the README since v1.0 but never present in this tree); `CHANGELOG.md` added.

**Verification.** 27 cells; every code cell parses; the notebook carries no stored output and no execution count on any cell; all five modules compile; the configuration chain executes to the documented defaults (`TEST_MODE=True`, `gemma_2b`, `PRECISION_OVERRIDE="bf16"`, `RUN_PRESET="all"`, F-dynamics off under TEST_MODE, 72 `PIPELINE_OPTIONS` keys); all three prompt paths resolve from `BASE_DIR="."`; and after the cell removals, no surviving cell references a name that only a removed cell bound. **Not yet verified: an actual GPU run.** That is the outstanding gate.

## Post-test-run fixes, 2026-08-06

A TEST_MODE run on gemma-2-2b-it (1×H100 80GB, 14.9 min, no exceptions) is recorded in `test_run/`. Assessment: `../revision_documents/github_release_code_readiness_audit_2026-08-06.md` §7. It confirmed Patch 8 in output (`residual_rank_matched: 18` from n=48, P+D+S=29) and reproduced four of the paper's core results on a model outside the testbed. It also exposed four defects, all now fixed:

1. **TEST_MODE left eight backward-compatibility aliases stale.** The `_alias_map` fan-out ran once, before the `if TEST_MODE:` block mutated twelve `run_geometry_*` keys. Several phase drivers read the aliases, so Part G ran despite TEST_MODE turning it off and the printed option block contradicted itself. The fan-out is now a function, `_apply_option_aliases()`, called both at its original position and again at the end of the TEST_MODE block. TEST_MODE additionally now sets `part_g: True` deliberately — on 4 prompts it costs ~2 s and is the only coverage of the intervention path, and the TEST_MODE banner already claimed it ran.
2. **`save_json` emitted bare `NaN`.** Three of 23 output files were not valid JSON (RFC 8259 forbids `NaN`): Python read them back silently, `JSON.parse` / Go / `serde_json` / `jsonlite` reject the whole file. A `_json_sanitize` pass now maps non-finite floats to `null` and `json.dump` runs with `allow_nan=False`. Verified by rewriting all 23 test-run files through the new writer: 3 fixed, 23/23 now parse under Node, every finite value byte-identical. **The same defect is present throughout `data/canonical/`** in the `part_l`, `resolution_geometry_v11` and `per_prompt_trajectories` families — both of the first two are tier B in the archive plan, so the derived-data deposit needs the same treatment.
3. **The NaN source was catastrophic cancellation.** `softmax_gradient` computes √(E[g²] − E[g]²); where the softmax is near-uniform or near-deterministic the two terms cancel and the argument goes slightly negative (reproduced: −7.1e−15). Both call sites now clamp with `max(0.0, …)`, which yields 0.0 — the correct standard deviation of a numerically constant quantity. `softmax_gradient` is a P-only diagnostic and `curvature_ratio` was finite throughout, so no reported result was affected.
4. **Part G's filename disagreed with its contents.** A reduced run wrote `…-part_g_v11_regime_transplant.json` containing only the eight non-transplant conditions, under a metadata flag asserting otherwise. `_part_g_output_key()` now selects the output name from `GEOMETRY_INTERVENTION_TYPES` — a full run still writes the canonical `…_regime_transplant.json` that Table A.4-1 derives from, a reduced run writes `…_part_g_v11_scramble.json`. `has_regime_transplant` in `pds_geometry.py` now reports whether a transplant condition actually ran, rather than whether regime ids were supplied.

Also: the two Part F `Mean of empty slice` warnings and the `np.corrcoef` zero-variance warning are suppressed at the three sites where the NaN is expected by construction, and undefined correlations are now dropped rather than averaged in; the reference environment is recorded in `README.md` and `CHANGELOG.md`; and `TEST_MODE_INCLUDE_FDYNAMICS` was added beside `TEST_MODE` so the four F-dynamics conditions and the Diverse continuation set can be smoke-tested on the small model (k-values reduced to `[0, 1]`, generations to 16 tokens).

**Verified across three configurations** (TEST_MODE default / TEST_MODE + F-dynamics / full run with `RUN_PRESET="all"`): no stale aliases in any of them, 72 `PIPELINE_OPTIONS` keys throughout, the `RUN_F_*` switches follow the intended pattern, and the Part G output key is `part_g_regime_transplant` on a full run and `part_g_v11` on a reduced one.

**Still unexercised on a GPU:** the F-dynamics conditions themselves — including the Diverse `run_step_offset_injection` wiring that makes Table 5 reproducible — plus Diverse continuation, SpecB/Diverse geometry, Part M, the MoE/bf16 path, native MXFP4 and multi-GPU sharding. A run with `TEST_MODE_INCLUDE_FDYNAMICS = True` closes the first two.

## Second test run, 2026-08-06 (`test_run_2/`)

gemma-2-2b-it, 1×H100 80GB, **13.5 min, no exceptions**, on the 27-cell post-cleanup notebook. All four post-run fixes verified live:

- **Aliases:** zero mismatched pairs in the printed option block (run 1 had five); `run_geometry_part_g` and `run_specA_part_g` now agree.
- **JSON:** 23/23 files pass `sanitize_json_nonfinite.py --check`. Run 1 had three failures.
- **sqrt clamp:** no non-finite value anywhere. It did more than replace NaN with 0.0 — at layers 19 and 22–25 the layer mean was NaN in run 1 only because a single prompt underflowed, and the clamp **recovered real values** there (0.0068, 0.092, 0.0013, 0.0893, 0.092). Only layer 0 is genuinely 0.0. Every other layer is bit-identical to run 1.
- **Part G naming:** writes `part_g_v11_scramble.json`, and `has_regime_transplant` is now `False`, matching the eight conditions actually run.

Also confirmed: zero `RuntimeWarning`, zero "Mean of empty slice", zero "invalid value encountered" (run 1 had all three); the kernels check prints a clean version line; the misleading GPU memory-profile block is gone.

**Determinism, and the one place it failed.** Comparing every shared output between the two runs: bases numerically identical (P/D/S bases, `pred_token_ids`, all zero delta), baselines byte-identical, Part G identical, and D-scramble / S-scramble / random-control / F-attenuate / F-transplant all identical. **F-mix was the sole exception**, and the cause is a defect, not noise:

```python
rng = np.random.RandomState(seed + (hash(key) % 10_000_000))   # pds_continuation.py, run_F_mix_experiment
```

Python randomizes `str` hashing per process (PEP 456), so this seed changed on every kernel restart. F-mix outputs were therefore irreproducible — across our two runs the identical-output rate moved 2.5% → 7.5% and the mean divergence token 0.090 → 0.324. It was the only `hash()` call in the release code. Now seeded from `zlib.crc32`, which is stable across processes, versions and platforms.

**What this means for the published numbers:** the F-mix figures in §5.2 / Figure 7 were produced under the randomized seed, so their exact per-prompt outputs cannot be regenerated by anyone at any precision — not a precision-sensitivity issue, an irreproducible-seed one. The finding is unaffected: F-mix ≫ F-attenuate holds in both runs (2.5% and 7.5% identical against 56%), and the paper's across-model spread on this statistic is wider than the run-to-run wobble. Disclosed in `README.md` under precision sensitivity and in `CHANGELOG.md`.

**Two cosmetic fixes from the same pass:** the header cell rendered part of a line in strikethrough — `~10–15 minute … (~5 GB)` put two tildes on one line and Jupyter's GFM renderer read them as `~…~`. Reworded to "roughly" / "about", and the same pattern fixed in `README.md`. A scan of every rendered markdown line in the notebook and the three shipped docs now returns no further tilde pairs.

**Still unexercised:** the four F-dynamics conditions — run 2 used `TEST_MODE_INCLUDE_FDYNAMICS = False`, so the Diverse step-offset wiring behind Table 5 remains verified by inspection only.

## Third test run, 2026-08-06 (`test_run_3/`) — the F-dynamics path, and the defect it exposed

First run with `TEST_MODE_INCLUDE_FDYNAMICS = True`. **The wiring worked and the code did not.** The log confirms all four conditions were correctly enabled — `Enabled experiments: F_topK_rotation, F_dynamic_rotation, step_offset_injection, passive_F_tracking, …` — and Phase 8 (Diverse continuation) opened for the first time, computed bases and baselines on 84 prompts. Then both continuation phases died at the same point.

### The defect: `FTrackingTelemetry` lost its `@dataclass` decorator in the port

```
⚠️ SpecB continuation failed: 'Field' object has no attribute 'append'
  pds_continuation.py:2458 run_passive_F_tracking -> generate_with_every_token_F_hook
```

`FTrackingTelemetry` declares `pre_hook_F_norms: List[float] = field(default_factory=list)`. Without `@dataclass`, `field()` is never processed and the attribute holds the `dataclasses.Field` sentinel itself, so the hook's `.append` fails on the first generated token. Confirmed three ways: the class at line 2149 had no decorator; the source it was ported from (`scripts/V11.8 (Paper 2)/pds_continuation.py`, line 150) **does** have `@dataclass` directly above the same class; and the error reproduces exactly in five lines of isolated Python.

**Consequence: none of the four F-dynamics conditions has ever been able to run in this release.** Passive tracking is the prerequisite for F-dynamic and runs first, so it took F-topK, F-dynamic and step-offset down with it before any of them was reached. The port was present but dead from 2026-07-02 until now. This is precisely the gap `github_release_paper1_completeness_audit.md` flagged in its own caveats — it checked "presence of the producing function", not that the function runs — and it is the reason this run was worth doing. Fixed by restoring the decorator; an AST sweep of all five modules found no other class using `field()` without `@dataclass`.

### Two further defects the run exposed

**Stale outputs masquerade as fresh ones.** `results/` persists between runs on the same pod, and a crashed phase simply leaves the previous run's files in place. The archived `results.zip` is a mix: `D_scramble_d70`, `S_scramble_d70`, `random_control_d70` and `SpecB-bases` carry 19:42–19:51 timestamps (run 2), while `part_g_v11_scramble`, `F_attenuate`, `F_mix`, `F_transplant` and the Diverse files carry 20:18–20:22 (run 3). The directory looked complete; it was not.

**The pipeline printed a success banner over two dead phases.** `run_full_pipeline` catches phase exceptions into `result["errors"]` and still returns `status: "completed"`, and the summary only tested that status — so the run ended with `PIPELINE COMPLETE` and `Completed (1)`. Combined with the stale files above, a broken run is indistinguishable from a good one. The EXECUTE PIPELINE summary now marks such models `⚠`, prints every phase/part error, and warns that files for a failed phase are left over from an earlier run.

### What the run did confirm

- **The F-mix seeding fix is in and working.** Run 1 → run 2 (`hash()` seeding) differed on essentially all 80 prompts; run 2 → run 3 (crc32) differs on 2, and the summary statistics are identical to four decimal places.
- 26/26 output files pass the JSON gate.
- Part G again wrote `part_g_v11_scramble.json`, Diverse loaded 84 prompts across 8 regimes, SpecB again converted 80 at the `standard` tier.

### A reproducibility fact worth recording

Comparing the freshly-written conditions between runs 2 and 3 — same pod, same GPU, same precision, same seeds — a small number of greedy continuations flip: F-attenuate 1/80, F-mix 2/80, F-transplant 7/168, and `F_norm_recipient` moves in the last significant figures. This is ordinary GPU-level nondeterminism (reduction order, kernel selection), not a code defect, and the aggregates absorb it — F-mix's summary was byte-identical across the two runs despite two text flips. Worth stating precisely because the README's existing caveat is about *precision*: this is run-to-run at fixed precision.

### Post-fix verification: all four conditions executed on CPU

Rather than leave the `@dataclass` fix resting on static analysis, the F-dynamics path was executed. `scripts/vault_utils/fdynamics_cpu_smoke.py` builds a 24-dimensional, 4-layer random model with a hand-written greedy `generate` and runs all four conditions through the real module code in a few seconds, on CPU, with no weights.

**4/4 conditions execute end to end** — `run_passive_F_tracking`, `run_step_offset_F_injection`, `compute_F_topK_basis` + `run_F_subspace_rotation_experiment`, and `compute_F_dynamic_temporal_basis` + rotation — each returning a well-formed summary that serialises to standards-compliant JSON.

Two semantic invariants also hold, and they are the ones that matter:

- **Identity-hook invariant:** passive tracking uses an identity `modify_F_fn`, so it must reproduce the baseline for every prompt. `baseline_match_pct = 100.0`. Anything less would mean the hook's read/decompose/write-back cycle perturbs the forward pass.
- **k=0 invariant:** step-offset injection at k=0 feeds each step its own F, so it must be identical to baseline. `identical_pct = 100.0` at k=0 and `0.0` at k=1. That is exactly the "the intervention infrastructure is lossless" claim behind Figure 9B, now demonstrated on an independent model.

A static pass supports this: the whole F-dynamics call graph resolves — every module-local callee exists with a matching signature, the only unresolved names are the function parameters `model_fn`, `modify_F_fn` and `tokenizer`, and both dataclasses in the subgraph carry their decorator.

### Still unexercised

The conditions have not run against **real weights through transformers' own `generate`**. Run 3 established that the real generate path does reach the hook — the crash came from inside it — so the integration is sound up to that point, and the CPU harness covers everything after. What remains untested is numerical behaviour on a real model and the notebook-side Diverse plumbing under live conditions.

**A fourth GPU run with `TEST_MODE_INCLUDE_FDYNAMICS = True`, starting from an empty `results/`, is still the right final gate** — it is the only test of the Diverse step-offset path behind Table 5 end to end. The CPU harness makes it very likely to pass rather than a coin flip.

## Fourth test run + defaults realignment, 2026-08-06

**`test_run_4/`** (results only, no notebook) is a clean standard smoke run: 23 files, all timestamps in a single contiguous 21:01–21:09 window — so the stale-leftover problem is gone — and 23/23 pass the JSON gate. **It did not exercise the F-dynamics path:** zero `passive_F_tracking`, `F_topK_rotation`, `F_dynamic_rotation`, `step_offset_F_injection` or `Continuation-Diverse` files, and the write order runs F-transplant → D-scramble with no passive-tracking step between them, so `TEST_MODE_INCLUDE_FDYNAMICS` was left at its shipped `False`. The F-dynamics path therefore remains verified only by `scripts/vault_utils/fdynamics_cpu_smoke.py`.

Against run 2, 20 of 23 files are identical. The three that differ are all explained: `F_mix` (crc32 seeding replaced `hash()`), `per_prompt_trajectories` (timestamp only), and `resolution_geometry_v11` — which exposed one more defect.

**Part J's shuffled PCA control was unseeded.** It drew from `np.random.permutation`, the global generator, so it was the only geometry output that varied run to run (2nd–3rd decimal on every `shuffled_control` bin) and it advanced the global RNG state for everything downstream. Now drawn from a local `RandomState` seeded by `RESOLUTION_SHUFFLE_SEED` (42) plus the layer index, so different layers still get different permutations and reruns match. The shuffled control is a diagnostic — `grep -c shuffl Paper_1_v63.md` returns 0 — so no reported number moves.

### Defaults now reproduce the paper

Adopted 2026-08-06 on Nelson's instruction that *"the initial run set by the script should match what is in the paper."* Two changes:

- **`CONTINUATION_F_TESTS["F_early"]` → `True`.** It shipped `False` while `F_standard` (70% depth) shipped `True` — backwards: the ~12% depth condition is **Figure 7 in the main text** (§5.2), and the 70% condition is its depth control in Appendix B.5. Both are reported; both now ship on.
- **`GEOMETRY_PROMPT_SETS`: SpecB and Diverse_group → `True`.** Figure 3 has three panels (A SpecA, B SpecB, C Diverse), and the §4.3 rank-matched PCA control and §4.4 Cohen's *d* hierarchy are reported across all three. `Diverse_regime` stays off — the regime-stratified geometry is a diagnostic Paper 1 does not report.

A reader should now not have to enable anything to reproduce a reported result; only `MODELS_TO_RUN`, since the testbed is 18 models and the default is one.

### "WHAT WILL RUN" summary replaces the option dump

The per-model phase opened with a 72-line dump of every `PIPELINE_OPTIONS` key. It is now a compact block grouping each enabled measurement under the section, figure or table it feeds — and, deliberately, a final group headed **"Enabled but NOT reported in Paper 1"** listing the Part G transplant conditions, F-transplant, Part M spirality and the Part E / Part H diagnostics.

Suppressing those lines was considered and rejected: they describe work that actually runs and writes files, and a silently-running condition is precisely the failure mode that let the dead F-dynamics port survive a month. Labelling them costs four lines and removes the ambiguity instead of hiding it.

### New switch: `TEST_MODE_FDYNAMICS_ONLY`

Set alongside `TEST_MODE_INCLUDE_FDYNAMICS` to narrow the smoke test to the F-dynamics path: geometry, Part G and every non-F-dynamics continuation condition off; bases and baselines still run because the four conditions are computed against them. Verified: 30 enabled `run_*` options drop to 4, geometry parts 14/15 → 0/15. This is the configuration for the outstanding F-dynamics GPU test.

## Fifth test run, 2026-08-06 (`test_run_4/test_run_5/`) — the Table 5 path finally runs

`TEST_MODE_INCLUDE_FDYNAMICS = True` + `TEST_MODE_FDYNAMICS_ONLY = True`. 10.1 min. The focused mode worked exactly as intended — every geometry phase, Part G and all six non-F-dynamics continuation conditions skipped.

**The step-offset F injection ran on Diverse for the first time.** `gemma_2b-Continuation-Diverse-step_offset_F_injection_d70.json`, 168 rows, k ∈ {0, 1}. This is the wiring fix from the very first audit — the one that made Table 5 reproducible — executing end to end against real weights. Passive F-tracking also ran on both sets (SpecB τ = 0.78 ± 0.25, asymptote 0.499, R² 0.767; Diverse τ = 0.63 ± 0.20, asymptote 0.428, R² 0.817), and the F-topK basis built at rank 36. All outputs pass the JSON gate.

**The k=0 dose-response reproduces the published shape:** k=0 gives 0% immediate divergence and a mean divergence token of 8.75 of 16; k=1 gives 28.6% immediate and 0.71. Same contrast as Figure 9B.

### The error-surfacing fix proved itself

The run ended with `⚠ gemma_2b (10.1 min) — 1 phase/part error(s)` and the loud `PHASE / PART ERRORS` block. Under the old summary this would have printed `PIPELINE COMPLETE` / `Completed (1)` with the failure buried 200 lines up. The fix from run 3 did its job on its first live outing.

### The defect it caught: a print statement destroyed a completed experiment

```
F_topK_rotation: 100%|██████████| 80/80 [01:38<00:00,  1.23s/it]
TypeError: unsupported format string passed to NoneType.__format__
  pds_continuation.py:2799  run_F_subspace_rotation_experiment
```

`compute_divergence_summary` **deliberately** sets `mean_divergence_token`, `median_` and `std_` to `None` when nothing diverged (line 819). The verbose print used `summary.get("mean_divergence_token", float("nan"))` — but the key *exists* and holds `None`, so the default never applies and `f"{None:.1f}"` raises. The F-topK experiment had already finished all 80 prompts; the exception threw away 98 seconds of completed work and took the whole SpecB continuation phase with it, so neither `F_topK_rotation` nor `F_dynamic_rotation` was written.

Fixed with a `_num()` helper that maps `None` to the default. A sweep of both modules and the notebook for the same shape found the step-offset print already guards correctly (`if mean_div is not None`), and passive tracking is safe because its keys are *absent* rather than None when the fits fail. Two notebook prints — Part A's `PR_P` and Part C's `cohens_d` — had the same latent trap and were hardened with a `_f()` formatter; they sit inside the per-part `try/except`, so a print failure there would have marked a **successfully computed part as failed**.

The pattern is worth naming: the untested code path carried the untested bug. `run_F_subspace_rotation_experiment` is reachable only from F-topK and F-dynamic, the two conditions that had never executed.

### Two observations, neither a defect

**The identity-hook "should be 100%" message is miscalibrated.** Passive tracking printed `Baseline match rate: 87.5% (should be 100% — verifies hook is truly read-only)`, and k=0 gave 90.5% identical rather than 100%. This is not a regression: the **published** canonical runs show the same statistic ranging from 0% to 100% across models (gemma_9b 100%, llama_8b 100% on SpecB but 30% on Diverse, gemma_27b and both GPT-OSS 0%). Decomposing to P+D+S+F in float32 and summing back does not return bitwise-identical hidden states, and greedy decoding then flips a fraction of prompts. The message should be recalibrated — as written it will alarm every reader who runs the smoke test.

**Figure 9B's "lossless" claim was checked and holds.** The k=0 `identical_pct` in canonical is 0% for gemma_9b and 20% for mistral_7b — the two models the caption names — which looks like a contradiction until you read the companion field: `mean_divergence_token = 32.0`, `median = 32.0`, `std = 0.0`, against `n_generate = 32`. Divergence token equal to the generation length means *no prompt ever diverged*. `identical_pct` is a stricter string-equality flag; the divergence-token distribution is the measure the caption refers to, and at k=0 it is unanimous. No manuscript change needed.

### Still not exercised

`F_topK_rotation` and `F_dynamic_rotation` output. Both were killed by the print bug after computing successfully, so the remaining risk is confined to the summary/save step that follows them. One more focused run closes it.

**Note on run hygiene:** this run loaded SpecB bases and baselines from cache (`use_cached_bases` / `use_cached_baselines`), so `results/` had not been cleared. Harmless here — bases are deterministic and baselines were byte-identical across runs 1 and 2 — but the next run should start clean so the file set is unambiguous.

## Sixth test run, 2026-08-06 (`test_run_5/`) — all four conditions produce output, and a summary bug surfaces

18.5 min, **no phase errors**, 12 files, all passing the JSON gate. Every F-dynamics condition ran and wrote output for the first time: `F_topK_rotation` (rank 36) and `F_dynamic_rotation` (rank 37) on SpecB, `step_offset_F_injection` on both sets, `passive_F_tracking` on both. The print fix from run 5 held — the exact line that crashed now prints `first_div=nan` and the phase completes.

### The defect: F-topK and F-dynamic summaries were empty

Both rotation files were written with `identical_pct = 0.0`, `immediate_divergence_pct = 0.0`, `mean_divergence_token = None` — while the per-prompt rows were perfectly good (13 identical, 67 diverged, first-divergence tokens spread 0–15).

`run_F_subspace_rotation_experiment` splats the comparison onto the row (`**comp`) and leaves `"effect"` as `None`; every other condition nests it under `"effect"`. `compute_divergence_summary` read only `r.get("effect", {})`, so for these two conditions it aggregated an empty dict. This is also why the run-5 print crashed: `mean_divergence_token` was `None` for *every* row of these conditions, not occasionally.

Fixed in `compute_divergence_summary` — it now falls back to the row itself when `"effect"` is not a dict, which repairs new output and also recomputes correctly from existing files. The row schema is deliberately left alone, since the canonical files carry the top-level shape and downstream analysis may rely on it.

Recomputed from run 6's own per-prompt rows, the two summaries become sensible and paper-consistent: F-topK 16.2% identical, 13.4% immediate, mean divergence 5.43 (canonical per-model range 1.7–11.6); F-dynamic 0% identical, 100% immediate, mean 0.00 — matching §7.3's finding that rotated F-dynamic prevents coherent generation.

### Figure 8's number was checked against this and holds

The same empty-summary shape is present in **canonical**: `llama_8b`'s published `F_topK_rotation` summary reads `mean_divergence_token = None`, though the other nine models carry correct values. Recomputing the statistic from per-prompt rows across all ten models gives a cross-model mean of **6.52**, exactly the value Figure 8 reports for $\overline{F_{\mathrm{topK}}}$. The published figure is right; only the per-file `summary` block was unreliable, and only for some files. No manuscript change.

### Per-prompt-set routing for the F-dynamics conditions

Run 6 also answered the "why is F-tracking running twice?" question: `RUN_PASSIVE_F_TRACKING` was a single global switch copied into **both** the SpecB and Diverse option dicts, so Phase 6 and Phase 8 each ran it (2:00 + 2:08). Within a phase it runs exactly once — the driver's `if run_passive_tracking or run_F_dynamic_rot:` guard reuses `tracking_results` for the F-dynamic basis, which is correct.

But only the SpecB run is wanted. Canonical shows each condition on exactly one set: `F_topK_rotation` and `F_dynamic_rotation` SpecB (10 files each), `step_offset_F_injection` **Diverse only** (12), and `passive_F_tracking` on SpecB for precisely the ten instruct models Figure 9A reports. The Diverse passive files exist but cover a different twelve-model mix including base models, and nothing in a run consumes them — step-offset does not depend on tracking.

Added `F_DYNAMICS_PROMPT_SET`, an explicit and editable mapping from condition to prompt set, defaulting to what the paper reports. This removes a redundant Diverse passive-tracking pass and a redundant SpecB step-offset pass — about 6 of run 6's 18.5 minutes — and stops the release writing files the paper does not use.

### TEST_MODE banner

The `TEST_MODE_FDYNAMICS_ONLY` notice printed before the general TEST_MODE line. The block is now one banner in reading order: what the mode is, what its scope is, then prompt and generation counts.

## Seventh test run, 2026-08-06 (`test_run_6/`) — clean, and the code is done

`TEST_MODE_INCLUDE_FDYNAMICS` + `TEST_MODE_FDYNAMICS_ONLY`. **11.0 min, no errors, no `PHASE / PART ERRORS` block, 10 files, all passing the JSON gate.** The only `⚠️` in the log is the routine Gemma-2 bfloat16 loader notice.

Everything fixed since run 6 is confirmed working:

- **Per-prompt-set routing.** Exactly one passive-tracking pass (SpecB) and exactly one step-offset pass (Diverse). The redundant Diverse passive and SpecB step-offset passes are gone, and the run dropped from 18.5 min to 11.0.
- **Summaries populate.** The line that crashed in run 5 and printed empty in run 6 now reads `F_topK_rotation (rank=36): first_div=5.4, imm_div=13.4%, identical=16.2%` and `F_dynamic_rotation (rank=37): first_div=0.0, imm_div=100.0%, identical=0.0%`.
- **TEST_MODE banner** prints as one block in reading order.
- **WHAT WILL RUN** lists the four target conditions with their paper references and nothing else.

Final numbers, all paper-consistent: F-topK mean first-divergence 5.43 (canonical per-model range 1.7–11.6, cross-model mean 6.52); F-dynamic 100% immediate divergence, mean 0.00 (§7.3: rotated F-dynamic prevents coherent generation); passive tracking τ = 0.78 ± 0.25, asymptote 0.499, R² 0.767; step-offset k=0 → 0% immediate, k=1 → 28.6% immediate (Figure 9B's contrast).

**Every measurement the paper reports has now executed and produced sane output.** Seven GPU runs; each one found something. Three of the defects lived in code reachable only from the four F-dynamics conditions — the untested path carried the untested bug, three times over.

### Where the validation data went

`data/_release_validation/gemma_2b_testmode_2026-08-06/` — the run-7 outputs, with a README recording the model, the reduced parameters and the code state. **Deliberately not `data/canonical/`.** These files carry canonical filename grammar but ran at 16-token generation against the paper's 64 and k ∈ {0,1} against {0,1,2,3,5}; dropped into canonical they would be indistinguishable from research data to any script that enumerates a directory. The concrete risk: `glob("data/canonical/*/*F_topK_rotation*.json")` matches exactly the ten models Figure 8 aggregates, and one bare-keyed validation file makes it eleven — the Figure 8 recomputation done during this review would have silently averaged in a 16-token run. Model keys are additionally suffixed `gemma_2b_testmode`, so no glob keyed on a testbed model name can reach them even if a copy escapes. Verified: the three canonical patterns still match 10 / 12 / 22 files, and `data/**/gemma_2b-*.json` matches 0.

## Reproduces

- **Patch 8 — the rank-matched PCA control.** `r_F = max(0, min(50, (n−1) − (P+D+S)))`, matching both the PCA `bin_F` and a PDSF F at that rank. This is what Paper 1 **v63** §4.3 cites; v62's figures came from the un-matched control.
- The 2026-08-03/04 rerun across 10 instruct models × 3 prompt sets, and the 2026-07-01/02 Mixtral and GPT-OSS 120B full-depth reruns.

## Produces

| Pattern | Note |
|---|---|
| `{model}-Geometry-{set}-resolution_geometry_v11.json` | 40 files promoted to `data/canonical/` 2026-08-04, superseding the February set. New fields: `basis_ranks.residual_rank_matched`, `residual_rank_common31`, `pca_control_budget`, `pca_control_rule`; `subspaces` gained matching blocks. |
| `{model}-Geometry-Diverse-group-resolution_geometry_v11_within_regime.json` | regenerated in the same run |

Superseded files and their dependent Q-entries: `data/_archive/README.md`.

## The data tier, added 2026-08-07

Scoped by one rule: **ship only what a pipeline run cannot reproduce.** Everything the notebook emits deterministically — the geometry parts, rank estimators, complexity profiles, rotation profiles — stays out, because a reader who wants it can clone and run. Three families fail that test and are therefore in.

| directory | contents | size | source |
|---|---|---|---|
| `data/classification/` | `all_classifications_v4.json` (2,950 cells), `all_classifications_v5_fdynamic_audit.json` (800 cells), `MR_classification_guide.md` | 2.5 MB | `agent_classification/results/` and `agent_classification/v5_audit/` |
| `data/behavioral_interventions/` | 121 files, `{model}-Continuation-{set}-{condition}.json`, across 10 conditions | 12 MB | `data/canonical/` |
| `data/part_g_derived/` | 4 files: per-prompt KL, its model summary, per-layer rotation-recovery angles, and the archived GPT-OSS 120B record | 0.4 MB | `revision_documents/_job2_scratch/`, renamed and given `_schema` headers |

The v5 corpus is taken from `v5_audit/`, not the duplicate in `results/` — that resolves the naming problem logged in the archive plan §8. The Part G scratch files were merged from their arbitrary a/b model splits into one file per family, ordered by model size, and each carries a `_schema` block giving layout, units, aggregation convention and Paper 1 scope.

**Verified 2026-08-07, recomputing from the shipped copies:**

- Table 3 and Table B.7-1 — all 25 cells, counts and percentages, from the two classification corpora under the four-category collapse.
- All 3,750 classification cells resolve to a generation pair in `behavioral_interventions/` on `(model, prompt_key, condition)`. None missing.
- Table A.4-1 — all 14 conditions, means and SEMs, to machine precision. F-mix 2.846, F-attenuate 0.047, the 60× gap, P-rotation exactly 0.000.
- Appendix A.4's rotation-recovery paragraph — every quoted range and per-model value, both prompt sets, both rotations.
- Appendix B.7.1 — D immediate divergence 64.0 ± 19.8%, S 28.2 ± 14.3%, D/S mean-token ratio 4.8×, and GPT-OSS 120B's partial anomaly (D 1.55 / S 3.80; D 28.8% / S 36.2%).
- `sanitize_json_nonfinite.py --check` over the whole tree: 130 files scanned, **zero** bare `NaN`/`Infinity` tokens. The four derived files were written with `allow_nan=False`. The only non-compliant JSON anywhere under `github/` is in `test_run/` — the pre-fix first run, gitignored.

**Deliberately excluded:** the 492 MB `Continuation-*-baselines` (redundant — every intervention row carries `baseline_text` and `scrambled_text` inline), the 1.6 GB raw Part G files, all superseded classification corpora (v1–v3, the pre-identity v4 backup), the manual-review working sheets, and the v11.1 KL extracts (`_job2_kl_extract_*`, `_job2_kl_div_*`) — which back nothing in the paper and would invite a reader to average the wrong pipeline generation.

**Two defects surfaced while verifying, both recorded rather than fixed:**

1. `llama_8b-Continuation-SpecB-F_topK_rotation.json` and `…-F_dynamic_rotation.json` carry empty or wrong `summary` blocks — the same `compute_divergence_summary` shape mismatch fixed in `pds_continuation.py` this session, frozen into two canonical files from the 2026-03-04 run. Per-prompt rows are intact; the other 107 fixed-schema files agree with their own rows exactly. The shipped README tells readers to recompute from `results`. Fixing the canonical files would mean regenerating them, which is a separate decision.
2. Table B.7-3 in the supplement (F-topK per-model divergence metrics) does not reproduce from these files. Its **Mean ± SD row does** — 6.52 and 23.0 both recompute exactly — but the per-model column does not, and its own mean-token column averages to 7.52, not the 6.52 it states. No artefact in the vault produces the per-model values. This is the same shape as the Table S1 problem in `Job2_E2_L2_angles_findings.md` §1b: a supplement table whose aggregate reproduces and whose per-model column does not. Not a blocker for the data tier; a blocker for rendering the supplement.

## Does NOT include

Any Paper 2 work. Paper 2 shares this lineage's modules but its post-release line is `scripts/V11.10 (Paper 2 post-release)/`.

Also not included, and still owed against Appendix E's promise of "all figure generation" code: the figure scripts. Four of ten figures have no generating script anywhere in the vault (Figures 6, 7, 8, 10), and Figure 6 was regenerated by hand for the v63 row swap. See `../revision_documents/Paper_1_supplementary_archive_plan.md` §4.

## Environment

bf16 on 2×H200 for the August run. Mixtral requires forced bf16 (the transformers MoE grouped-GEMM kernel is bf16-only); GPT-OSS 120B runs native MXFP4. Patches 1–4 exist because the auto-4-bit path masked bugs that only surface on hardware large enough to avoid it.

## Known defects

**None release-blocking as of the 2026-08-06 preparation pass**, subject to the GPU smoke test passing. `[VERIFIED 2026-08-06 — static verification only: cell parse, config-chain execution, prompt-path resolution, module compile. No model was loaded.]`

*Open, minor:* the notebook's Diverse continuation block pins `run_F_topK_rotation` and `run_F_dynamic_rotation` to `False` because the paper runs both on SpecB. If a later experiment wants them on Diverse, that is a one-line change, not a design constraint.

## Lineage

- Predecessor: the 2026-03-22 tree in `releases/`, never published.
- Snapshots in `versions/`: `v1_loader_patches_only_2026-07-02`, `v2_partJ_spectrum_layers_bug_2026-07-04`, `v3_pre_patch8_rankmatched_pca_2026-08-03`, `v4_post_rankmatched_run_2026-08-04`.
- Patch record (vault-only): `PATCHES.md`. Run specs: `../revision_documents/Job1_rankmatched_PCA_control_runspec.md`, `Job1_run_checklist.md`. Readiness audit: `../revision_documents/github_release_code_readiness_audit_2026-08-06.md`.

## To freeze this as a release

**Gate: has the tree actually been pushed to GitHub?** A release directory records what a reader can fetch. Freezing before upload creates a `releases/` entry that points at nothing public, and freezing a *different* state than what was uploaded is worse — it makes the vault disagree with the repository while both look authoritative. **Ask, and get a yes, before doing any of the steps below.**

Once the answer is yes:

1. Copy the **shipped subset** (per `.gitignore`) to `releases/paper_1-pdsf-residual-geometry-v1.0-{upload-date}/`, dated by the upload, not by the last edit. The existing `...-v1.0-2026-03-22/` directory needs renaming or a status note first, so that two directories do not both claim v1.0.
2. Replace this file with a frozen manifest — same fields, `Status: frozen`, provenance naming the uploaded repo and commit.
3. Add the row to `releases/README.md` and retire the "not yet frozen" line there.
4. Retarget `methods.md` **Release script** lines and any `scripts:` citation on a Stage A or B entry.
5. Verify the frozen copy against the uploaded repo — hash the files, not just the names. The 2026-03-22 lineage is the reason: its pre-upload staging copy differed from the tree beside it in exactly one file, the notebook, and nobody noticed for five months.

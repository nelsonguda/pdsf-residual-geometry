# Behavioral intervention outputs

121 files of generation pairs: for each prompt, the model's unperturbed continuation and its continuation under one PDSF intervention, plus the divergence measurement between them. These are the outputs the classification corpora in `../classification/` judge, and the inputs to the divergence statistics in §5.2, §6 and §7. Thirteen model keys appear — the ten instruction-tuned models the paper aggregates over, plus three base models — across ten intervention conditions.

Filenames are `{model}-Continuation-{prompt_set}-{condition}.json`.

## Conditions

| condition | prompt set | intervention | models | backs |
|---|---|---|---|---|
| `D_scramble_d70` | SpecB (80) | Haar-random rotation within D at 70% depth | 10 instruct + 3 base | §6, Tables 3–4 |
| `S_scramble_d70` | SpecB (80) | same, within S | 10 instruct + 3 base | §6, Tables 3–4 |
| `F_topK_rotation` | SpecB (80) | rotation within F's top-k variance basis | 10 instruct | §6.4, Figure 8 |
| `random_control_d70` | SpecB (80) | rotation within a random subspace of matched rank | 10 instruct + 3 base | §6, control |
| `F_dynamic_rotation` | SpecB (80) | rotation within F's temporal-variation basis | 10 instruct | §7.3, Figure 9 |
| `F_mix_d70` | Diverse (84) | coordinate permutation + sign flips **within** the prompt's own F vector, at 70% depth | 10 instruct + 1 base | §5.2, App. B.5 |
| `F_attenuate_d70` | Diverse (84) | F scaled toward zero at 70% depth | 10 instruct + 1 base | §5.2, App. B.5 |
| `F_early_mix` | Diverse (84) | F-mix at ~12% depth | 10 instruct + 1 base | §5.2, App. B.5 |
| `F_early_attenuate` | Diverse (84) | F-attenuate at ~12% depth | 10 instruct + 1 base | §5.2, App. B.5 |
| `step_offset_F_injection_d70` | Diverse (20 prompts; 84 in Gemma 9B) | F from generation step *t−k* injected at step *t* | 10 instruct + 2 base | §7.2, Table 5 |

**F-mix is not a transplant.** It permutes coordinates and flips signs inside a prompt's *own* F vector, preserving the norm while destroying the coordinate structure. Nothing is copied between prompts. Cross-prompt F substitution is F-transplant, a different condition that was removed from Paper 1 at v30 and does not appear here.

The `_d70` files carry `depth_pct` in their metadata; the early files record `depth_pct: 12`. The step-offset files hold one row per (prompt, k), so a 20-prompt × k∈{0,1} run is 40 rows — read `n_prompts_subsampled` and `k_values` from the metadata rather than counting rows.

Base-model files (`*_base`) support the base-vs-instruct comparison and are not part of the ten-model aggregates the paper reports. Six extra `llama_8b` files run a condition on the *other* prompt set (e.g. `llama_8b-Continuation-Diverse-D_scramble_d70.json`); they were cross-checks and are likewise outside the reported aggregates. Filter on the ten instruct models and the prompt set named above to reproduce a published number.

## Two record schemas

This matters — reading the wrong one silently yields nothing.

**The `_d70` family** (`D_scramble`, `S_scramble`, `random_control`, `F_mix`, `F_attenuate`, `F_early_*`, `step_offset`) nests measurements under `effect`, and names the intervened text `scrambled_text`:

```json
{"group_id": "group_01", "variant_id": "v0", "prompt": "...",
 "baseline_text": "...", "scrambled_text": "...", "hook_fired": true,
 "effect": {"first_divergence_token": 0, "first_divergence_word": 0,
            "identical": false, "prefix_same_frac_tokens": 0.0}}
```

**The F sub-basis rotation family** (`F_topK_rotation`, `F_dynamic_rotation`) puts the same fields flat on the record, and names the intervened text `generated_text`:

```json
{"group_id": "group_01", "variant_id": "v0", "prompt_key": "group_01_v0",
 "baseline_text": "...", "generated_text": "...", "condition": "F_topK_rotation",
 "f_sub_rank": 50, "first_divergence_token": 10, "identical": false}
```

Read defensively: `(row.get("effect") or row)["first_divergence_token"]`.

`first_divergence_token` is `null` where the intervened continuation is byte-identical to the baseline — check `identical` rather than treating `null` as zero. The prompt key used by the classification corpora is `group_id + "_" + variant_id`, which the second schema also stores directly as `prompt_key`.

## Summary blocks: recompute, don't trust

Each file carries a `summary` block (`n_prompts`, `identical_count`, `immediate_divergence_pct`, `mean_divergence_token`, …). **Two files' summary blocks are empty or wrong** while their per-prompt rows are intact:

- `llama_8b-Continuation-SpecB-F_topK_rotation.json` — summary reports `mean_divergence_token: null` and `immediate_divergence_count: 0`; the 80 rows give mean 7.84 and 11 immediate divergences.
- `llama_8b-Continuation-SpecB-F_dynamic_rotation.json` — summary reports `immediate_divergence_count: 0`; all 80 rows diverge at token 0.

The cause is a shape mismatch in the summarizer, which read the `effect` key against the flat schema above and found nothing. The defect is in the aggregate only; no generation text or per-prompt measurement is affected. The remaining 107 fixed-schema files agree with their own rows exactly (verified 2026-08-07). **Compute statistics from `results`, not from `summary`.**

## What reproduces, verified 2026-08-07

Over the ten instruct models on SpecB, recomputing from `results`:

- **D-scramble immediate divergence 64.0 ± 19.8%, S-scramble 28.2 ± 14.3%** — Appendix B.7.1, exact.
- **D/S mean-first-divergence-token ratio 4.8×** — Table 4, exact.
- **GPT-OSS 120B's partial anomaly:** mean divergence token D = 1.55 < S = 3.80 while immediate-divergence rates reverse, S = 36.2% > D = 28.8% — Appendix B.7.1, exact.
- **F-topK cross-model mean first-divergence token 6.52** — Figure 8 / Table B.7-3 summary row, exact.

## Known irregularity

`mistral_7b-Continuation-SpecB-F_dynamic_rotation.json` records `f_dynamic_rank: 10`; every other model's F-dynamic file records 64. The rank is the dimensionality of F's temporal-variation basis and is data-determined per model, so this is not necessarily an error, but it is a large outlier and any per-model comparison involving Mistral 7B's F-dynamic condition should account for it.

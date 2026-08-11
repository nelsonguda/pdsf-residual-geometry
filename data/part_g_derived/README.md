# Part G derived extracts

Paper 1 §5.1 reports a five-orders-of-magnitude KL hierarchy across single-pass PDSF interventions (Figure 6, Table A.4-1), and Appendix A.4 reports how far a one-time rotation is repaired before readout. Both come from the Part G intervention files, `{model}-Geometry-{set}-part_g_v11_regime_transplant.json` — **1.6 GB across 34 files, with single files reaching 93 MB**, past what GitHub will hold.

These four files are the per-prompt arrays extracted from those sources. At 400 KB they reproduce every published number exactly, so the tables are checkable without anyone downloading the source.

| file | what it holds |
|---|---|
| `partG_kl_divergence_per_prompt.json` | per-prompt, per-scramble KL for 14 conditions × 10 models, SpecA at 40 prompts |
| `partG_kl_divergence_model_summary.json` | the aggregation of the above: mean, SEM, median, min, max per condition |
| `partG_rotation_recovery_angles.json` | per-layer D and S rotation angles after intervention, both prompt sets, 10 models |
| `partG_kl_divergence_gpt_oss_120b_archived.json` | the superseded 2026-02-21 GPT-OSS 120B record (see below) |

Each file carries a `_schema` block stating its layout, units, aggregation convention and scope. Read that first.

## The aggregation convention

Mean over a prompt's scrambles first, then over the 40 prompts within a model, then an **unweighted** mean across the ten instruction-tuned models. SEM is the population standard deviation (`ddof = 0`) divided by √10. No trimming, no median, no outlier filter, no prompt subsetting. `rotation` and `mix` carry three scrambles per prompt; `attenuate` and `transplant` carry one.

```python
import json, statistics, math
kl = json.load(open('partG_kl_divergence_per_prompt.json'))['by_model']
per_model = [statistics.fmean([statistics.fmean(s) for s in kl[m]['conds']['F|mix']]) for m in kl]
statistics.fmean(per_model)                                  # 2.846 nats
statistics.pstdev(per_model) / math.sqrt(len(per_model))     # 0.556
```

Verified 2026-08-07: all 14 conditions reproduce `partG_kl_divergence_model_summary.json` to machine precision, and the summary matches Table A.4-1 — including F-mix 2.846, F-attenuate 0.047, the 60× gap §5.1 reports, and P-rotation exactly 0.000.

## Scope: 14 conditions, of which Paper 1 uses 8

The source protocol has 14 conditions. Paper 1 reports 8. The six omitted are all transplant conditions — `F|transplant_within`, `F|transplant_cross`, `F|transplant_null`, `D|transplant`, `S|transplant`, `P|transplant` — which were removed from the paper at v30. They are retained in these files so the extract reproduces its source report in full, and so their exclusion is visible rather than silent. **They are not Paper 1 results and should not be added to its tables, figures or condition counts.**

## Why an archived GPT-OSS 120B file is included

GPT-OSS 120B was re-run on 2026-07-02 under native MXFP4. Paper 1 reports the post-rerun canonical values. The 2026-02-26 internal report that first established the aggregation convention saw the earlier 2026-02-21 file, and substituting that record reproduces its fourteen rows — means and SEMs — exactly. Publishing it makes the difference between the two inspectable instead of asserted. It is not a Paper 1 value; its `_schema` block says so.

## Rotation recovery angles

`partG_rotation_recovery_angles.json` gives, per model and prompt set, the angle of D and of S relative to their unperturbed orientation at every post-intervention layer. Two positions are read from it:

- **Peak** — the angle at the intervention layer itself (the key equal to `scramble_layer`), not the maximum over layers.
- **L−2** — the penultimate layer, `layers[-2]`. The final layer is excluded because the LM-head projection inflates all residual perturbations uniformly; Mistral 7B, for instance, reads 0.59° at L−2 and 39.48° at the final layer under D-rotation on SpecA.

Verified 2026-08-07: every value quoted in Appendix A.4 reproduces from this file — SpecA D-rotation peak 100.5–103.5° and L−2 0.19–1.72° in the seven dense models, with GPT-OSS 20B 1.76°, Mixtral 4.30°, GPT-OSS 120B 4.70°; Diverse D-rotation L−2 0.30–1.58° dense, 3.24° / 4.77° / 29.98°; and the S-rotation figures for both sets.

## Not included

`part_g_v11_scramble.json` (pipeline v11.1 — 9 models, SpecA at 20 prompts, a single F-transplant condition, 12 conditions) is a superseded earlier generation and is **not** the source of any Paper 1 KL figure. Extracts from it exist in the working vault and are deliberately not published, since shipping them alongside these would invite a reader to average the wrong generation.

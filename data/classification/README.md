# Behavioral classification corpora

Paper 1 §6 and §7.3 classify what an intervention *does to the output*, not just whether the output changed. That classification is a supervised judgment over 3,750 generation pairs, and it is the one part of the paper that cannot be reproduced by re-running the pipeline. Both corpora are published here in full, including the reasoning text for every cell.

| file | cells | canonical for | backs |
|---|---|---|---|
| `all_classifications_v4.json` | 2,950 | D-scramble, S-scramble, F-topK, Random | Table 3, Table 4, Table B.7-1 (four of five rows), §6.4 |
| `all_classifications_v5_fdynamic_audit.json` | 800 | F-dynamic rotation | Table B.7-1 (F-dynamic row), §7.3 |
| `MR_classification_guide.md` | — | the taxonomy itself | Appendix A.5 |

The v4 file also carries a partial F-dynamic subset (193 cells) from an earlier pass. **It is superseded by the v5 audit and must not be used** — the v5 audit re-classified all 800 F-dynamic cells under uniform v4 rules with a baseline-degeneracy screen. Filter on `condition != "F-dynamic"` when working with v4.

## The taxonomy

`MR_classification_guide.md` is the full instrument. The paper reports a four-category collapse:

| collapsed category | fine-grained `final_subtype` values |
|---|---|
| Equivalent | `Ident`, `MR-1` |
| Variant | `MR-2` |
| Frame-shift | `MR-3`, `MSC` |
| Failure | `FM`, `BSS`, `TD` |

`Ident` cells — where the intervened text is byte-identical to the baseline — are recovered from the continuation files in `../behavioral_interventions/` rather than being judged, since there is nothing to judge.

## Reproducing Table 3 and Table B.7-1

```python
import json, collections

COLLAPSE = {'Ident':'Equivalent', 'MR-1':'Equivalent', 'MR-2':'Variant',
            'MR-3':'Frame-shift', 'MSC':'Frame-shift',
            'FM':'Failure', 'BSS':'Failure', 'TD':'Failure'}

v4 = json.load(open('all_classifications_v4.json'))
for cond in ['D-scramble', 'S-scramble', 'F-topK', 'Random']:
    print(cond, collections.Counter(
        COLLAPSE[r['final_subtype']] for r in v4 if r['condition'] == cond))

v5 = json.load(open('all_classifications_v5_fdynamic_audit.json'))['records']
kept = [r for r in v5 if not r['excluded_baseline_degenerate']]   # 756 of 800
print('F-dynamic', collections.Counter(COLLAPSE[r['final_subtype']] for r in kept))
```

Verified 2026-08-07: this reproduces all 25 cells of Table B.7-1 — every count and every percentage — including D-scramble's 42.3% Frame-shift, F-topK's 6.5%, Random's 1.5% and F-dynamic's 92.3% Failure.

## The baseline-degeneracy screen (v5 only)

44 of the 800 F-dynamic cells were excluded because the *baseline* generation was already degenerate — list repetition, tail collapse — so an intervention could not be said to have caused a failure. All 44 are GPT-OSS. Each carries `excluded_baseline_degenerate: true`, a `baseline_screen_score`, the four component scores `S1..S4`, and a written `exclusion_reason`. The F-dynamic denominator in the paper is the screened 756; the unscreened 800-cell Failure rate is 87.2%, reported in the Table B.7-1 caption.

The v4 corpus has no equivalent screen — it predates the instrument. This is a real asymmetry between the two corpora and is why they are kept as separate files rather than merged.

## Provenance and its limits

Both corpora were classified by a language model (`claude_opus_4_7` via Cowork) working from `MR_classification_guide.md`, with human spot-checking by the author: 10 cells per pass × 6 passes plus a 50-cell calibration for v5, and per-batch review sheets for v4. `all_classifications_v5_fdynamic_audit.json` carries the audit trail in its `metadata` block; every record in both files carries a free-text `reasoning` field and a `confidence` level, and `borderline: true` marks cells the classifier itself flagged as near a category boundary.

This is a supervised-LLM classification with human audit, not independent multi-rater human coding, and no inter-rater reliability statistic is reported because there is only one rater. The reasoning text is published precisely so a reader can disagree with individual calls and recount.

## Joining to the generation pairs

Every cell keys on `(model, prompt_key, condition)`. `prompt_key` is `group_id + "_" + variant_id` in the continuation files:

```python
fam = {'D-scramble':'D_scramble_d70', 'S-scramble':'S_scramble_d70',
       'F-topK':'F_topK_rotation', 'Random':'random_control_d70',
       'F-dynamic':'F_dynamic_rotation'}[record['condition']]
path = f"../behavioral_interventions/{record['model']}-Continuation-SpecB-{fam}.json"
row  = next(r for r in json.load(open(path))['results']
            if f"{r['group_id']}_{r['variant_id']}" == record['prompt_key'])
row['baseline_text'], row['scrambled_text']
```

Verified 2026-08-07: all 3,750 cells across both corpora resolve, none missing.

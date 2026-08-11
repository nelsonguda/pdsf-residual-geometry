# Behavioral classification corpora

Paper 1 §5.2, §6 and §7.3 classify what an intervention *does to the output*, not just whether the output changed. For the persistent-rotation conditions that classification is a supervised judgment over 3,750 generation pairs — the one part of the paper that re-running the pipeline cannot reproduce. Everything is published here in full, including the reasoning text for every judged cell.

| file | cells | canonical for | backs |
|---|---|---|---|
| `all_classifications_v4.json` | 2,950 | D-scramble, S-scramble, F-topK, Random | Table 3, Table 4, Table B.7-1 (four of five rows), §6.4 |
| `all_classifications_v5_fdynamic_audit.json` | 800 | F-dynamic rotation | Table B.7-1 (F-dynamic row), §7.3 |
| `f_structure_classification.json` | 1,680 | F-mix, F-attenuate at ~12% depth | Table B.6-1, §5.2 |
| `f_structure_70pct_review.json` | 739 | the ~70%-depth adjudication | Appendix B.6's reclassification claim |
| `classify_f_structure.py` | — | regenerates `f_structure_classification.json` | — |
| `MR_classification_guide.md` | — | the taxonomy itself | Appendix A.5 |

**Two instruments produced these labels, and conflating them would misread the evidence.**

The **v4 and v5 corpora are supervised judgments** — a language model read each generation pair against a written rubric, with human spot-checking. Nothing regenerates them.

The **F-structure labels are rule-based**. A deterministic classifier assigns them from the text pair; `python3 classify_f_structure.py` reproduces `f_structure_classification.json` exactly, and every record names the rule that fired. `f_structure_70pct_review.json` is mostly the same rule pass — of its 739 records, 8 carry a hand-assigned label, 58 were settled by a secondary rule pass, and 673 are primary rule output. Check the `note` prefix (`MANUAL:` / `RESOLVED:` / neither) to tell which.

That is why the classifier ships alongside its output. **The script is the claim.** A reader who doubts Table B.6-1 is really doubting how zero lexical overlap is treated in non-Latin scripts, or where the degeneracy thresholds sit — visible only if the rules and the per-record `rule` field are both in front of them. Read the header of `classify_f_structure.py` first; it is the place to push back.

**A defect found and fixed on 2026-08-11 — read this before comparing against an earlier draft.** `is_degenerate()` used to measure length with `text.split()`. Chinese and Japanese are written without inter-word spaces, so a fluent paragraph scored as a one- or two-token fragment and was labelled `Failure_mode`. It now measures over `tokenize()`, which falls back to per-character tokens when a text's characters-per-whitespace-token exceeds 12 — a threshold with a wide margin on both sides (≤ 7.5 at the 95th percentile for every space-separated regime in this corpus, including Korean, which does use spaces; 30+ for Mandarin and Japanese).

**98 of the 1,680 ~12%-depth trials moved, all of them out of `Failure_mode`** (86 to `Minor_reframe`, 12 to `Mode_stance_change`); none moved in, and `Identical` and `Topic_drift` are unchanged. 97 were Mandarin, Japanese or Korean regimes; the one English-regime cell had an intervention that switched into Chinese. The same fix applied as a delta at ~70% depth moves 71 more. Superseded distributions are recorded in `classify_f_structure.py`'s REVISION note and in each file's `_schema`.

Two `too_few_words` cells survive at ~12% depth and both are genuine fragments (`'assistant\n\n古い井戸'`, `'¿Qué encontró?'`).

**What this does not fix.** `Mode_stance_change` has a milder problem of the same kind: most of its cells fire on `register_shift`, where a generic meta-phrase regex (`Here are`, `This is`, …) matches one side of an otherwise near-identical pair, mostly in Latin-script and code regimes. And against the only independent human reading these pairs ever received — a 120-item development sample, stratified toward hard cases — the classifier agrees on **59/80** of the F-mix/F-attenuate items it covers. That figure predates this fix and is a floor, not an unbiased estimate. `Identical` and `Minor_reframe` remain the rows to trust most.

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

## The F-structure interventions (Table B.6-1)

`f_structure_classification.json` covers F-mix and F-attenuate at ~12% depth: 10 instruction-tuned models × 84 Diverse prompts × 2 conditions = 1,680 trials. Its five outcome categories (Identical, Minor_reframe, Mode_stance_change, Topic_drift, Failure_mode) are the six-category heuristic's successor, not the four-category collapse used for the persistent-rotation conditions — the two taxonomies are not interchangeable.

To reproduce Table B.6-1, or to check the classifier against your own reading:

```
cd data/classification
python3 classify_f_structure.py
```

It reads only `../behavioral_interventions/`, needs nothing outside the standard library, and prints its counts against the published ones. Verified 2026-08-10: all ten cells match.

`f_structure_70pct_review.json` is a different kind of record. At ~70% depth the two heuristic categories with the worst error rate — `topic_drift` and `catastrophic_redirect` — were re-read one by one and reassigned; those 739 adjudications (F-mix and F-attenuate only; the source review also covered F-transplant, which is out of scope for Paper 1) are what the file holds, with the original label, the reviewer's new label, the Jaccard score the heuristic used, and the stated reason. Because a person made those calls, the ~70% distribution is **not** script-reproducible — running the classifier on the ~70% files gives close but not identical counts, and the difference is exactly the adjudication.

## Provenance and its limits

The v4 and v5 corpora were classified by a language model (`claude_opus_4_7` via Cowork) working from `MR_classification_guide.md`, with human spot-checking by the author: 10 cells per pass × 6 passes plus a 50-cell calibration for v5, and per-batch review sheets for v4. `all_classifications_v5_fdynamic_audit.json` carries the audit trail in its `metadata` block; every record in both files carries a free-text `reasoning` field and a `confidence` level, and `borderline: true` marks cells the classifier itself flagged as near a category boundary.

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

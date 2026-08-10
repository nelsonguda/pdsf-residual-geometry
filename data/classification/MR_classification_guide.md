# Minor-reframe Sub-typing Classification Guide

*Guide for agent-based hand-review of D-scramble, S-scramble, F-topK, F-dynamic, and Random continuation cells from the Paper 1 v30 residual-stream intervention study. Author: Nelson Guda. Date prepared: 2026-04-22.*

---

## 0. Task summary

You are classifying pairs of (baseline, intervened) text outputs from a language model. Each pair was produced by running the same prompt through a baseline forward pass and a forward pass where a specific PDSF subspace was rotated or scrambled at ~70% depth. Your job is to classify the **relationship between baseline and intervened** into one of 8 categories, with special attention to sub-typing the "Minor reframe" cases.

The 8 categories are:

- **Ident** — Baseline and intervened are identical or so close (single character/space difference) that they should be treated as identical.
- **MR-1** — Trivial wording swap. Same topic, same register, same specific content; only single-word substitutions or punctuation differences. Divergence token typically ≥ 15.
- **MR-2** — Content substitution within preserved frame. Baseline's task framing, register, and communicative mode are preserved, but specific content details are substituted (different setting, different examples, different named characters, different specific facts, different described sensory details).
- **MR-3** — frame-shift. The baseline's task framing is reinterpreted into a different task framing (narrative → commentary; empathetic listener → advisory list; analytical → narrative; prose → bulleted list; first-person → third-person meta). Response is still coherent and on-topic, but the *kind of work the model is doing* has shifted.
- **MSC** — Mode/stance change. Larger version of MR-3: the communicative register is definitively different (e.g., baseline is a narrative continuation, intervened is a writing-coach offering options; or baseline is empathetic listener, intervened is clinical advisory). The cell would be unambiguously classified as a mode shift by any reader.
- **TD** — Topic drift. Both texts are coherent and in the same register, but the specific topic is substantially different. Rare across all conditions (< 5%).
- **BSS** — Belief-state shift. The epistemic or evaluative stance toward the same content has flipped (e.g., baseline accepts a premise, intervened denies it; baseline validates a feeling, intervened reframes it as incorrect). Rare (< 1%).
- **FM** — Failure mode. Either side is degenerate (repetition loop, token-soup, hyphen-fragment, unique-word ratio < 0.25), in a wrong language, or otherwise not a valid natural-language continuation.

## 1. Input format

Each cell record you receive has:

```json
{
  "model": "llama_8b",
  "prompt_key": "group_01_v0",
  "prompt": "I woke up this morning feeling sick.",
  "condition": "D-scramble",
  "existing_category": "Minor_reframe",
  "baseline_text": "Sorry to hear that you're not feeling well...",
  "intervened_text": "When you're not feeling well, it can be a real challenge...",
  "div_tok": 0,
  "identical": false,
  "overlap": 0.31
}
```

- **`existing_category`** is authoritative from the 4-category reclassification. If it is `"Minor_reframe"`, your job is to sub-type into MR-1 / MR-2 / MR-3. If it is `null` or `"UNKNOWN"`, you must assign the full category (Ident / FM / TD / MSC / MR-1 / MR-2 / MR-3 / BSS). Records with `existing_category` = FM / TD / MSC / BSS / Ident are NOT included in the batches — those are already classified and do not need review.
- **`div_tok`** is the first generated token where intervened differs from baseline. Useful signal: div_tok ≥ 15 with high `overlap` (> 0.55) suggests MR-1.
- **`overlap`** is Jaccard word overlap between baseline and intervened.

## 2. Decision tree (priority order — apply in sequence)

Apply these rules top-to-bottom. The first rule that applies determines the category.

### Step 1: Is it Ident?
- `identical == True` → **Ident**
- Texts differ only in whitespace or punctuation → **Ident**
- Otherwise, proceed.

### Step 2: Is it FM?
- Either side is degenerate: short-token repetition loop (10+ copies of a bigram), hyphen fragment, token soup, unique-word ratio < 0.25 in the text → **FM**
- One side is in a wrong language (CJK, Korean, Cyrillic, Devanagari) while the other is in English → **FM**
- Output collapses to <5 words with no coherent content → **FM**
- If one side is slightly degraded but the other is coherent and they're still roughly on-topic, prefer the MR or MSC classification and flag as borderline.

### Step 3: Is there a topic change?
- The specific subject matter of the response is substantially different from the baseline (e.g., baseline discusses Roman road engineering; intervened discusses modern traffic jokes — same domain but different topic focus) → could be **TD**
- The register is the same but the topic genuinely shifted → **TD**
- If the register shifted AND the topic shifted, the MSC classification usually dominates. Genuine TD requires register preservation.
- TD is rare (target rate < 5% across conditions). If you're tempted to call something TD, ask: is the register really the same? Often it's MR-3 or MSC.

### Step 4: Is there a belief-state flip?
- Baseline asserts X; intervened asserts not-X on the same factual/evaluative proposition → **BSS**
- Examples: "Submarines can be submerged for 3 weeks" → "It's not possible for a submarine to be underwater for 3 weeks." Baseline celebrates a decision → intervened criticizes it.
- BSS is rare (target rate < 1%). Do not over-apply.

### Step 5: Is there a mode/stance change?
- **MSC vs MR-3 distinction** (the most important hard call):
  - **MSC**: The communicative register is unambiguously different. No reader would describe both as "the same kind of response." Example: baseline is a **narrative story continuation**; intervened is a **writing-coach offering numbered options**. These are different types of output. → **MSC**
  - **MR-3**: The task framing has shifted, but the response type is still broadly the same. Example: baseline is **empathetic listener asking a question**; intervened is **empathetic listener asserting the user's emotional state** (declarative vs. invitational — same mode, different framing). → **MR-3**
  - When in doubt: MSC requires a **mode jump**; MR-3 is a **task framing reinterpretation within the same broad mode**.

### Step 6: Sub-type the Minor-reframe cells (MR-1 / MR-2 / MR-3)

If you've ruled out Ident / FM / TD / BSS / MSC, the cell is MR. Now classify:

- **MR-1 — Trivial wording swap.** Same content, same specific details, same frame. Divergence at token ≥ 15. Differences are single-word substitutions, minor rephrasing, punctuation. If a reader read both without knowing which was which, they would describe the difference as "trivial."
  - Quick test: If you remove the intervened's single-word changes, would it match baseline verbatim? If yes → MR-1.
  - Examples: baseline ends with "best toys"; intervened ends with "hand-me-downs." baseline says "pasta"; intervened says "pasta sauce."

- **MR-2 — Content substitution within preserved frame.** Baseline and intervened are doing the same kind of work in the same register, but the specific content is substituted. Divergence typically at tokens 0–10. The baseline's narrative frame, analytical frame, empathetic frame, or advisory frame is preserved, but what's IN that frame is different.
  - Quick test: If you classified "what kind of response is this?" (e.g., "third-person narrative continuing a grief scene"), would both baseline and intervened get the same label? If yes → MR-2.
  - Examples: baseline describes woman in "pasta aisle"; intervened describes woman in "cereal aisle" with different sensory details. Baseline invents teacher named "Mrs. Smith"; intervened invents teacher named "Mrs. Johnson" with a different student. Baseline describes castle "surrounded by rolling hills"; intervened describes castle "surrounded by lush gardens."

- **MR-3 — frame-shift at the MR/MSC boundary.** The baseline's task framing is reinterpreted. The model is now doing a subtly different kind of work than in baseline. Divergence typically at tokens 0–2. Still coherent, still on-topic, but the "mode of engagement" has shifted.
  - Quick test: Would a reader say "these outputs are answering different questions" or "these outputs are playing different roles"? If yes → MR-3.
  - The six dominant MR-3 forms (described below) are the templates.

## 3. The six dominant MR-3 forms

These are the concrete types of frame-shift that D-scramble commonly produces. Learn to recognize all six.

### 3.1 LLM self-disclaimer

Baseline engages with the prompt substantively; intervened pivots to an AI-limitation disclaimer.

- Baseline (Everest metaphor prompt): "*What a powerful and evocative phrase! The idea of struggling to breathe at the summit...*"
- Intervened: "*As a digital AI assistant, I'm not a doctor, but I can try to help you figure out what might be going on...*"
- Marker phrases: "As a large language model," "As a conversational AI," "I don't have personal experiences," "I'm a digital AI assistant," "As a text-based AI model, I don't have the ability to...", "I can help you with...instead."

### 3.2 Writing-coach intrusion

Baseline continues a narrative or engages empathetically with the content; intervened pivots to writing-coach mode.

- Baseline (grief prompt): "*She had been going about her day, list in hand, checking off items as she made her way through the aisles of the grocery store...*" (narrative)
- Intervened: "*That's a great start! You've set the scene perfectly for a poignant story. Here are some ideas to build on it: Focus on the Sensory Details...*" (writing-coach)
- Marker phrases: "That's a great start!", "Here's a possible continuation:", "Here are some ideas:", "Let's continue the story," "Would you like me to expand on it?", "Here are a few ways you could continue."

### 3.3 Grammar or linguistic analysis

Baseline engages with the prompt's content; intervened analyzes the sentence itself.

- Baseline (elderly woman prompt): "*The elderly woman sat in her rocking chair, her eyes gazing out the window...*" (narrative)
- Intervened: "*I think there may be a small issue with the sentence. The phrase 'as it had been' is a bit unclear. Here are a few possible revisions...*" (grammar critique)
- Marker phrases: "The sentence 'X' is grammatically correct," "This sentence is a bit ambiguous," "Could you clarify," "I need a bit more context," "Here are a few possible revisions."

### 3.4 Multiple-choice or revision format

Baseline produces prose; intervened produces structured choice format.

- Baseline (Tokyo intersection): "*Tokyo's Shibuya Crossing, in particular, is known for being one of the busiest intersections...*" (exposition)
- Intervened: "***Answer: B.** **Explanation:** Option B is the correct answer because it correctly uses the comparative structure ('more...than') to compare two items...*" (multiple-choice)
- Marker phrases: "**Answer:**", "Option 1:", "**Option A:**", numbered list with multiple alternatives where baseline gave a single response.

### 3.5 Third-person analytical reframing

Baseline engages in first- or second-person; intervened shifts to third-person analytical.

- Baseline (feeling sick prompt): "*I'm sorry to hear that you're not feeling well. Can you tell me more about how you're feeling?*" (second-person empathetic)
- Intervened: "*It sounds like the patient is experiencing a viral infection. The patient might be experiencing...*" (third-person clinical)
- Marker phrases: "This person might," "It sounds like the individual," "It seems that the person," "The patient is experiencing."

### 3.6 Recipe / how-to / structured-output generation

Baseline engages with the prompt's framing; intervened produces structured content not requested by baseline's framing.

- Baseline (preheat oven prompt): "*It looks like you're getting ready to bake something! What's the recipe you're working on?*" (conversational)
- Intervened: "*Here's a revised version of the recipe: Ingredients: - 1 cup all-purpose flour - 1/2 cup sugar - 1/2 cup butter, softened...*" (recipe generation)
- Marker phrases: "Here's a revised version," "Ingredients:", "Step 1:", "Here's how you can..."

**Other MR-3 forms that don't fit cleanly into these six templates:** literary reference ("This is a nod to 'Paul Clifford' by Edward Bulwer-Lytton"), premise denial ("It's not possible to..."), joke interpretation, meta-commentary on the prompt itself. If you see a clear frame-shift but it doesn't match the six forms, still classify as **MR-3** and note the form type in your note field.

## 4. The boundary between MR-2 and MR-3 — the critical hard call

This is where most disagreement happens. Use these tests:

**Test A: The "type of response" test.**
- If you showed both outputs to a blind reader and asked "what kind of response is this?" for each, would they give the same label?
  - Same label (e.g., "both are empathetic conversational responses") → MR-2.
  - Different labels (e.g., "one is a narrative continuation, the other is writing-coach advice") → MR-3.

**Test B: The task framing test.**
- What question is the baseline answering? What question is the intervened answering?
- If the same question → MR-2.
- If different questions (e.g., baseline: "what is the model's continuation of this story?"; intervened: "what grammatical issues does this sentence have?") → MR-3.

**Test C: The MR-3 template test.**
- Does the intervened match any of the six MR-3 forms (LLM self-disclaimer, writing-coach, grammar analysis, MC format, third-person analytical, structured-output)?
- Yes → MR-3.
- No → MR-2 (assuming baseline and intervened are in the same broad mode).

### Specific hard cases

**Case: Both baseline and intervened are in writing-coach / meta mode.** This happens frequently on Gemma 9B (see §6). If baseline says "Here are some ways you could expand this sentence:" and intervened says "Here are some possibilities for continuing:" — both are in writing-coach mode with different suggestions. **This is MR-2, not MR-3**, because the task framing has not shifted — it was already writing-coach on both sides.

**Case: Both are in empathetic-listener mode with different emotional framings.** Baseline asks "Would you like to talk about what happened?"; intervened asks "Are you experiencing intense emotions right now?" — both are empathetic listener mode; the question asked differs but the mode is preserved. **MR-2.**

**Case: Baseline is narrative; intervened is narrative but with a named-character swap (e.g., gender or name change).** Baseline has male rookie cop; intervened has female "Emily" rookie cop. Both are narrative continuations of the same prompt. The character identity shifted but the task framing did not. **MR-2** (content substitution of character identity).

**Case: Baseline is analytical prose; intervened is analytical bulleted list.** Format shift (prose → bullets) with same content and register. This is a borderline case. If the content is substantially the same and the only difference is formatting, treat as MR-1 if div ≥ 15 or MR-2 otherwise. If the bullet points introduce content not in baseline's prose, lean MR-3 (structured-output generation).

## 5. F-dynamic failure sub-types (for reference only — F-dynamic cells are classified as FM and are not in your batches unless GPT-OSS 120B)

F-dynamic produces six degeneration subtypes. You don't need to classify F-dynamic cells (they're pre-categorized as FM in the reclassification). But if you encounter a F-dynamic cell for GPT-OSS 120B (the 80 cells needing full classification), use:

- **loop_severe**: Short-token repetition, unique-token ratio < 0.15 ("and the and the and the...")
- **fragment_short**: <15 words, partial halt ("hope you feel better soon!, do you have.")
- **fragment_tiny**: <5 words ("found. found.")
- **loop_moderate**: Partial coherent start, then collapse
- **loop_mild**: Mostly coherent with repetitive tail
- **long_other**: Longer half-coherent output with embedded correct content

All six subtypes are classified as **FM** at the top level; subtype can be noted in the `note` field.

## 6. Model-specific guidance

**Each model has slightly different default behaviors. These are real and worth calibrating to.**

### LLaMA 8B
- Baseline is typically conversational or narrative; clearly distinguishable from D-scramble's frame-shifts.
- Known MR-3 signatures: empathy → advisory numbered list, narrative → meta-commentary.
- Default classification pace: easiest of all 10 models.

### LLaMA 70B
- Baseline is typically richer narrative than LLaMA 8B, often with descriptive detail.
- Known MR-3 signatures: strong LLM self-disclaimer pattern ("As a digital AI assistant"), writing-coach intrusion.
- Watch for: "*I'm happy to continue the story. Here's a possible next sentence:*" — strong MR-3 signal.

### Gemma 9B
- **IMPORTANT: Baseline is frequently already in writing-coach / meta mode** (e.g., "This is a great start! Here are some ways you could expand..."). When both baseline and intervened are in writing-coach mode, classify as **MR-2**, not MR-3, because the task framing was already writing-coach.
- Gemma 9B MR-3 cells are rarer because the baseline is pre-meta. Look for cases where baseline is truly narrative or truly empathetic and intervened shifts.
- Do not over-classify MR-3 on Gemma 9B.

### Gemma 27B
- Very few MR cells exist (93.8% failure mode rate on D-scramble). The 5 or so MR cells tend to be borderline between MR-3 and FM.
- If the text is partially degenerate but has some coherent content, classify as FM only if the unique-word ratio is genuinely < 0.25 or there's clear looping.

### GPT-OSS 20B
- Baseline is often more terse than other models.
- Known MR-3 signatures: grammar/linguistic analysis of the prompt, recipe-generation when prompt mentions anything food-related, multiple-choice answer format.
- "The sentence 'X' is grammatically correct" is a strong MR-3 marker.

### GPT-OSS 120B
- **SPECIAL CASE for F-topK: 80 cells need FULL categorization (not just MR sub-typing)** because GPT-OSS 120B F-topK data was added AFTER the main reclassification. These cells have `existing_category: null`. You must assign Ident / FM / TD / BSS / MSC / MR-1 / MR-2 / MR-3.
- Baseline often exhibits its own text-generation pathologies (cut-off messages, repetition). Many cells have both baseline and intervened partially degenerate; classify as FM if both sides are broken.
- Known MR-3 signature: multiple-choice answer format ("**Answer: B.** **Explanation:**"), grammar-revision format with asterisks.

### Mistral 7B
- Baseline is typically advisory or empathetic.
- Known MR-3 signatures: LLM self-disclaimer ("As a conversational AI," "I'm not a doctor"), writing-coach pivot ("Let's continue the story"), third-person reframing.
- Strong D MR-3 rate expected (~35–40%).

### Mixtral 8×7B
- Baseline is fluid — sometimes narrative, sometimes analytical, sometimes advisory. Check baseline carefully before classifying intervened.
- Known MR-3 signatures: narrative ↔ analytical mode flips (either direction), prose → bulleted structure.
- Strong D MR-3 rate expected (~50–60%). Be thorough — the signature is clear when you look for mode flips.

### Qwen 14B and Qwen 72B
- Baseline often in writing-coach or conversational mode.
- Known MR-3 signatures: meta-writing-coach ("Would you like me to expand on it?"), "the scene you've painted" literary framing, format shifts (prose → bulleted), story-prompt interpretation ("this sentence could be the beginning of many stories").
- Strong D MR-3 rate expected (~35–65%).

## 7. Borderline flagging and confidence

### Confidence levels

- **high**: Clear-cut classification. You're confident this category is correct and no reasonable alternative applies.
- **medium**: Defensible classification but another category could be argued. Note which alternative in the `secondary_category` field.
- **low**: Genuinely hard to classify. The cell sits between two categories or has unusual features. **Flag `borderline: true`** for these.

### Borderline criteria — FLAG FOR FRONTIER REVIEW

Mark `borderline: true` if any of the following apply:

1. **MR-2 vs MR-3 boundary.** The frame-shift is subtle — the baseline is partly meta, or the intervened is mostly the same mode but with a small task framing change. Flag and note "MR-2/MR-3 boundary" in the note.

2. **MR-3 vs MSC boundary.** The mode shift is substantial but you're not sure if it crosses from "reinterpretation within mode" (MR-3) to "different mode entirely" (MSC). Flag and note "MR-3/MSC boundary."

3. **FM vs partial-coherence.** One side is partially degenerate but the other is coherent, and there's enough coherent content to try to compare. Flag and note "FM/partial-coherence boundary."

4. **Baseline itself is degraded.** If the baseline has repetition, truncation, or other pathologies, classification becomes hard. Flag and note "baseline-degraded."

5. **Both outputs are in the same uncommon mode.** If both are in writing-coach mode, or both in grammar-analysis mode, and you suspect the task framing was already shifted BEFORE the intervention, flag as MR-2 with note "both-meta-already."

6. **Topic-proximate shift.** If the topic is on the borderline between "same" (MR-2) and "substantially different" (TD). Flag and note "topic-drift borderline."

7. **Any classification where you're genuinely unsure.** Don't try to resolve hard cases via guessing — flag for frontier review.

### Target borderline rate

Expect ~10–20% of cells to be borderline. If you're flagging < 5% of cells, you may be over-confident; if > 30%, you may be under-calibrated. Aim for explicit engagement with the boundary cases.

## 8. Output JSON schema

Each classified cell should produce a record:

```json
{
  "model": "llama_8b",
  "prompt_key": "group_01_v0",
  "condition": "D-scramble",
  "subtype": "MR-3",
  "confidence": "high",
  "borderline": false,
  "secondary_category": null,
  "mr3_form": "advisory_list",
  "note": "Baseline is empathetic listener ('Sorry to hear...Can you tell me'); D pivots to numbered advisory list ('Here are some suggestions: 1. Stay Hydrated')."
}
```

### Field definitions

- **`subtype`**: One of `Ident`, `MR-1`, `MR-2`, `MR-3`, `MSC`, `TD`, `BSS`, `FM`. For cells with `existing_category: "Minor_reframe"`, only MR-1/MR-2/MR-3 are valid (the existing classification rules out other categories). For cells with `existing_category: null` (GPT-OSS 120B F-topK), any of the 8 categories is valid.

- **`confidence`**: `high`, `medium`, or `low`.

- **`borderline`**: `true` if the cell meets any of the borderline criteria in §7. `false` otherwise. Borderline cells are set aside for frontier-model review.

- **`secondary_category`**: If `confidence` is `medium` or `borderline` is `true`, indicate which other category is the most plausible alternative. Null otherwise.

- **`mr3_form`**: If `subtype == "MR-3"`, indicate which of the six forms from §3 (use one of: `llm_disclaimer`, `writing_coach`, `grammar_analysis`, `mc_format`, `third_person_analytical`, `structured_output`, or `other` with explanation in note). Null for other subtypes.

- **`note`**: One-line description of the distinguishing feature. Cite specific phrases from both baseline and intervened that support the classification. Keep under 200 characters.

## 9. Worked examples (three per category, diverse across models)

### Ident examples

**Example Ident-1** — `group_02_v6` D-scramble on llama_8b: identical=True. Both sides verbatim: "This sentence is a great example of a common phenomenon in human behavior, where someone chooses not to speak up or take action...". **Classify: Ident, high, no note needed.**

### FM examples

**Example FM-1** — `group_01_v1` F-dynamic on llama_8b: "*what did the and of the and and and and and and and and and and and and and and and and and...*" — short-token repetition loop. **Classify: FM, high, note: "token loop 'and and...'".**

**Example FM-2** — `group_02_v0` F-dynamic on llama_8b: "*found. found.*" — fragment-tiny. **Classify: FM, high, note: "2-word fragment halt".**

### MR-1 examples

**Example MR-1-a** — `group_04_v6` Random on llama_8b: Baseline ends with "best toys"; intervened ends with "hand-me-downs" at token 34. Single-word swap, everything else identical. **Classify: MR-1, high, no borderline.**

**Example MR-1-b** — `group_06_v5` Random on llama_8b: Baseline: "*Growing up before television was a vastly different time*"; intervened: "*Life before television was a vastly different time*" at token 15. Trivial phrase swap. **Classify: MR-1, high.**

### MR-2 examples

**Example MR-2-a** — `group_01_v4` S-scramble on llama_8b: Baseline narrative in "pasta aisle"; S-scramble narrative in "cereal aisle, fluorescent lights flickering." Same third-person narrative register, substituted location and sensory details. **Classify: MR-2, high, note: "narrative preserved; pasta aisle→cereal aisle substitution".**

**Example MR-2-b** — `group_08_v0` F-topK on llama_8b: Baseline offers brief dialogue suggestion; intervened writes full narrative with invented "Mrs. Johnson" teacher. Both in narrative mode but with different content (one short dialogue, one full story). **Classify: MR-2, medium, note: "narrative mode preserved; different narrative content with invented character name".** *(Medium because the length and structural shift is substantial — could be borderline MR-3 if you consider "dialogue-suggestion" and "full-narrative" as different task framings. Mark borderline.)*

**Example MR-2-c** — `group_05_v3` D-scramble on mixtral_8x7b: Both baseline and intervened are analytical prose about "a single catastrophic decision by the board." D-scramble rephrases slightly. Same analytical register, same topic. Divergence at token 11, overlap ~0.6. **Classify: MR-2, high, note: "analytical register preserved, minor rephrasing within same content".**

### MR-3 examples

**Example MR-3-a** — `group_01_v0` D-scramble on llama_8b: Baseline: "*Sorry to hear that you're not feeling well. Can you tell me more about how you're feeling?*" (empathetic listener). D-scramble: "*When you're not feeling well, it can be a real challenge to get through the day. Here are some suggestions that might help: 1. Stay Hydrated...*" (advisory list). frame-shifted from empathetic listener to advisory list. **Classify: MR-3, high, mr3_form: "structured_output" (advisory list), note: "empathetic listener → numbered advisory list".**

**Example MR-3-b** — `group_04_v4` D-scramble on llama_8b: Baseline: "*The elderly woman sat in her rocking chair, her eyes gazing out the window...*" (narrative). D-scramble: "*I think there may be a small issue with the sentence. The phrase 'as it had been' is a bit unclear. Here are a few possible revisions: * The elderly woman remembered the neighborhood as it was sixty years ago...*" (grammar critique). **Classify: MR-3, high, mr3_form: "grammar_analysis", note: "narrative continuation → grammar critique with suggested revisions".**

**Example MR-3-c** — `group_01_v3` D-scramble on mistral_7b: Baseline: "*That's wonderful to hear! Feeling content is a beautiful emotion. It seems like he's experiencing a moment of peace...*" (conversational). D-scramble: "*I'm not a person, so I don't have personal experiences or feelings. However, I can tell you that it's common for people to feel content when...*" (LLM self-disclaimer then analysis). **Classify: MR-3, high, mr3_form: "llm_disclaimer", note: "conversational engagement → LLM self-disclaimer followed by third-person analysis".**

### MSC examples (for GPT-OSS 120B F-topK or borderline cases only)

**Example MSC-a** — Hypothetical: Baseline is "It sounds like you're going through a tough time — would you like to talk about it?" (empathetic listener). Intervened is "Here is a five-paragraph essay on the psychological effects of grief in retail environments. **Introduction.** Grief in public spaces presents unique challenges..." (academic essay). This is a clear mode jump from empathetic listener to academic essay. **Classify: MSC.**

### TD examples

**Example TD-a** — Uncommon. Most cells suspected of TD are actually MR-3 or MSC. A genuine TD would be: baseline discusses Roman road engineering; intervened discusses modern highway maintenance in prose-paragraph analytical form. Same register, same abstract domain (roads), but the specific topic (ancient Roman engineering vs. modern highway maintenance) is different. **If you're tempted to classify TD, strongly consider whether it's actually MR-3 first. Flag all TD classifications as borderline.**

### BSS examples

**Example BSS-a** — `group_10_v0` D-scramble on llama_8b (known BSS case): Baseline accepts the premise "The submarine had been underwater for three weeks" and describes it as a long time. D-scramble flips: "*It's not possible for a submarine to be underwater for three weeks without any issues. Submarines... need to resurface regularly.*" Epistemic flip. **Classify: BSS, high, note: "baseline accepts premise; D denies it on factual grounds".**

## 10. Classification pace and self-monitoring

**Target pace**: 20–40 seconds per cell. If a cell takes more than 90 seconds, flag as borderline and move on.

**Every 50 cells**, briefly review your recent classifications and check:
- Are you consistently applying the MR-2 vs MR-3 boundary?
- Are you over-applying MR-3 (> 50% of D-scramble cells labeled MR-3 suggests over-classification on models where expected rate is 25–40%)?
- Are you under-applying borderline flags (< 5% of cells flagged suggests over-confidence)?

**If you notice yourself drifting toward a particular classification**, pause and re-read the §4 hard-case criteria.

## 11. What to output

For each batch of cells, produce a JSON array of classification records (one per cell). Save to:

```
outputs/agent_classifications/batch_N_results.json
```

Where `N` is the batch number (01 through 08).

**Also produce a per-batch summary**:

```
outputs/agent_classifications/batch_N_summary.md
```

with:
- Per-condition MR-1/MR-2/MR-3/other counts across all models in the batch
- Cross-model patterns observed (e.g., "On prompt group_01_v4 (grief grocery), D-scramble produced MR-3 in 7/10 models")
- Borderline cell count and a list of their prompt_keys
- Any classification issues or ambiguities encountered

## Appendix A: Prompt category reference

The 80 SpecB prompts are grouped into 10 categories, 8 prompts per group:

- `group_01`: Emotional State
- `group_02`: Decision/Choice
- `group_03`: Goal-Driven Action
- `group_04`: Perspective/Voice
- `group_05`: Causal Setup
- `group_06`: Temporal Anchor
- `group_07`: Genre/Register
- `group_08`: Social Role
- `group_09`: Evaluative Stance
- `group_10`: Physical Setting

Prompt keys take the form `group_XX_vY` where Y is 0–7.

## Appendix B: Expected rate ranges (for self-calibration)

Based on preliminary hand-review of subsets:

- **D-scramble MR-3 rate**: 15–60% across models (median ~29%, weighted mean ~33%)
- **S-scramble MR-3 rate**: 1–5% across models (likely median ~3%)
- **F-topK MR-3 rate**: 1–12% across models (median ~3–5%, Mixtral 8×7B outlier ~11%)
- **F-dynamic MR-3 rate** (for the small fraction of F-dynamic cells that are MR): near 0%
- **Random MR-3 rate**: 0–6% across models

If your classifications produce rates outside these ranges by > 2×, pause and re-calibrate.

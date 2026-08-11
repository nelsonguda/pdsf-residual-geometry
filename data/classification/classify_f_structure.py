#!/usr/bin/env python3
"""
classify_f_structure.py — regenerate f_structure_classification.json.

WHAT THIS IS
    The outcome labels behind Table B.6-1 (F-mix vs F-attenuate at ~12% depth)
    are produced by this deterministic rule-based classifier, not by a person and
    not by a language model. That distinguishes it sharply from
    all_classifications_v4.json and all_classifications_v5_fdynamic_audit.json,
    which are supervised judgments over generated text. Here the label is a
    function of the text pair, so it is fully reproducible — and equally, it is
    only as good as the rules below. Read them before trusting a cell.

THE RULES, IN ORDER
    1. Either side degenerate (word/trigram/sentence/line repetition, too short,
       truncation stutter)                                     -> Failure_mode
       Length and repetition are measured over tokenize(), which falls back to
       characters for scripts written without inter-word spaces. See the comment
       above CHARS_PER_TOKEN_MAX; before 2026-08-11 this step used a bare
       text.split() and misread fluent Chinese and Japanese as fragments.
    2. Writing system changes between baseline and intervention -> Mode_stance_change
    3. Topic-word overlap >= 0.10, with a meta-commentary register shift
       on exactly one side                                     -> Mode_stance_change
    4. Topic-word overlap >= 0.10                              -> Minor_reframe
    5. Topic-word overlap >= 0.03                              -> Minor_reframe
    6. Both sides CJK/Japanese, zero overlap                   -> Minor_reframe
       (or Mode_stance_change on a one-sided register shift)
    7. Otherwise fall back to shared content keywords:
         >= 2 shared        -> Minor_reframe
         == 1 shared        -> Minor_reframe for specialized registers
                               (proofs, code, legal, patent, academic, recipe),
                               else Topic_drift
         == 0 shared        -> Mode_stance_change for Romance/archaic registers,
                               else Topic_drift
    Byte-identical continuations bypass all of this and are labelled Identical.

    Rules 6, 7's specialized-category branch and 7's Romance branch exist because
    zero lexical overlap does not mean topic change when the two sides are in a
    non-Latin script, in a formulaic register, or in a language whose function
    words dominate the token counts. They are the classifier's judgment calls and
    the places a reader should push back first.

USAGE
    python3 classify_f_structure.py                  # verify against the shipped file
    python3 classify_f_structure.py --write out.json # regenerate

    Run it from data/classification/. It reads only
    ../behavioral_interventions/{model}-Continuation-Diverse-F_early_{mix,attenuate}.json
    and needs nothing but the Python standard library.

VERIFIED
    2026-08-11 — reproduces all ten cells of Table B.6-1 exactly:
    F-mix 51 / 622 / 127 / 19 / 21, F-attenuate 326 / 425 / 65 / 11 / 13.

REVISION 2026-08-11 — CJK tokenization
    The degeneracy test previously tokenized on whitespace. Chinese and Japanese
    are written without inter-word spaces, so a fluent paragraph scored as a
    one- or two-token fragment and was labelled Failure_mode. 98 of the 1,680
    trials were affected — 97 in Mandarin, Japanese or Korean regimes, and one
    English-prompt trial whose intervention switched into Chinese. All 98 moved
    out of Failure_mode (86 to Minor_reframe, 12 to Mode_stance_change); no cell
    moved into it, and Identical and Topic_drift are unchanged at both depths.
    The superseded distribution was F-mix 51 / 570 / 120 / 19 / 80 and
    F-attenuate 326 / 391 / 60 / 11 / 52.

    Note it does NOT reproduce the ~70%-depth distribution. Those trials had the
    two worst-performing heuristic categories re-read and reassigned by hand; that
    adjudication is in f_structure_70pct_review.json and is not script-derivable.
    The same tokenization fix was applied there as a delta over the reviewed
    labels, so Table B.6-2 moves too.
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

MODELS = ['mistral_7b', 'llama_8b', 'gemma_9b', 'qwen_14b', 'gpt_oss_20b',
          'gemma_27b', 'mixtral_8x7b', 'llama_70b', 'qwen_72b', 'gpt_oss_120b']
CONDITIONS = {'F_mix': 'F_early_mix', 'F_attenuate': 'F_early_attenuate'}

# Table B.6-1 as published in the revision of 2026-08-11. The superseded
# pre-CJK-fix values are recorded in the REVISION note in the module docstring.
EXPECTED = {
    'F_mix':       {'Identical': 51, 'Minor_reframe': 622, 'Mode_stance_change': 127,
                    'Topic_drift': 19, 'Failure_mode': 21},
    'F_attenuate': {'Identical': 326, 'Minor_reframe': 425, 'Mode_stance_change': 65,
                    'Topic_drift': 11, 'Failure_mode': 13},
}


def detect_script(text):
    scripts = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch, '')
            except:
                name = ''
            if 'CJK' in name or 'HANGUL' in name:
                scripts['CJK'] += 1
            elif 'HIRAGANA' in name or 'KATAKANA' in name:
                scripts['Japanese'] += 1
            elif 'ARABIC' in name:
                scripts['Arabic'] += 1
            elif 'CYRILLIC' in name:
                scripts['Cyrillic'] += 1
            else:
                scripts['Latin'] += 1
    return scripts.most_common(1)[0][0] if scripts else 'Unknown'

# Chinese and Japanese are written without inter-word spaces, so text.split()
# returns one or two "words" for a fluent paragraph and every length-based
# heuristic below misreads it as a fragment. Measured across the 3,360 texts in
# this corpus, characters-per-whitespace-token sits at or below 7.5 (95th pct)
# for every space-separated regime -- including Korean, which does use spaces --
# and at 30+ for Mandarin and Japanese. A threshold of 12 separates them with a
# wide margin on both sides.
CHARS_PER_TOKEN_MAX = 12


def tokenize(text):
    """Whitespace tokens, falling back to characters for space-free scripts.

    Returning characters for Chinese and Japanese gives the length, repetition
    and unique-ratio tests below a unit comparable to a word in a spaced script.
    """
    words = text.split()
    stripped = ''.join(words)
    if words and len(stripped) / len(words) > CHARS_PER_TOKEN_MAX:
        return list(stripped)
    return words


def is_degenerate(text):
    if not text or len(text.strip()) < 10:
        return True, 'too_short'
    words = tokenize(text)
    if len(words) < 3:
        return True, 'too_few_words'
    wc = Counter(w.lower() for w in words)
    top_word, top_count = wc.most_common(1)[0]
    if top_count / len(words) > 0.35:
        return True, f'word_repetition:{top_word}({top_count}/{len(words)})'
    if len(words) >= 6:
        trigrams = [' '.join(words[j:j+3]).lower() for j in range(len(words)-2)]
        tc = Counter(trigrams)
        top_tg, tg_count = tc.most_common(1)[0]
        if tg_count >= max(3, len(trigrams) * 0.25):
            return True, f'trigram_repetition({tg_count}x)'
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.2 and len(words) > 10:
        return True, f'low_unique_ratio:{unique_ratio:.2f}'
    sentences = re.split(r'[.!?。！？]\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 4:
        sent_set = set(s[:50].lower() for s in sentences)
        if len(sent_set) <= max(1, len(sentences) * 0.3):
            return True, f'sentence_repetition:{len(sent_set)}/{len(sentences)}'
    if re.search(r'(It looks like|your message got cut off|It seems like)', text) and len(words) < 30:
        return True, 'truncated_stuttering'
    frags = re.findall(r'"([^"]{5,50})"', text)
    if frags:
        frag_counts = Counter(frags)
        top_frag, fc = frag_counts.most_common(1)[0]
        if fc >= 3:
            return True, f'repeated_fragment:{top_frag[:30]}({fc}x)'
    lines = text.strip().split('\n')
    if len(lines) >= 4:
        line_set = set(l.strip()[:40] for l in lines if len(l.strip()) > 5)
        if len(line_set) <= max(1, len(lines) * 0.25):
            return True, f'line_repetition:{len(line_set)}/{len(lines)}'
    return False, ''

def extract_topic_words_multilingual(text):
    script = detect_script(text)
    if script in ('CJK', 'Japanese'):
        chars = [ch for ch in text if unicodedata.category(ch).startswith('L')]
        bigrams = [''.join(chars[i:i+2]) for i in range(len(chars)-1)]
        return bigrams[:30], script
    else:
        stops = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'will', 'more',
                 'when', 'who', 'way', 'may', 'than', 'them', 'some', 'into', 'from',
                 'this', 'that', 'with', 'they', 'each', 'which', 'their', 'what',
                 'about', 'would', 'there', 'could', 'other', 'were', 'also', 'just',
                 'like', 'its', 'very', 'here', 'your', 'how', 'both', 'she', 'his',
                 'does', 'most', 'then', 'being', 'well', 'make', 'made', 'est', 'les',
                 'des', 'une', 'que', 'pas', 'par', 'sur', 'dans', 'pour', 'avec', 'son',
                 'del', 'los', 'las', 'una', 'por', 'con', 'como', 'pero', 'mas'}
        words = re.findall(r'[a-zA-Z\u00C0-\u024F]{3,}', text.lower())
        return [w for w in words if w not in stops][:30], script

meta_patterns = [
    r"(That's a great|That's a nice|That's a lovely|That's a good|Great start|Here's a|Here are)",
    r"(This is a great|This sentence|This phrase|This story|This passage|This is|This example)",
    r"(You could|You might|Consider|I'd suggest|Let me help|I can help)",
    r"(上記|以下|英訳|翻訳|日本語|この文|この話|素晴らしい|いい文)",
    r"(Voici|C'est une|Qué frase|Es una)",
]

def classify_pair(baseline, intervention, category):
    b_degen, b_reason = is_degenerate(baseline)
    i_degen, i_reason = is_degenerate(intervention)

    if b_degen or i_degen:
        return 'Failure_mode', f"{'baseline' if b_degen else 'intervention'} degenerate: {b_reason if b_degen else i_reason}"

    b_script = detect_script(baseline)
    i_script = detect_script(intervention)

    if b_script != i_script and b_script != 'Unknown' and i_script != 'Unknown':
        return 'Mode_stance_change', f'cross_language: {b_script} -> {i_script}'

    b_topics, _ = extract_topic_words_multilingual(baseline)
    i_topics, _ = extract_topic_words_multilingual(intervention)

    if b_topics and i_topics:
        b_set = set(b_topics[:20])
        i_set = set(i_topics[:20])
        overlap = len(b_set & i_set) / max(len(b_set | i_set), 1)
    else:
        overlap = 0

    b_meta = any(re.search(p, baseline[:200], re.I) for p in meta_patterns)
    i_meta = any(re.search(p, intervention[:200], re.I) for p in meta_patterns)

    if overlap >= 0.10:
        if (b_meta and not i_meta) or (i_meta and not b_meta):
            return 'Mode_stance_change', f'overlap={overlap:.2f}, register_shift'
        return 'Minor_reframe', f'overlap={overlap:.2f}'
    elif overlap >= 0.03:
        return 'Minor_reframe', f'overlap={overlap:.2f}, low_but_nonzero'
    elif b_script in ('CJK', 'Japanese') and i_script in ('CJK', 'Japanese'):
        if (b_meta and not i_meta) or (i_meta and not b_meta):
            return 'Mode_stance_change', f'CJK_zero_overlap, register_shift'
        return 'Minor_reframe', f'CJK_zero_overlap, same_category={category}'
    else:
        if (b_meta and not i_meta) or (i_meta and not b_meta):
            return 'Mode_stance_change', f'overlap={overlap:.2f}, register_shift'
        # Low overlap Latin text — use keyword check
        def key_words(text, n=15):
            kw_stops = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                     'was', 'one', 'has', 'have', 'been', 'will', 'more', 'who', 'his',
                     'her', 'its', 'this', 'that', 'with', 'they', 'each', 'which', 'what',
                     'from', 'into', 'about', 'would', 'there', 'could', 'other', 'also',
                     'just', 'like', 'very', 'here', 'your', 'how', 'both', 'she', 'were',
                     'some', 'them', 'than', 'then', 'made', 'make', 'well', 'being', 'does',
                     'most', 'way', 'may', 'our', 'out', 'even', 'only', 'back', 'after',
                     'over', 'such', 'take', 'own', 'through', 'too', 'any', 'same', 'much',
                     'few', 'still', 'felt', 'feel', 'feeling', 'thought', 'think', 'knew',
                     'know', 'something', 'started', 'began', 'going', 'come', 'came'}
            words = re.findall(r'[a-zA-Z]{4,}', text.lower())
            return set(w for w in words[:30] if w not in kw_stops)
        bk = key_words(baseline)
        ik = key_words(intervention)
        shared = bk & ik
        if len(shared) >= 2:
            return 'Minor_reframe', f'shared_keywords={sorted(shared)[:5]}'
        elif len(shared) == 1:
            if category in ['Mathematical Proof', 'Mathematical Derivation', 'Tutorial',
                            'JavaScript Code', 'Python Code', 'Legal Register',
                            'Patent Application', 'Academic Abstract', 'Cooking Show Transcript']:
                return 'Minor_reframe', f'specialized_category, shared={sorted(shared)}'
            return 'Topic_drift', f'minimal_overlap, shared={sorted(shared)}'
        else:
            if category in ['Spanish Narrative', 'French Narrative', 'Portuguese Narrative',
                            'Archaic Register']:
                return 'Mode_stance_change', f'non_english_zero_overlap'
            return 'Topic_drift', f'zero_shared_keywords'



def classify(src_dir='../behavioral_interventions'):
    rows = []
    for cond, fam in CONDITIONS.items():
        for m in MODELS:
            path = os.path.join(src_dir, f'{m}-Continuation-Diverse-{fam}.json')
            for r in json.load(open(path))['results']:
                e = r.get('effect') or {}
                pk = r.get('prompt_key') or f"{r['group_id']}_{r['variant_id']}"
                if e.get('identical'):
                    label, rule = 'Identical', 'byte-identical to baseline'
                else:
                    label, rule = classify_pair(r['baseline_text'],
                                                r['scrambled_text'],
                                                r.get('category', ''))
                rows.append({'model': m, 'condition': cond, 'prompt_key': pk,
                             'linguistic_regime': r.get('category', ''),
                             'label': label, 'rule': rule,
                             'identical': bool(e.get('identical')),
                             'first_divergence_token': e.get('first_divergence_token')})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('USAGE')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='../behavioral_interventions',
                    help='directory holding the continuation files')
    ap.add_argument('--write', metavar='FILE', help='write the regenerated records to FILE')
    args = ap.parse_args()

    rows = classify(args.src)
    print(f'{len(rows)} trials classified\n')

    ok = True
    for cond in CONDITIONS:
        got = Counter(r['label'] for r in rows if r['condition'] == cond)
        exp = EXPECTED[cond]
        print(f'{cond}  (n = {sum(got.values())})')
        for label in ('Identical', 'Minor_reframe', 'Mode_stance_change',
                      'Topic_drift', 'Failure_mode'):
            g, p = got.get(label, 0), exp[label]
            flag = '' if g == p else '   <-- differs from Table B.6-1'
            if g != p:
                ok = False
            print(f'   {label:20s} {g:4d}   published {p:4d}{flag}')
        print()

    if args.write:
        json.dump(rows, open(args.write, 'w'), indent=1, allow_nan=False)
        print(f'wrote {args.write}')

    print('Table B.6-1 reproduced exactly.' if ok else 'MISMATCH — see the rows marked above.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

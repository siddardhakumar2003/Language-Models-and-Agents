#!/usr/bin/env python3
"""
Generate a synthetic Telugu comparative-reasoning finetuning dataset (Phase 3, PDF Sec 3.1).

Per the project spec: this is NOT downloaded from an existing reasoning benchmark --
examples are generated programmatically from controlled templates, so ground-truth
labels are known exactly. Covers:
  - Pairwise comparisons with explicit numeric values (height / weight / age / price)
  - Yes/No comparison questions
  - Three-entity transitive chains (A>B, B>C => tallest/shortest, and A vs C by transitivity)
  - Three-entity numeric superlatives (cheapest/costliest/oldest/youngest/etc.)
  - Equal-value cases

Leakage avoidance: the entity name pool (persons and objects, separately) is split into
disjoint train/val/test subsets BEFORE any examples are generated, so no name seen in
val/test ever appears in train. One template variant per question type is also reserved
for val/test only, holding out a relation-pattern in addition to entity names.

Usage:
    python3 generate_reasoning_data.py --num-samples 10000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "data"

# ============================================================================
# Vocabulary (sourced from common Telugu names / places / objects -- not
# mined from the pretraining corpus, to keep this dataset fully independent
# of it; entity names are split into train/val/test pools below).
# ============================================================================

PERSON_NAMES = [
    "రాముడు", "సీత", "కృష్ణ", "లక్ష్మి", "వెంకటేశ్", "పార్వతి", "సూర్య", "గీత",
    "రాజు", "అనిత", "కిరణ్", "దీప", "మురళి", "శ్రీను", "రాధ", "గోపాల్",
    "సుధ", "రవి", "మీనా", "ప్రసాద్", "విజయ", "నరేష్", "కావ్య", "సతీష్",
    "పూర్ణిమ", "మహేష్", "స్వాతి", "రాజేష్", "భారతి", "వెంకట్", "రేఖ", "శంకర్",
    "లత", "గణేష్", "ఉమ", "రామకృష్ణ", "జ్యోతి", "నాగరాజు", "శైలజ", "సాయి",
    "అనూష", "హరి", "పద్మ", "వేణు", "సుమతి", "చందు", "విద్య", "శివ", "రమ్య", "అర్జున్",
    # Second batch: widens the entity pool so no single name dominates the answer
    # distribution and held-out test names have more train-side analogues to generalize from.
    "నాగేశ్వర్", "సుబ్బారావు", "వెంకటలక్ష్మి", "రామలక్ష్మి", "సురేష్", "రామారావు",
    "జానకి", "పద్మావతి", "శ్రీనివాస్", "వెంకటరమణ", "అనసూయ", "భాస్కర్", "రాఘవ",
    "సుజాత", "మల్లికార్జున్", "సరళ", "కృష్ణమూర్తి", "విజయలక్ష్మి", "రామ్మోహన్",
    "గోవింద్", "శశి", "రామకుమార్", "ప్రియ", "ఉదయ్", "చంద్రిక", "రాజశేఖర్", "కమల",
    "శ్రీదేవి", "రామనాథ్", "సుభాష్", "అనురాధ", "వెంకటేశ్వరరావు", "గాయత్రి",
    "రాజేంద్ర", "సుమన్", "మోహన్రావు", "రాజ్యలక్ష్మి", "సాయిరామ్", "పవన్", "దుర్గ",
    "రామస్వామి", "కళ్యాణి", "శేఖర్", "విమల", "రామరాజు", "సునంద", "అశోక్",
    "మంజుల", "రామచంద్ర", "సుధాకర్",
]

OBJECT_NAMES = [
    "పుస్తకం", "కారు", "సైకిల్", "మామిడిపండు", "ఇల్లు", "ఫోన్", "కుర్చీ",
    "బ్యాగ్", "వాచ్", "టేబుల్", "గడియారం", "ల్యాప్‌టాప్", "గొడుగు", "బూట్లు",
    "పెన్ను", "బల్ల", "అద్దం", "బొమ్మ", "పెట్టె", "తువ్వాలు",
    # Second batch (see PERSON_NAMES comment above).
    "కంప్యూటర్", "టీవీ", "రేడియో", "కెమెరా", "స్కూటర్", "బస్సు", "రైలు",
    "విమానం", "పడవ", "మంచం", "సోఫా", "అల్మారా", "తలుపు", "కిటికీ", "దీపం",
    "బకెట్", "గిన్నె", "చెంచా", "కత్తి", "గొడ్డలి",
]

# ============================================================================
# Attribute definitions: unit label + realistic numeric range.
# ============================================================================

ATTRIBUTES = {
    "height": {"label": "ఎత్తు", "unit": "సెం.మీ", "lo": 90, "hi": 200, "entity_pool": "person"},
    "weight": {"label": "బరువు", "unit": "కిలోలు", "lo": 3, "hi": 120, "entity_pool": "person"},
    "age": {"label": "వయస్సు", "unit": "సంవత్సరాలు", "lo": 5, "hi": 90, "entity_pool": "person"},
    "price": {"label": "ధర", "unit": "రూపాయలు", "lo": 10, "hi": 500000, "entity_pool": "object"},
}

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# Mode weights (not uniform): "equal" always answers the single fixed token "సమానం"
# regardless of which entities are involved, and transitive's "yesno" sub-case always
# answers "అవును" (A>B>C implies A>C by construction) -- both are answer-token-imbalance
# sinks that teach the model nothing about reading values/copying names, and uniform
# sampling let them dominate the answer-token distribution enough to cause the finetuned
# model to collapse onto those two fixed tokens instead of learning to compare/copy. Down-
# weighting them here (and pairing with class-weighted loss in finetune.py) fixes that.
MODE_WEIGHTS = {
    "pairwise_value": 0.30,
    "pairwise_yesno": 0.25,
    "transitive": 0.20,
    "three_value": 0.20,
    "equal": 0.05,
}
TRANSITIVE_ASK_WEIGHTS = {"max": 0.4, "min": 0.4, "yesno": 0.2}


def split_pool(names, seed):
    rng = random.Random(seed)
    names = names[:]
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }


# ============================================================================
# Template functions. Each returns (prompt, answer, template_id).
# "held_out" templates are only ever used for val/test splits (relation-pattern holdout,
# alongside the entity-name holdout).
# ============================================================================

def tpl_pairwise_value(attr_key, A, vA, B, vB, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    is_person = attr["entity_pool"] == "person"
    who = "ఎవరు" if is_person else "ఏది"
    be = "ఉన్నారు" if is_person else "ఉంది"
    if not held_out:
        q = f"{A} {label} {vA} {unit}. {B} {label} {vB} {unit}. {who} ఎక్కువ {label} కలిగి {be}?"
    else:
        q = f"{A} యొక్క {label} {vA} {unit}, మరియు {B} యొక్క {label} {vB} {unit}. వీటిలో {label} ఎక్కువగా ఉన్నది {who}?"
    ans = A if vA > vB else (B if vB > vA else "సమానం")
    return q, ans, f"pairwise_value_{attr_key}" + ("_ho" if held_out else "")


def tpl_pairwise_yesno(attr_key, A, vA, B, vB, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    is_person = attr["entity_pool"] == "person"
    beq = "ఉన్నారా" if is_person else "ఉందా"
    if not held_out:
        q = f"{A} {label} {vA} {unit}, {B} {label} {vB} {unit}. {A}, {B} కంటే ఎక్కువ {label} కలిగి {beq}?"
    else:
        q = f"{A} ({vA} {unit}) {B} ({vB} {unit}) కంటే {label}లో ఎక్కువగా {beq}?"
    ans = "అవును" if vA > vB else "కాదు"
    return q, ans, f"pairwise_yesno_{attr_key}" + ("_ho" if held_out else "")


def tpl_transitive_relation(attr_key, A, B, C, ask, held_out=False):
    """Only relations given (A>B, B>C), no numbers -- matches the PDF's literal example style."""
    attr = ATTRIBUTES[attr_key]
    label = attr["label"]
    is_person = attr["entity_pool"] == "person"
    be = "ఉన్నారు" if is_person else "ఉంది"
    beq = "ఉన్నారా" if is_person else "ఉందా"
    who = "ఎవరు" if is_person else "ఏది"
    among = "వీరిలో" if is_person else "వీటిలో"
    has_who = "కలిగినవారు ఎవరు" if is_person else "కలిగినది ఏది"

    if not held_out:
        premise = f"{A}, {B} కంటే ఎక్కువ {label} కలిగి {be}. {B}, {C} కంటే ఎక్కువ {label} కలిగి {be}."
    else:
        premise = f"{A} యొక్క {label} {B} కంటే ఎక్కువ. అలాగే {B} యొక్క {label} {C} కంటే ఎక్కువ."

    if ask == "max":
        q = f"{premise} {among} అందరికంటే ఎక్కువ {label} {has_who}?"
        ans = A
    elif ask == "min":
        q = f"{premise} {among} అందరికంటే తక్కువ {label} {has_who}?"
        ans = C
    else:  # yesno A vs C
        q = f"{premise} {A}, {C} కంటే ఎక్కువ {label} కలిగి {beq}?"
        ans = "అవును"
    return q, ans, f"transitive_{ask}_{attr_key}" + ("_ho" if held_out else "")


def tpl_three_value_superlative(attr_key, A, vA, B, vB, C, vC, ask, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    if not held_out:
        premise = f"{A} {label} {vA} {unit}. {B} {label} {vB} {unit}. {C} {label} {vC} {unit}."
    else:
        premise = f"{A}: {vA} {unit}, {B}: {vB} {unit}, {C}: {vC} {unit} ({label} ప్రకారం)."

    values = {A: vA, B: vB, C: vC}
    if ask == "max":
        q = f"{premise} వీటిలో అత్యధిక {label} కలిగినది ఏది?"
        ans = max(values, key=values.get)
    else:
        q = f"{premise} వీటిలో అత్యల్ప {label} కలిగినది ఏది?"
        ans = min(values, key=values.get)
    return q, ans, f"three_superlative_{ask}_{attr_key}" + ("_ho" if held_out else "")


def tpl_equal(attr_key, A, B, v, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    is_person = attr["entity_pool"] == "person"
    both = "వీరిద్దరి" if is_person else "వీటి రెండిటి"
    pair = "ఇద్దరి" if is_person else "రెండిటి"
    whose = "ఎవరిది" if is_person else "దేనిది"
    if not held_out:
        q = f"{A} {label} {v} {unit}. {B} {label} కూడా {v} {unit}. {both} {label} ఎలా ఉంది?"
    else:
        q = f"{A}, {B} {pair} {label} {v} {unit} చొప్పున సమానంగా ఉంటే, {whose} ఎక్కువ?"
    ans = "సమానం"
    return q, ans, f"equal_{attr_key}" + ("_ho" if held_out else "")


def gen_value(attr, rng):
    return rng.randint(attr["lo"], attr["hi"])


def gen_distinct_values(attr, rng, n):
    vals = set()
    while len(vals) < n:
        vals.add(gen_value(attr, rng))
    return list(vals)


def build_examples(split_name, person_pool, object_pool, n_target, rng, allow_held_out):
    examples = []
    seen_prompts = set()
    attempts = 0
    max_attempts = n_target * 30

    while len(examples) < n_target and attempts < max_attempts:
        attempts += 1
        attr_key = rng.choice(list(ATTRIBUTES.keys()))
        attr = ATTRIBUTES[attr_key]
        pool = person_pool if attr["entity_pool"] == "person" else object_pool
        if len(pool) < 3:
            continue

        held_out = allow_held_out and rng.random() < 0.25
        mode = rng.choices(list(MODE_WEIGHTS.keys()), weights=list(MODE_WEIGHTS.values()), k=1)[0]

        if mode == "pairwise_value" or mode == "pairwise_yesno":
            A, B = rng.sample(pool, 2)
            vA, vB = gen_distinct_values(attr, rng, 2)
            if mode == "pairwise_value":
                q, ans, tid = tpl_pairwise_value(attr_key, A, vA, B, vB, held_out)
            else:
                q, ans, tid = tpl_pairwise_yesno(attr_key, A, vA, B, vB, held_out)

        elif mode == "transitive":
            A, B, C = rng.sample(pool, 3)
            ask = rng.choices(list(TRANSITIVE_ASK_WEIGHTS.keys()), weights=list(TRANSITIVE_ASK_WEIGHTS.values()), k=1)[0]
            q, ans, tid = tpl_transitive_relation(attr_key, A, B, C, ask, held_out)

        elif mode == "three_value":
            A, B, C = rng.sample(pool, 3)
            vA, vB, vC = gen_distinct_values(attr, rng, 3)
            ask = rng.choice(["max", "min"])
            q, ans, tid = tpl_three_value_superlative(attr_key, A, vA, B, vB, C, vC, ask, held_out)

        else:  # equal
            A, B = rng.sample(pool, 2)
            v = gen_value(attr, rng)
            q, ans, tid = tpl_equal(attr_key, A, B, v, held_out)

        if q in seen_prompts:
            continue
        seen_prompts.add(q)

        prompt = f"ప్రశ్న: {q}\nసమాధానం:"
        text = f"{prompt} {ans}"
        examples.append({
            "prompt": prompt,
            "answer": ans,
            "text": text,
            "template_id": tid,
            "split": split_name,
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-samples", type=int, default=10000, help="Total examples across train+val+test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    person_split = split_pool(PERSON_NAMES, args.seed)
    object_split = split_pool(OBJECT_NAMES, args.seed + 1)

    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    targets = {k: int(args.num_samples * v) for k, v in ratios.items()}
    targets["train"] = args.num_samples - targets["val"] - targets["test"]  # remainder to train

    all_examples = {}
    for split_name in ["train", "val", "test"]:
        split_rng = random.Random(args.seed + hash(split_name) % 10000)
        allow_held_out = split_name in ("val", "test")
        exs = build_examples(
            split_name,
            person_split[split_name],
            object_split[split_name],
            targets[split_name],
            split_rng,
            allow_held_out,
        )
        all_examples[split_name] = exs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, exs in all_examples.items():
        out_path = args.output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"✓ {split_name}: {len(exs)} examples -> {out_path}")

    # Also save a flat one-line-per-example .txt version, matching the pretraining data
    # layout (<lang>/data/<split>/<file>.txt), for finetuning code that reads plain text.
    # Internal prompt/answer newlines are collapsed to a single space so each example stays
    # on exactly one line.
    for split_name, exs in all_examples.items():
        flat_path = SCRIPT_DIR.parent / "data" / split_name / "fine_tune.txt"
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flat_path, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(ex["text"].replace("\n", " ") + "\n")
        print(f"✓ {split_name}: {len(exs)} examples -> {flat_path}")

    template_ids = sorted(set(ex["template_id"] for exs in all_examples.values() for ex in exs))
    stats = {
        "language": "telugu",
        "seed": args.seed,
        "total_examples": sum(len(v) for v in all_examples.values()),
        "split_sizes": {k: len(v) for k, v in all_examples.items()},
        "template_count": len(template_ids),
        "template_ids": template_ids,
        "person_name_pool": {k: len(v) for k, v in person_split.items()},
        "object_name_pool": {k: len(v) for k, v in object_split.items()},
        "leakage_avoidance": (
            "Person and object entity-name pools were each shuffled and split 70/15/15 into "
            "train/val/test BEFORE generation, so no name used in val/test ever appears in "
            "train. Additionally, one template variant per question type ('_ho' suffix, "
            "held-out phrasing) is reserved exclusively for val/test, holding out a "
            "relation-pattern in addition to entity names."
        ),
        "attribute_types": list(ATTRIBUTES.keys()),
        "question_types": ["pairwise_value", "pairwise_yesno", "transitive_max_min_yesno", "three_value_superlative", "equal"],
    }
    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ stats -> {stats_path}")
    print(f"\nTotal: {stats['total_examples']} examples, {stats['template_count']} template variants")


if __name__ == "__main__":
    main()

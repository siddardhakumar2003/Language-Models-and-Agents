#!/usr/bin/env python3
"""
Generate a synthetic Bhojpuri comparative-reasoning finetuning dataset (Phase 3, PDF Sec 3.1).

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

NOTE: Bhojpuri is the lower-resource language in this project and template phrasing below
is best-effort, not verified by a native speaker -- spot-check the rendered .jsonl output
before relying on it for finetuning, and correct phrasing in the tpl_* functions if needed.

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
# Vocabulary (common Bhojpuri-region names / places / objects -- not mined
# from the pretraining corpus, to keep this dataset fully independent of it;
# entity names are split into train/val/test pools below).
# ============================================================================

PERSON_NAMES = [
    "राम", "सीता", "कृष्ण", "राधा", "शिव", "गीता", "मोहन", "सुनीता",
    "विजय", "कमला", "सूरज", "पूजा", "अनिल", "सरिता", "रमेश", "गीतांजलि",
    "दिनेश", "सावित्री", "राजू", "मीना", "संतोष", "कविता", "अजय", "रेखा",
    "विनोद", "शांति", "प्रकाश", "ममता", "सुरेश", "इंदु", "महेश", "उर्मिला",
    "नरेश", "सुशीला", "जगदीश", "पार्वती", "हरीश", "ललिता", "गोपाल", "कुसुम",
    "छोटू", "बबिता", "मुन्ना", "सरोज", "रामू", "कल्पना", "बिनोद", "अनीता", "श्याम", "रीता",
]

OBJECT_NAMES = [
    "किताब", "गाड़ी", "साइकिल", "आम", "घर", "फोन", "कुर्सी",
    "बैग", "घड़ी", "मेज", "छाता", "लैपटॉप", "जूता", "कलम",
    "आईना", "खिलौना", "बक्सा", "तौलिया", "टोकरी", "थाली",
]

# ============================================================================
# Attribute definitions: unit label + realistic numeric range.
# ============================================================================

ATTRIBUTES = {
    "height": {"label": "ऊँचाई", "unit": "सेंटीमीटर", "lo": 90, "hi": 200, "entity_pool": "person"},
    "weight": {"label": "वजन", "unit": "किलो", "lo": 3, "hi": 120, "entity_pool": "person"},
    "age": {"label": "उमिर", "unit": "बरिस", "lo": 5, "hi": 90, "entity_pool": "person"},
    "price": {"label": "दाम", "unit": "रुपिया", "lo": 10, "hi": 500000, "entity_pool": "object"},
}

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


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
    if not held_out:
        q = f"{A} के {label} {vA} {unit} बा। {B} के {label} {vB} {unit} बा। के जादा {label} वाला बा?"
    else:
        q = f"{A} के {label} {vA} {unit} अवुरी {B} के {label} {vB} {unit} बा। इनमें {label} में जादा के बा?"
    ans = A if vA > vB else (B if vB > vA else "बराबर")
    return q, ans, f"pairwise_value_{attr_key}" + ("_ho" if held_out else "")


def tpl_pairwise_yesno(attr_key, A, vA, B, vB, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    if not held_out:
        q = f"{A} के {label} {vA} {unit} बा, {B} के {label} {vB} {unit} बा। का {A}, {B} से जादा {label} वाला बा?"
    else:
        q = f"{A} ({vA} {unit}) {B} ({vB} {unit}) से {label} में जादा बा का?"
    ans = "हँ" if vA > vB else "ना"
    return q, ans, f"pairwise_yesno_{attr_key}" + ("_ho" if held_out else "")


def tpl_transitive_relation(attr_key, A, B, C, ask, held_out=False):
    """Only relations given (A>B, B>C), no numbers -- matches the PDF's literal example style."""
    attr = ATTRIBUTES[attr_key]
    label = attr["label"]
    if not held_out:
        premise = f"{A}, {B} से जादा {label} वाला बा। {B}, {C} से जादा {label} वाला बा।"
    else:
        premise = f"{A} के {label} {B} से जादा बा। ओइसहीं {B} के {label} {C} से जादा बा।"

    if ask == "max":
        q = f"{premise} इनमें सभसे जादा {label} वाला के बा?"
        ans = A
    elif ask == "min":
        q = f"{premise} इनमें सभसे कम {label} वाला के बा?"
        ans = C
    else:  # yesno A vs C
        q = f"{premise} का {A}, {C} से जादा {label} वाला बा?"
        ans = "हँ"
    return q, ans, f"transitive_{ask}_{attr_key}" + ("_ho" if held_out else "")


def tpl_three_value_superlative(attr_key, A, vA, B, vB, C, vC, ask, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    if not held_out:
        premise = f"{A} के {label} {vA} {unit} बा। {B} के {label} {vB} {unit} बा। {C} के {label} {vC} {unit} बा।"
    else:
        premise = f"{A}: {vA} {unit}, {B}: {vB} {unit}, {C}: {vC} {unit} ({label} के हिसाब से)।"

    values = {A: vA, B: vB, C: vC}
    if ask == "max":
        q = f"{premise} इनमें सभसे जादा {label} वाला के बा?"
        ans = max(values, key=values.get)
    else:
        q = f"{premise} इनमें सभसे कम {label} वाला के बा?"
        ans = min(values, key=values.get)
    return q, ans, f"three_superlative_{ask}_{attr_key}" + ("_ho" if held_out else "")


def tpl_equal(attr_key, A, B, v, held_out=False):
    attr = ATTRIBUTES[attr_key]
    label, unit = attr["label"], attr["unit"]
    if not held_out:
        q = f"{A} के {label} {v} {unit} बा। {B} के {label} भी {v} {unit} बा। दुनु के {label} कईसन बा?"
    else:
        q = f"{A} अवुरी {B} दुनु के {label} {v} {unit} बा त केकर जादा बा?"
    ans = "बराबर"
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
        mode = rng.choice(["pairwise_value", "pairwise_yesno", "transitive", "three_value", "equal"])

        if mode == "pairwise_value" or mode == "pairwise_yesno":
            A, B = rng.sample(pool, 2)
            vA, vB = gen_distinct_values(attr, rng, 2)
            if mode == "pairwise_value":
                q, ans, tid = tpl_pairwise_value(attr_key, A, vA, B, vB, held_out)
            else:
                q, ans, tid = tpl_pairwise_yesno(attr_key, A, vA, B, vB, held_out)

        elif mode == "transitive":
            A, B, C = rng.sample(pool, 3)
            ask = rng.choice(["max", "min", "yesno"])
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

        prompt = f"सवाल: {q}\nजवाब:"
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
        "language": "bhojpuri",
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
        "note": "Bhojpuri templates are best-effort and not yet verified by a native speaker -- spot-check data/*.jsonl before finetuning.",
    }
    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ stats -> {stats_path}")
    print(f"\nTotal: {stats['total_examples']} examples, {stats['template_count']} template variants")


if __name__ == "__main__":
    main()

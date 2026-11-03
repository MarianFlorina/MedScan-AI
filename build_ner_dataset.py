"""
Build the NER dataset (v2)
==========================
Replaces the hand-typed character offsets in create_prescription_dataset.py,
~1/3 of which were misaligned (e.g. 'ice daily for ' labelled FREQUENCY).

Here every span is computed programmatically, so offsets are always correct.

Splits (written to ./training_data/):
  train.json       synthetic, templates A, drugs/conditions from the TRAIN vocab
  val.json         synthetic, templates A, drugs/conditions from the VAL vocab
  test_synth.json  synthetic, UNSEEN templates B, drugs/conditions from the TEST vocab
  gold_test.json   hand-written, hand-annotated realistic lines (independent of templates)

Because val/test drugs and conditions never appear in train, scores measure
generalisation to new names rather than memorisation.

Run:  python build_ner_dataset.py
"""

import json
import random
import re
from pathlib import Path

from drug_knowledge import DRUG_INFO, CONDITION_NAMES

SEED = 42
OUT = Path("training_data")
N_TRAIN, N_VAL, N_TEST = 1500, 200, 200

# Label set used for training/evaluation. FORM (Tab/Cap) is left as plain text.
LABELS = ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "CONDITION", "ROUTE"]

# ------------------------------------------------------------------
# Slot vocabularies
# ------------------------------------------------------------------
FORMS = ["Tab", "Cap", "Inj", "Syp", "Susp", "Cream", "Gel", "Tablet", "Capsule", "Syrup"]
FREQS = [
    "once daily", "twice daily", "three times daily", "four times daily",
    "BD", "TDS", "QID", "OD", "HS", "SOS", "stat", "at bedtime", "every 8 hours",
    "every 12 hours", "every 6 hours", "as needed", "once weekly", "in the morning",
    "1-0-1", "1-1-1", "before meals", "after meals",
]
DURS = ["3 days", "5 days", "7 days", "10 days", "14 days", "2 weeks", "4 weeks", "6 weeks", "3 months", "6 months"]
ROUTES = ["oral", "IV", "IM", "subcut", "sublingual", "topical", "PO", "inhaled", "SC"]
EXTRA_DOSES = ["5ml", "10ml", "2 puffs", "20 units", "0.5g", "1g", "2g", "75mcg", "100 mcg", "1 tablet"]

# OCR-style confusions used for noise augmentation
OCR_SWAPS = {"l": "1", "O": "0", "0": "O", "I": "l", "S": "5", "e": "c", "m": "rn"}


def _title(s):
    return " ".join(w.capitalize() for w in s.split())


def split_vocab(items, seed=SEED):
    """Entity-disjoint 70/15/15 split of a vocabulary."""
    items = sorted(items)
    random.Random(seed).shuffle(items)
    a, b = int(len(items) * 0.70), int(len(items) * 0.85)
    return items[:a], items[a:b], items[b:]


def noisy(text, rng, p_case=0.15, p_ocr=0.10):
    """Light augmentation: random case change or one OCR-style character confusion."""
    r = rng.random()
    if r < p_case:
        return text.upper() if rng.random() < 0.5 else text.lower()
    if r < p_case + p_ocr and len(text) > 3:
        idxs = [i for i, c in enumerate(text) if c in OCR_SWAPS]
        if idxs:
            i = rng.choice(idxs)
            return text[:i] + OCR_SWAPS[text[i]] + text[i + 1:]
    return text


# ------------------------------------------------------------------
# Templates.  {SLOT} or {SLOT2}; SLOT in DRUG DOSAGE FREQ DUR ROUTE COND FORM
# ------------------------------------------------------------------
TEMPLATES_A = [
    "{FORM} {DRUG} {DOSAGE} {FREQ} for {DUR}",
    "{FORM} {DRUG} {DOSAGE} {FREQ} for {COND}",
    "{FORM} {DRUG} {DOSAGE} {ROUTE} {FREQ} for {DUR} for {COND}",
    "Patient prescribed {DRUG} {DOSAGE} {FREQ} for {COND}.",
    "{DRUG} {DOSAGE} {FREQ} x {DUR}",
    "Rx: {FORM} {DRUG} {DOSAGE} {ROUTE} {FREQ}",
    "Diagnosis: {COND}. {FORM} {DRUG} {DOSAGE} {FREQ} for {DUR}.",
    "{FORM} {DRUG} {DOSAGE} {FREQ} and {FORM2} {DRUG2} {DOSAGE2} {FREQ2} for {COND}",
    "History of {COND}. Started on {DRUG} {DOSAGE} {FREQ}.",
    "{DRUG} {DOSAGE} {ROUTE} {FREQ} for {DUR}",
]
TEMPLATES_B = [  # never seen during training
    "Pt c/o {COND}; advised {FORM} {DRUG} {DOSAGE} {FREQ} {DUR}",
    "{COND} - {DRUG} {DOSAGE} {ROUTE}, {FREQ} for {DUR}",
    "Continue {DRUG} {DOSAGE} {FREQ}. Add {DRUG2} {DOSAGE2} {FREQ2} for {COND}.",
    "R/ {DRUG} {DOSAGE} {FREQ} ({DUR})",
]

SLOT_RE = re.compile(r"\{([A-Z]+)(\d?)\}")
SLOT_LABEL = {"DRUG": "DRUG", "DOSAGE": "DOSAGE", "FREQ": "FREQUENCY",
              "DUR": "DURATION", "ROUTE": "ROUTE", "COND": "CONDITION", "FORM": None}


def render(template, values, rng=None, noise=False):
    """Fill a template; return {'text', 'entities'} with exact character offsets."""
    out, ents, pos = [], [], 0
    last = 0
    for m in SLOT_RE.finditer(template):
        lit = template[last:m.start()]
        out.append(lit)
        pos += len(lit)
        key = m.group(1) + m.group(2)
        val = values[key]
        label = SLOT_LABEL[m.group(1)]
        if noise and label in ("DRUG", "CONDITION") and rng is not None:
            val = noisy(val, rng)
        out.append(val)
        if label:
            ents.append({"start": pos, "end": pos + len(val), "label": label})
        pos += len(val)
        last = m.end()
    out.append(template[last:])
    text = "".join(out)
    for e in ents:  # self-check: spans must be non-empty and tight
        s = text[e["start"]:e["end"]]
        assert s and s == s.strip(), (text, e)
    return {"text": text, "entities": ents}


def make_examples(n, templates, drugs, conds, rng, noise):
    examples = []
    for _ in range(n):
        d1, d2 = rng.choice(drugs), rng.choice(drugs)

        def dose(d):
            if rng.random() < 0.12:
                return rng.choice(EXTRA_DOSES)
            opts = DRUG_INFO.get(d, {}).get("common_dosages") or EXTRA_DOSES
            v = rng.choice(opts)
            return v.replace("mg", " mg") if rng.random() < 0.1 else v

        cond = rng.choice(conds)
        vals = {
            "FORM": rng.choice(FORMS), "FORM2": rng.choice(FORMS),
            "DRUG": _title(d1) if rng.random() < 0.9 else d1,
            "DRUG2": _title(d2) if rng.random() < 0.9 else d2,
            "DOSAGE": dose(d1), "DOSAGE2": dose(d2),
            "FREQ": rng.choice(FREQS), "FREQ2": rng.choice(FREQS),
            "DUR": rng.choice(DURS), "ROUTE": rng.choice(ROUTES),
            "COND": _title(cond) if rng.random() < 0.8 else cond,
        }
        examples.append(render(rng.choice(templates), vals, rng, noise))
    return examples


# ------------------------------------------------------------------
# Hand-annotated gold set.  [[text|LABEL]] marks an entity.
# Written independently of the templates above, with real-world shorthand,
# multi-drug lines, Indian-style prescriptions and a few OCR-style errors.
# ------------------------------------------------------------------
GOLD_MARKUP = [
    "Tab [[Amoxicillin|DRUG]] [[500mg|DOSAGE]] [[twice daily|FREQUENCY]] for [[7 days|DURATION]]",
    "Inj [[Ceftriaxone|DRUG]] [[1g|DOSAGE]] [[IV|ROUTE]] [[once daily|FREQUENCY]] for [[pneumonia|CONDITION]]",
    "Patient prescribed [[Warfarin|DRUG]] [[5mg|DOSAGE]] [[daily|FREQUENCY]] for [[Atrial Fibrillation|CONDITION]]",
    "Tab [[Metformin|DRUG]] [[500mg|DOSAGE]] [[BD|FREQUENCY]] after meals for [[Type 2 Diabetes|CONDITION]]",
    "Tab [[Sertraline|DRUG]] [[50mg|DOSAGE]] [[once daily|FREQUENCY]] for [[Depression|CONDITION]] and [[Anxiety|CONDITION]]",
    "Tab [[Levothyroxine|DRUG]] [[75mcg|DOSAGE]] [[in the morning|FREQUENCY]] on empty stomach for [[Hypothyroidism|CONDITION]]",
    "Tab [[Clopidogrel|DRUG]] [[75mg|DOSAGE]] [[OD|FREQUENCY]] after [[MI|CONDITION]]",
    "Cap [[Pregabalin|DRUG]] [[75mg|DOSAGE]] [[twice daily|FREQUENCY]] for [[Neuropathic Pain|CONDITION]]",
    "Inj [[Insulin Glargine|DRUG]] [[20 units|DOSAGE]] [[subcut|ROUTE]] [[at bedtime|FREQUENCY]]",
    "Tab [[Paracetamol|DRUG]] [[650mg|DOSAGE]] [[TDS|FREQUENCY]] for [[3 days|DURATION]] for [[fever|CONDITION]]",
    "Tab [[Atorvastatin|DRUG]] [[20mg|DOSAGE]] [[HS|FREQUENCY]] for [[Hyperlipidemia|CONDITION]]",
    "Tab [[Pantoprazole|DRUG]] [[40mg|DOSAGE]] [[OD|FREQUENCY]] before breakfast for [[GERD|CONDITION]]",
    "Syp [[Salbutamol|DRUG]] [[5ml|DOSAGE]] [[three times daily|FREQUENCY]] for [[5 days|DURATION]]",
    "Tab [[Azithromycin|DRUG]] [[500mg|DOSAGE]] [[once daily|FREQUENCY]] for [[3 days|DURATION]]",
    "Pt has [[hypertension|CONDITION]]. Continue [[Amlodipine|DRUG]] [[5mg|DOSAGE]] [[once daily|FREQUENCY]].",
    "Tab [[Aspirin|DRUG]] [[75mg|DOSAGE]] [[OD|FREQUENCY]] and Tab [[Atorvastatin|DRUG]] [[40mg|DOSAGE]] [[HS|FREQUENCY]]",
    "Inj [[Enoxaparin|DRUG]] [[40mg|DOSAGE]] [[subcut|ROUTE]] [[once daily|FREQUENCY]] for [[DVT|CONDITION]]",
    "Tab [[Prednisolone|DRUG]] [[20mg|DOSAGE]] [[once daily|FREQUENCY]] for [[5 days|DURATION]] for [[Asthma|CONDITION]]",
    "Tab [[Cetirizine|DRUG]] [[10mg|DOSAGE]] [[at bedtime|FREQUENCY]] for [[Allergic Rhinitis|CONDITION]]",
    "Cap [[Omeprazole|DRUG]] [[20mg|DOSAGE]] [[BD|FREQUENCY]] for [[2 weeks|DURATION]]",
    "Tab [[Losartan|DRUG]] [[50mg|DOSAGE]] [[1-0-1|FREQUENCY]] for [[Hypertension|CONDITION]]",
    "Tab [[Ibuprofen|DRUG]] [[400mg|DOSAGE]] [[SOS|FREQUENCY]] for [[pain|CONDITION]]",
    "Inj [[Diclofenac|DRUG]] [[75mg|DOSAGE]] [[IM|ROUTE]] [[stat|FREQUENCY]]",
    "Tab [[Ciprofloxacin|DRUG]] [[500mg|DOSAGE]] [[every 12 hours|FREQUENCY]] for [[7 days|DURATION]] for [[Urinary Tract Infection|CONDITION]]",
    "Tab [[Furosemide|DRUG]] [[40mg|DOSAGE]] [[in the morning|FREQUENCY]] for [[Heart Failure|CONDITION]]",
    "Tab [[Gabapentin|DRUG]] [[300mg|DOSAGE]] [[three times daily|FREQUENCY]]",
    "Rx: [[Metoprolol|DRUG]] [[50mg|DOSAGE]] [[PO|ROUTE]] [[BD|FREQUENCY]]",
    "Tab [[Montelukast|DRUG]] [[10mg|DOSAGE]] [[HS|FREQUENCY]] for [[Asthma|CONDITION]]",
    "Tab [[Doxycycline|DRUG]] [[100mg|DOSAGE]] [[twice daily|FREQUENCY]] for [[10 days|DURATION]]",
    "Take [[Lisinopril|DRUG]] [[10mg|DOSAGE]] [[once daily|FREQUENCY]]. History of [[diabetes|CONDITION]].",
    "Tab [[Rosuvastatin|DRUG]] [[10 mg|DOSAGE]] [[at bedtime|FREQUENCY]]",
    "Tab [[Fluoxetine|DRUG]] [[20mg|DOSAGE]] [[once daily|FREQUENCY]] for [[OCD|CONDITION]]",
    "Cap [[Tamsulosin|DRUG]] [[0.4mg|DOSAGE]] [[at bedtime|FREQUENCY]] for [[BPH|CONDITION]]",
    "Tab [[Levofloxacin|DRUG]] [[750mg|DOSAGE]] [[once daily|FREQUENCY]] for [[5 days|DURATION]] for [[pneumonia|CONDITION]]",
    "Tab [[Methotrexate|DRUG]] [[15mg|DOSAGE]] [[once weekly|FREQUENCY]] for [[Rheumatoid Arthritis|CONDITION]]",
    "Tab [[Carvedilol|DRUG]] [[6.25mg|DOSAGE]] [[BD|FREQUENCY]] for [[Heart Failure|CONDITION]]",
    "Patient on [[Digoxin|DRUG]] [[0.25mg|DOSAGE]] [[daily|FREQUENCY]] and [[Spironolactone|DRUG]] [[25mg|DOSAGE]] [[daily|FREQUENCY]]",
    "Tab [[Metronidazole|DRUG]] [[400mg|DOSAGE]] [[TDS|FREQUENCY]] for [[7 days|DURATION]]",
    "Inj [[Ondansetron|DRUG]] [[4mg|DOSAGE]] [[IV|ROUTE]] [[SOS|FREQUENCY]] for [[nausea|CONDITION]]",
    "Tab [[Lev0thyroxine|DRUG]] [[100mcg|DOSAGE]] [[OD|FREQUENCY]] for [[hypothyroidism|CONDITION]]",  # OCR error
    "TAB [[AMLODIPINE|DRUG]] [[10MG|DOSAGE]] [[ONCE DAILY|FREQUENCY]] FOR [[HYPERTENSION|CONDITION]]",  # all caps
]

_MARK = re.compile(r"\[\[(.+?)\|([A-Z]+)\]\]")


def parse_markup(s):
    out, ents, last, pos = [], [], 0, 0
    for m in _MARK.finditer(s):
        lit = s[last:m.start()]
        out.append(lit)
        pos += len(lit)
        ents.append({"start": pos, "end": pos + len(m.group(1)), "label": m.group(2)})
        out.append(m.group(1))
        pos += len(m.group(1))
        last = m.end()
    out.append(s[last:])
    text = "".join(out)
    for e in ents:
        assert e["label"] in LABELS, e
        assert text[e["start"]:e["end"]].strip() == text[e["start"]:e["end"]]
    return {"text": text, "entities": ents}


def main():
    rng = random.Random(SEED)
    drugs = [d for d in DRUG_INFO if re.fullmatch(r"[a-z][a-z \-]+", d)]
    conds = [c for c in CONDITION_NAMES if re.fullmatch(r"[a-z][a-z \-']+", c) and len(c) > 3]
    d_tr, d_va, d_te = split_vocab(drugs)
    c_tr, c_va, c_te = split_vocab(conds)

    OUT.mkdir(exist_ok=True)
    splits = {
        "train": make_examples(N_TRAIN, TEMPLATES_A, d_tr, c_tr, rng, noise=True),
        "val": make_examples(N_VAL, TEMPLATES_A, d_va, c_va, rng, noise=False),
        "test_synth": make_examples(N_TEST, TEMPLATES_B, d_te, c_te, rng, noise=False),
        "gold_test": [parse_markup(s) for s in GOLD_MARKUP],
    }
    for name, data in splits.items():
        with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        counts = {}
        for ex in data:
            for e in ex["entities"]:
                counts[e["label"]] = counts.get(e["label"], 0) + 1
        print(f"{name:11s} {len(data):5d} examples  {counts}")
    print(f"vocab: drugs {len(d_tr)}/{len(d_va)}/{len(d_te)}  conditions {len(c_tr)}/{len(c_va)}/{len(c_te)} (train/val/test)")


if __name__ == "__main__":
    main()

"""
Shared helpers for NER training and evaluation.

Key point: we fine-tune INSIDE the base model's own label space
(d4data/biomedical-ner-all has 84 labels such as B-Medication, B-Dosage).
The previous trainer used a hand-written id table that did not match the
model's real ids (e.g. id 28 is B-Other_event, not B-DRUG), so the fine-tune
was teaching the wrong labels.  Here ids always come from model.config.label2id.
"""

from typing import Dict, List, Tuple

LABELS = ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "CONDITION", "ROUTE"]

# our label  ->  base-model label name (without B-/I-)
OURS_TO_BASE = {
    "DRUG": "Medication",
    "DOSAGE": "Dosage",
    "FREQUENCY": "Frequency",
    "DURATION": "Duration",
    "CONDITION": "Disease_disorder",
    "ROUTE": "Administration",
}
BASE_TO_OURS = {v: k for k, v in OURS_TO_BASE.items()}

Span = Tuple[int, int, str]  # (start, end, label)


def base_label_id(label2id: Dict[str, int], prefix: str, ours: str) -> int:
    return label2id[f"{prefix}-{OURS_TO_BASE[ours]}"]


def example_to_token_labels(text, entities, offsets, label2id) -> List[int]:
    """BIO ids per token from exact character spans. Special tokens -> -100."""
    labels = []
    ent_sorted = sorted(entities, key=lambda e: e["start"])
    started = set()
    for (s, e) in offsets:
        if s == e:  # [CLS]/[SEP]/[PAD]
            labels.append(-100)
            continue
        lab = label2id["O"]
        for i, ent in enumerate(ent_sorted):
            if ent["label"] not in OURS_TO_BASE:
                continue
            if s < ent["end"] and e > ent["start"]:  # token overlaps entity
                prefix = "I" if i in started else "B"
                started.add(i)
                lab = base_label_id(label2id, prefix, ent["label"])
                break
        labels.append(lab)
    return labels


def decode_spans(text, offsets, pred_ids, id2label) -> List[Span]:
    """Turn per-token predictions into character spans in OUR label set."""
    spans: List[Span] = []
    cur = None  # [start, end, label]
    prev_end = None
    for (s, e), pid in zip(offsets, pred_ids):
        if s == e:
            continue
        name = id2label[int(pid)]
        if name == "O":
            tag, base = "O", None
        else:
            tag, base = name.split("-", 1)
        ours = BASE_TO_OURS.get(base)
        continuation = prev_end is not None and s == prev_end and text[s - 1:s].isalnum() and text[s:s + 1].isalnum()
        if ours is None:
            if cur and continuation:
                cur[1] = e  # glue stray sub-word onto current entity
            else:
                if cur:
                    spans.append(tuple(cur))
                cur = None
        elif tag == "B" and not (continuation and cur and cur[2] == ours):
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, ours]
        else:  # I-tag, or B-tag sub-word continuation
            if cur and cur[2] == ours:
                cur[1] = e
            else:
                if cur:
                    spans.append(tuple(cur))
                cur = [s, e, ours]
        prev_end = e
    if cur:
        spans.append(tuple(cur))
    return spans


def gold_spans(entities) -> List[Span]:
    return [(e["start"], e["end"], e["label"]) for e in entities if e["label"] in LABELS]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(all_gold: List[List[Span]], all_pred: List[List[Span]], lenient=False) -> Dict:
    """Entity-level precision/recall/F1.
    strict  : exact (start, end, label) match
    lenient : same label and character overlap
    """
    tot = {l: [0, 0, 0] for l in LABELS}  # tp, fp, fn
    for gold, pred in zip(all_gold, all_pred):
        matched_pred = set()
        for g in gold:
            hit = None
            for j, p in enumerate(pred):
                if j in matched_pred or p[2] != g[2]:
                    continue
                ok = (p[0] < g[1] and p[1] > g[0]) if lenient else (p[0] == g[0] and p[1] == g[1])
                if ok:
                    hit = j
                    break
            if hit is None:
                tot[g[2]][2] += 1
            else:
                matched_pred.add(hit)
                tot[g[2]][0] += 1
        for j, p in enumerate(pred):
            if j not in matched_pred and p[2] in tot:
                tot[p[2]][1] += 1
    out = {}
    TP = FP = FN = 0
    for l, (tp, fp, fn) in tot.items():
        p, r, f = prf(tp, fp, fn)
        out[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}
        TP += tp; FP += fp; FN += fn
    p, r, f = prf(TP, FP, FN)
    out["micro"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": TP + FN}
    return out


def entities_to_spans(text: str, entities: List[Dict]) -> List[Span]:
    """Convert pipeline entity dicts (word/entity_group) to spans by locating the
    word in the text (the pipelines' own offsets are not reliable)."""
    low = text.lower()
    used = set()
    spans: List[Span] = []
    for ent in entities:
        label = ent.get("entity_group") or ent.get("type")
        label = {"STRENGTH": "DOSAGE"}.get(label, label)
        if label not in LABELS:
            continue
        word = ent.get("word") or ent.get("text") or ""
        w = word.lower().strip()
        if not w:
            continue
        idx, start = -1, 0
        while True:
            idx = low.find(w, start)
            if idx == -1:
                break
            if idx not in used:
                break
            start = idx + 1
        if idx == -1:
            continue
        used.add(idx)
        spans.append((idx, idx + len(w), label))
    return spans

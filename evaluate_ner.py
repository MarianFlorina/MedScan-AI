"""
Evaluate NER systems on held-out data
======================================
Systems compared (entity-level precision / recall / F1):
  base      d4data/biomedical-ner-all, no fine-tuning
  legacy_v1 the old adapter (wrong label ids, corrupted offsets)   [if folder exists]
  finetuned the current LoRA adapter in ./finetuned_model
  regex     dictionary/regex extractor only (module2_ner)
  hybrid    the production path: extract_medical_entities()

Test sets (never used for training or model selection):
  test_synth  200 synthetic lines, UNSEEN templates and UNSEEN drug/condition names
  gold_test   41 hand-annotated realistic lines

Metrics: strict = exact span + label; lenient = label + any character overlap.

Run:  python evaluate_ner.py     -> results/ner_eval.json and results/ner_eval.md
"""

import json
import warnings
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_common import LABELS, decode_spans, entities_to_spans, gold_spans, score

warnings.filterwarnings("ignore")
BASE = "d4data/biomedical-ner-all"
DATA = Path("training_data")
RESULTS = Path("results")
MAXLEN = 128


def load(name):
    with open(DATA / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def model_system(adapter_dir=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    @torch.no_grad()
    def predict(text):
        enc = tok(text, max_length=MAXLEN, truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        ids = model(**enc).logits[0].argmax(-1).tolist()
        return decode_spans(text, offsets, ids, id2label)

    return predict


def regex_system():
    from module2_ner import _extract_entities_regex
    return lambda text: entities_to_spans(text, _extract_entities_regex(text))


def hybrid_system():
    from module2_ner import extract_medical_entities
    return lambda text: entities_to_spans(text, extract_medical_entities(text))


def main():
    systems = {"base": model_system()}
    if Path("finetuned_model_v1_legacy/adapter_config.json").exists():
        systems["legacy_v1"] = model_system("finetuned_model_v1_legacy")
    systems["finetuned"] = model_system("finetuned_model")
    systems["regex"] = regex_system()
    systems["hybrid"] = hybrid_system()

    results = {}
    for ds_name in ["test_synth", "gold_test"]:
        data = load(ds_name)
        gold = [gold_spans(e["entities"]) for e in data]
        results[ds_name] = {"n_examples": len(data)}
        for sys_name, predict in systems.items():
            preds = [predict(e["text"]) for e in data]
            results[ds_name][sys_name] = {
                "strict": score(gold, preds),
                "lenient": score(gold, preds, lenient=True),
            }
            m = results[ds_name][sys_name]["strict"]["micro"]
            print(f"{ds_name:10s} {sys_name:10s} strict P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "ner_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# NER evaluation\n", "Entity-level scores on held-out data. See evaluate_ner.py for definitions.\n"]
    for ds_name, res in results.items():
        lines.append(f"\n## {ds_name} ({res['n_examples']} examples)\n")
        lines.append("| System | Strict P | Strict R | Strict F1 | Lenient F1 |")
        lines.append("|---|---|---|---|---|")
        for s in systems:
            m, l = res[s]["strict"]["micro"], res[s]["lenient"]["micro"]
            lines.append(f"| {s} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {l['f1']:.3f} |")
        lines.append("\nPer-class strict F1:\n")
        lines.append("| System | " + " | ".join(LABELS) + " |")
        lines.append("|---|" + "---|" * len(LABELS))
        for s in systems:
            row = [f"{res[s]['strict'][l]['f1']:.3f}" for l in LABELS]
            lines.append(f"| {s} | " + " | ".join(row) + " |")
    (RESULTS / "ner_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {RESULTS / 'ner_eval.md'}")


if __name__ == "__main__":
    main()

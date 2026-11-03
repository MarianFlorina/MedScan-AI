"""
OCR + extraction accuracy on sample prescription images
=======================================================
Ground truth below was transcribed by hand from the images in ./sample_prescriptions
(collage_01.png ... collage_10.png, cropped from a 10-prescription collage and upscaled 3x).

For every prescription it measures
  * drug recall      - how many of the written drugs were found (brand names accepted via aliases)
  * dose accuracy    - of the drugs found, how many have the right dose attached
  * required alerts  - hazards that MUST be flagged (e.g. warfarin + aspirin)
  * false alerts     - CRITICAL/HIGH alerts on prescriptions that should have none

Run:  python evaluate_ocr_samples.py      -> results/ocr_samples_eval.json
"""

import json
import os
import re
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="medscan_ocr_eval_")
os.environ["MEDSCAN_USERS_FILE"] = os.path.join(_TMP, "users.json")
os.environ["MEDSCAN_AUDIT_DIR"] = os.path.join(_TMP, "audit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# (file, language, allergies entered by the user, drugs [(names/aliases..., dose or None)], required alert substrings)
GROUND_TRUTH = [
    ("collage_01.png", "en", [], [(("crocin", "paracetamol"), "650mg"), (("azithral", "azithromycin"), "500mg"),
                                   (("levocetirizine",), "5mg"), (("pantoprazole",), "40mg")], []),
    ("collage_02.png", "en", [], [(("dolo", "paracetamol"), "650mg"), (("mox", "amoxicillin"), "500mg"),
                                   (("ascoril",), "10ml"), (("pantodac", "pantoprazole"), "40mg")], []),
    ("collage_03.png", "en", [], [(("metformin",), "500mg"), (("glimepiride",), "1mg"), (("telmisartan",), "40mg"),
                                   (("amlodipine",), "5mg"), (("atorvastatin",), "10mg")], []),
    ("collage_04.png", "en", [], [(("isotretinoin",), "10mg"), (("clindamycin",), None)], ["TERATOGEN"]),
    ("collage_05.png", "en", [], [(("paracetamol",), None), (("cefixime",), None), (("levocetirizine",), None)], []),
    ("collage_06.png", "hi", [], [(("paracetamol", "पैरासिटामोल"), "500mg"), (("amoxicillin", "अमॉक्सिसिलिन"), "500mg"),
                                   (("levocetirizine", "लेवोसेटिरिज़िन"), "10ml")], []),
    ("collage_07.png", "ta", [], [(("omeprazole", "ஒமிப்ராசோல்"), "20mg"), (("domperidone", "டோம்பெரிடோன்"), "10mg"),
                                   (("simethicone", "சிமெத்திக்கோன்"), "40mg")], []),
    ("collage_08.png", "en", ["penicillin"], [(("warfarin",), "5mg"), (("aspirin",), "75mg"), (("metoprolol",), "25mg"),
                                               (("furosemide",), "40mg")], ["INTERACTION"]),
    ("collage_09.png", "en", [], [(("diclofenac",), "50mg"), (("thiocolchicoside",), "4mg"), (("pregabalin",), "75mg"),
                                   (("pantoprazole",), "40mg")], []),
    ("collage_10.png", "en", [], [(("telmisartan",), "40mg"), (("amlodipine",), "5mg"), (("torsemide",), "10mg"),
                                   (("calcium carbonate", "calcium"), "500mg"), (("vitamin d3", "cholecalciferol"), None)], []),
]


def script():
    import json as _json
    import os
    import streamlit as st
    from PIL import Image

    st.session_state["authenticated"] = True
    st.session_state["user_info"] = {"username": "t", "email": "t@x.org", "role": "admin", "full_name": "T T",
                                     "permissions": {"upload": True, "process": True, "export": True, "admin": True}}
    from app import run_full_pipeline  # noqa: E402

    cfg = _json.loads(os.environ["OCR_EVAL_CFG"])
    out = {}
    for fname, allergies, lang in cfg:
        img = Image.open(os.path.join("sample_prescriptions", fname))
        r = run_full_pipeline(image=img, is_handwritten=False, allergies=allergies, conditions=[], use_ml=False, lang=lang)
        out[fname] = {
            "text": r["raw_text"], "conf": r["ocr_confidence"],
            "meds": [(m["drug"], m["dosage"]) for m in r["medications"]],
            "drug_words": [e["word"] for e in r["entities"] if e["entity_group"] == "DRUG"],
            "alerts": [(f["severity"], f["type"]) for f in (r["anomalies"] or {}).get("findings", [])],
        }
    st.session_state["out"] = out


def norm_dose(s):
    return re.sub(r"\s+", "", (s or "").lower()).replace("milligrams", "mg")


def main():
    only = [x for x in os.environ.get("OCR_EVAL_ONLY", "").split(",") if x]
    if only:
        GROUND_TRUTH[:] = [g for g in GROUND_TRUTH if g[0] in only]
    langs = {"en": "eng", "hi": "eng+hin", "ta": "eng+tam"}
    os.environ["OCR_EVAL_CFG"] = json.dumps([(f, a, langs[l]) for f, l, a, _d, _r in GROUND_TRUTH])
    at = AppTest.from_function(script, default_timeout=1800).run()
    if at.exception:
        print("pipeline error:", [e.value for e in at.exception])
        sys.exit(1)
    out = at.session_state["out"]

    rows, tot_drugs, found_drugs, dose_ok, dose_total, false_alerts = [], 0, 0, 0, 0, 0
    for fname, lang, _al, drugs, required in GROUND_TRUTH:
        o = out[fname]
        words = [w.lower() for w in o["drug_words"]] + [m[0].lower() for m in o["meds"]]
        text_low = o["text"].lower()
        meds = {m[0].lower(): norm_dose(m[1]) for m in o["meds"]}
        f_found, f_dose, f_dose_total, detail = 0, 0, 0, []
        for names, dose in drugs:
            tot_drugs += 1
            hit = next((w for w in words if any(n.lower() in w for n in names)), None)
            in_text = any(n.lower() in text_low for n in names)
            if hit:
                found_drugs += 1; f_found += 1
                if dose:
                    dose_total += 1; f_dose_total += 1
                    from module11_indian_rx import to_generic
                    got = meds.get(hit) or next((meds[n.lower()] for n in names if n.lower() in meds), "") or meds.get(to_generic(hit), "")
                    if got == dose:
                        dose_ok += 1; f_dose += 1
                    detail.append(f"{names[0]}: dose {got or 'MISSING'} (want {dose})")
            else:
                detail.append(f"{names[0]}: NOT EXTRACTED ({'in OCR text' if in_text else 'not in OCR text'})")
        sev = [t for s, t in o["alerts"] if s in ("CRITICAL", "HIGH")]
        req_ok = all(any(r in t for t in (t for _s, t in o["alerts"])) for r in required)
        fa = 0 if required else len(sev)
        false_alerts += 1 if fa else 0
        rows.append({"file": fname, "lang": lang, "ocr_conf": round(o["conf"], 2), "drugs_found": f"{f_found}/{len(drugs)}",
                     "doses_ok": f"{f_dose}/{f_dose_total}", "required_alerts_ok": req_ok if required else None,
                     "serious_alerts": sev, "detail": detail, "ocr_text": o["text"],
                     "meds": o["meds"], "alerts": o["alerts"]})

    print(f"\n{'file':16s} {'lang':4s} {'conf':>5s} {'drugs':>6s} {'doses':>6s}  required-alert / serious alerts")
    for r in rows:
        print(f"{r['file']:16s} {r['lang']:4s} {r['ocr_conf']:5.2f} {r['drugs_found']:>6s} {r['doses_ok']:>6s}  "
              f"{'' if r['required_alerts_ok'] is None else ('OK' if r['required_alerts_ok'] else 'MISSED')} {r['serious_alerts']}")
        for d in r["detail"]:
            print(f"      - {d}")
        if os.environ.get("OCR_EVAL_VERBOSE"):
            print(f"      meds={r['meds']}")
            print(f"      alerts={r['alerts']}")
    en = [r for r in rows if r["lang"] == "en"]
    print(f"\nDrug recall (all): {found_drugs}/{tot_drugs} = {found_drugs / tot_drugs:.0%}")
    print(f"Dose accuracy (of drugs found with a written dose): {dose_ok}/{dose_total} = {dose_ok / max(dose_total, 1):.0%}")
    print(f"English-only drug recall: {sum(int(r['drugs_found'].split('/')[0]) for r in en)}/{sum(int(r['drugs_found'].split('/')[1]) for r in en)}")
    os.makedirs("results", exist_ok=True)
    with open("results/ocr_samples_eval.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "drug_recall": found_drugs / tot_drugs, "dose_accuracy": dose_ok / max(dose_total, 1)}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

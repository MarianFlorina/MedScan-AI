"""
OCR → NER → safety pipeline test on real images / PDFs
=======================================================
Renders a clean prescription image with PIL, a noisy/rotated copy, and also runs the
bundled scanned PDF, all through app.run_full_pipeline (the same code path as the UI).

Run:  python test_ocr_pipeline.py     (needs Tesseract installed)
"""

import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="medscan_ocr_test_")
os.environ["MEDSCAN_USERS_FILE"] = os.path.join(_TMP, "users.json")
os.environ["MEDSCAN_AUDIT_DIR"] = os.path.join(_TMP, "audit")

from streamlit.testing.v1 import AppTest  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += not cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {'' if cond else detail}")


def script():
    import random

    import streamlit as st
    from PIL import Image, ImageDraw, ImageFont

    st.session_state["authenticated"] = True
    st.session_state["user_info"] = {"username": "t", "email": "t@x.org", "role": "admin", "full_name": "T T",
                                     "permissions": {"upload": True, "process": True, "export": True, "admin": True}}
    from app import run_full_pipeline  # noqa: E402  (importing app renders the page once; harmless here)

    def make_img(lines, noisy=False):
        font = None
        for f in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(f, 34)
                break
            except OSError:
                continue
        img = Image.new("RGB", (1100, 90 + 60 * len(lines)), "white")
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            d.text((40, 40 + 60 * i), ln, fill="black", font=font)
        if noisy:
            rnd = random.Random(1)
            px = img.load()
            for _ in range(6000):
                x, y = rnd.randrange(img.width), rnd.randrange(img.height)
                px[x, y] = (rnd.randrange(256),) * 3
            img = img.rotate(2.5, expand=True, fillcolor="white")
        return img

    import time
    steps = []
    clean = make_img(["Rx: Tab Warfarin 5mg once daily", "Tab Aspirin 81mg once daily", "Diagnosis: Atrial Fibrillation"])
    out = {}
    out["clean"] = run_full_pipeline(image=clean, allergies=[], conditions=[], use_ml=False)
    out["noisy"] = run_full_pipeline(image=make_img(["Tab Amoxicillin 500mg three times daily", "Patient has penicillin allergy"], noisy=True),
                                     allergies=["penicillin"], conditions=[], use_ml=False)
    out["apollo"] = run_full_pipeline(image=Image.open("sample_prescriptions/apollo_urti_handwritten.png"),
                                      allergies=[], conditions=[], use_ml=False)
    t0 = time.time()
    with open("Prescription_1.pdf", "rb") as fh:
        out["pdf"] = run_full_pipeline(pdf_bytes=fh.read(), use_ml=False, progress=steps.append)
    pdf_seconds = time.time() - t0
    st.session_state["pdf_seconds"] = pdf_seconds
    st.session_state["steps"] = steps
    st.session_state["out"] = {k: {"text": v["raw_text"], "conf": v["ocr_confidence"],
                                   "drugs": [e["word"].lower() for e in v["entities"] if e["entity_group"] == "DRUG"],
                                   "types": [f["type"] for f in (v["anomalies"] or {}).get("findings", [])],
                                   "has_fhir": bool(v.get("fhir_bundle")), "pages": v["pages_processed"],
                                   "meds": v.get("medications", []),
                                   "conds": [e["word"].lower() for e in v["entities"] if e["entity_group"] == "CONDITION"],
                                   "unvalidated": [e["word"] for e in v["validated_entities"] if not e["validated"]],
                                   "severities": [f["severity"] for f in (v["anomalies"] or {}).get("findings", [])],
                                   "n_fhir": len((v.get("fhir_bundle") or {}).get("entry", []))}
                               for k, v in out.items()}


at = AppTest.from_function(script, default_timeout=600).run()
check("pipeline ran without exception", not at.exception, [e.value for e in at.exception])
out = at.session_state["out"] if not at.exception else {}

if out:
    print("\nClean rendered image")
    c = out["clean"]
    check("OCR text contains 'Warfarin' and '5mg'", "warfarin" in c["text"].lower() and "5mg" in c["text"].lower().replace(" ", ""), c["text"][:80])
    check("both drugs extracted", {"warfarin", "aspirin"} <= set(c["drugs"]), c["drugs"])
    check("warfarin+aspirin interaction raised", any("INTERACTION" in t for t in c["types"]), c["types"])
    check("FHIR bundle produced", c["has_fhir"])

    print("\nNoisy + rotated image with allergy")
    n = out["noisy"]
    check("drug still read through noise", "amoxicillin" in n["text"].lower(), n["text"][:80])
    check("allergy conflict flagged", any("ALLERGY" in t for t in n["types"]), n["types"])
    print("\nHandwritten Apollo prescription (4 drugs, 1-0-1 schedules, 'mg' read as 'mq')")
    a = out["apollo"]
    doses = {m["drug"].lower(): m["dosage"].replace(" ", "") for m in a["meds"]}
    check("all 4 drugs found", set(doses) == {"paracetamol", "azithromycin", "levocetirizine", "pantoprazole"}, doses)
    check("each drug has ITS dose (500/500/5/40 mg)", doses == {"paracetamol": "500mg", "azithromycin": "500mg", "levocetirizine": "5mg", "pantoprazole": "40mg"}, doses)
    check("diagnosis kept as one full phrase (no nested fragments)", a["conds"] == ["acute upper respiratory tract infection", "fever"], a["conds"])
    check("all drugs verified against knowledge base", not a["unvalidated"], a["unvalidated"])
    check("no false CRITICAL/HIGH alerts on a routine URTI prescription", not {"CRITICAL", "HIGH"} & set(a["severities"]), a["severities"])
    check("FHIR: one entry per drug x diagnosis (8), not a blow-up", a["n_fhir"] == 8, a["n_fhir"])


    print("\nScanned PDF (Prescription_1.pdf)")
    p = out["pdf"]
    check("PDF produced non-empty text", len(p["text"].strip()) > 50, p["text"][:50])
    check("page count reported", p["pages"] == 1)
    secs = at.session_state["pdf_seconds"]
    print(f"    (scanned PDF analysed in {secs:.0f}s)")
    check("scanned PDF analysed within 90 s (was ~4 min)", secs < 90, f"{secs:.0f}s")
    steps = at.session_state["steps"]
    check("live progress messages were emitted", len(steps) >= 5 and any("OCR" in m for m in steps), steps)

print(f"\n{'=' * 60}\n  RESULT: {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)

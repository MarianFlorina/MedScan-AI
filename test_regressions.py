"""
Regression tests for bugs found during the v5 review.  Fast: no OCR, no BART, no UI.

Run:  python test_regressions.py
"""
import sys

from module2_ner import extract_medical_entities, extract_relationships
from module5_anomaly import AnomalyDetector, _kw
from module9_medical_qa import _is_faithful

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += not cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {'' if cond else detail}")


print("\nShort keywords must match whole words")
check("'aki' does not match 'taking'", not _kw("aki", "also taking ibuprofen"))
check("'aki' still matches real 'AKI'", _kw("aki", "labs consistent with aki today"))
check("'ill' does not match 'pill'", not _kw("ill", "one pill daily"))
check("'ate' does not match 'rate'", not _kw("ate", "heart rate normal"))
check("'kid' does not match 'kidney'", not _kw("kid", "kidney function"))
check("'kid' matches 'kids'", _kw("kid", "dose for kids"))
check("long stems keep substring semantics", _kw("convuls", "patient had convulsions"))

print("\nNo kidney-injury alert just because the text says 'taking'")
det = AnomalyDetector()
res = det.analyze_prescription([{"drug": "ibuprofen", "dosage": "400mg"}], text="Patient is also taking Ibuprofen 400mg for pain.")
check("no LAB_EMERGENCY_AKI", not any("AKI" in f["type"] for f in res["findings"]), [f["type"] for f in res["findings"]])

print("\nIndication is not an adverse reaction")
check("'for Urticaria' not flagged", not det.check_adverse_reactions("Tab Cetirizine 10mg once daily for Urticaria"))
check("'developed urticaria after starting' flagged", bool(det.check_adverse_reactions("Patient developed urticaria after starting amoxicillin")))

print("\nRelationships pair each drug with ITS dose and condition")
text = ("Patient diagnosed with hypertension. Prescribed Lisinopril 20mg daily, Metformin 500mg for Type 2 Diabetes, "
        "Atorvastatin 40mg for cholesterol, and Omeprazole 20mg for GERD.")
rels = {r["drug"].lower(): r for r in extract_relationships(extract_medical_entities(text), text) if r["relationship_type"] != "co_prescribed"}
check("metformin -> 500mg -> diabetes", rels.get("metformin", {}).get("dosage") == "500mg" and "diabetes" in rels["metformin"]["condition"].lower(), rels.get("metformin"))
check("atorvastatin -> 40mg", rels.get("atorvastatin", {}).get("dosage") == "40mg", rels.get("atorvastatin"))
check("omeprazole -> 20mg -> GERD (repeated '20mg' handled)", rels.get("omeprazole", {}).get("dosage") == "20mg" and rels["omeprazole"]["condition"].lower() == "gerd", rels.get("omeprazole"))
check("no drug is linked to another drug's condition", len(rels) == 3, list(rels))

t2 = "Patient prescribed Warfarin 5mg and Aspirin 81mg."
pairs = [r for r in extract_relationships(extract_medical_entities(t2), t2) if r["relationship_type"] == "co_prescribed"]
check("warfarin+aspirin pair has both doses", pairs and pairs[0]["dosage"] == "5mg + 81mg", pairs)
check("known interacting pair flagged", pairs and pairs[0]["potential_interaction"])
t3 = "Prescribed Metformin 500mg and Amlodipine 5mg."
pairs3 = [r for r in extract_relationships(extract_medical_entities(t3), t3) if r["relationship_type"] == "co_prescribed"]
check("benign pair NOT flagged as interaction", pairs3 and not pairs3[0]["potential_interaction"], pairs3)

print("\nGeneric words are not drugs")
ents = extract_medical_entities("Prescription says 'take 1 tablet daily' but the tablet strength is missing.")
check("'prescription'/'tablet' not tagged DRUG", not [e for e in ents if e["entity_group"] == "DRUG"], [e["word"] for e in ents])

print("\nAssistant: LLM rewrites must stay faithful")
rule = "The prescription includes **2 medication(s)**:\n\n**Warfarin** — 5mg\n**Aspirin** — 81mg"
meds = [{"drug": "Warfarin"}, {"drug": "Aspirin"}]
check("looping output rejected", not _is_faithful("The answer is:\n" + "* 3 medication(s)\n" * 30, rule, meds))
check("answer that drops a drug is rejected", not _is_faithful("The patient takes Warfarin.", rule, meds))
check("faithful rewrite accepted", _is_faithful("Two medications are prescribed: Warfarin 5mg and Aspirin 81mg.", rule, meds))

print("\nSample-prescription regressions (found by testing 10 real-style prescriptions)")
import json as _json, subprocess as _sp
from medscan_helpers import build_meds
from module11_indian_rx import to_generic, transliterate_indic_drugs

t8 = "Known case of: Atrial Fibrillation\nAllergy | Penicillin (rash)\n1. Warfarin 5 mg 1-0-0\n2. Aspirin 75 mg 1-0-0\n3. Metoprolol 25 mg 1-0-1"
m8 = {m["drug"]: m["dosage"].replace(" ", "") for m in build_meds(t8)}
check("each drug keeps ITS dose (warfarin 5, aspirin 75, metoprolol 25)", m8 == {"warfarin": "5mg", "aspirin": "75mg", "metoprolol": "25mg"}, m8)
check("allergen after 'Allergy |' is not a prescribed drug", "penicillin" not in m8, m8)
res8 = AnomalyDetector().analyze_prescription(build_meds(t8), allergies=["penicillin"], text=t8)
check("warfarin+aspirin flagged, no false allergy/lethal alert",
      any(f["type"] == "DRUG_INTERACTION" for f in res8["findings"]) and not any(f["type"] in ("DRUG_ALLERGY", "DOSAGE_LETHAL") for f in res8["findings"]),
      [f["type"] for f in res8["findings"]])
t1 = "1. Crocin 650 mg 1-0-1 x 3 days\n2. Azithral 500 mg 1-0-0 x 3 days\nAdvice:\n- Take plenty of fluids\n- Rest\nDr. R. Meenakshi"
m1 = {m["drug"]: m["dosage"].replace(" ", "") for m in build_meds(t1)}
check("brand names map to generics (Crocin->paracetamol, Azithral->azithromycin)", m1 == {"paracetamol": "650mg", "azithromycin": "500mg"}, m1)
res1 = AnomalyDetector().analyze_prescription(build_meds(t1), text=t1)
check("'Azithral 500 mg 1-0-0' is not an overdose (schedule read via brand name)", not any("DOSAGE" in f["type"] for f in res1["findings"]), [f["type"] for f in res1["findings"]])
t5 = "1. Paracetamol syrup 250 mg/5 ml 7.5 ml 1-1-1 x 3 days\n2. Cefixime syrup 100 mg/5 ml 5 ml 1-0-1 x 5 days"
m5 = {m["drug"]: m["dosage"] for m in build_meds(t5)}
check("'paracetamol syrup' -> paracetamol; strength 250 mg/5 ml is not the dose", "paracetamol" in m5 and m5["paracetamol"] != "250 mg", m5)
t2 = "2) Cap. Mox 500 1-0-1 x 5 days\n4) Tab. Pantodac 40 1-0-0 x 5 days"
m2 = {m["drug"]: m["dosage"] for m in build_meds(t2)}
check("unitless 'Mox 500' / 'Pantodac 40' read as mg via the generic's known strengths", m2.get("amoxicillin") == "500mg" and m2.get("pantoprazole") == "40mg", m2)
check("OCR typo 'Thiocolchicosicde' corrected to thiocolchicoside",
      any(m["drug"] == "thiocolchicoside" for m in build_meds("3. Tab. Thiocolchicosicde 4 mg 1-0-1 x 5 days")))
tt = "Name : Anitha\nAge/Sex : 24 / F\n3) T. Isotretinoin 10 mg 0-0-1 1 month"
rt = AnomalyDetector().analyze_prescription(build_meds(tt), text=tt)
check("isotretinoin for a 24-year-old woman raises a teratogen alert", any(f["type"] == "TERATOGEN_CHILDBEARING_AGE" for f in rt["findings"]), [f["type"] for f in rt["findings"]])
tm = "Name : Ramesh\nAge/Sex : 65 / M\n1. Warfarin 5 mg 1-0-0"
rm = AnomalyDetector().analyze_prescription(build_meds(tm), text=tm)
check("no teratogen alert for a 65-year-old man", not any(f["type"] == "TERATOGEN_CHILDBEARING_AGE" for f in rm["findings"]))
check("Tamil drug name transliterated (ஒமிப்ராசோல் -> omeprazole)", "omeprazole" in transliterate_indic_drugs("1. டாப். ஒமிப்ராசோல் 20 mg"))
check("Hindi drug name transliterated (पैरासिटामोल -> paracetamol)", "paracetamol" in transliterate_indic_drugs("1. टैब. पैरासिटामोल 500 mg"))
code = ("import threading\n"
        "from module2_ml_ner import get_ml_ner\n"
        "r = []\n"
        "t = threading.Thread(target=lambda: r.append(get_ml_ner().is_available))\n"
        "t.start(); r.append(get_ml_ner().is_available); t.join()\n"
        "print('OK' if all(r) and len(r) == 2 else 'FAIL')\n")
proc = _sp.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
check("two threads loading the NER model at once both succeed (warm-up race)", "OK" in proc.stdout, proc.stdout[-200:] + proc.stderr[-200:])

print(f"\n{'=' * 60}\n  RESULT: {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)

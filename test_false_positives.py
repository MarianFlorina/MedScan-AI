"""
False-positive test: SAFE prescriptions must not raise serious alerts
======================================================================
test_all_70.py measures recall (are dangerous cases caught?).  It cannot tell
you whether the system cries wolf.  This file feeds ordinary, guideline-concordant
prescriptions through the same pipeline and counts how many wrongly trigger
CRITICAL or HIGH findings.

Metric:  false-positive rate = safe cases with >=1 CRITICAL/HIGH finding / all safe cases
         (MODERATE/LOW findings are reported but not counted as failures: they are
          advisory notes, e.g. "monitor INR".)

Run:  python test_false_positives.py            (exit code 1 if the rate is above MAX_FP_RATE)
"""

import json
import sys
from pathlib import Path

from medscan_helpers import build_meds
from module5_anomaly import AnomalyDetector

MAX_FP_RATE = 0.10  # target: at most 10% of safe prescriptions raise CRITICAL/HIGH

# (text, allergies, conditions) -- standard drugs at standard doses, no known hazards
SAFE_CASES = [
    ("Tab Amoxicillin 500mg three times daily for 7 days for sinusitis", [], []),
    ("Tab Paracetamol 500mg three times daily for 3 days for fever", [], []),
    ("Tab Metformin 500mg twice daily with meals for Type 2 Diabetes", [], []),
    ("Tab Amlodipine 5mg once daily for Hypertension", [], []),
    ("Tab Atorvastatin 20mg at bedtime for Hyperlipidemia", [], []),
    ("Tab Levothyroxine 75mcg once daily in the morning for Hypothyroidism", [], []),
    ("Tab Pantoprazole 40mg once daily before breakfast for GERD", [], []),
    ("Tab Cetirizine 10mg once daily at bedtime for Allergic Rhinitis", [], []),
    ("Tab Losartan 50mg once daily for Hypertension", [], []),
    ("Cap Omeprazole 20mg once daily for 14 days for Gastritis", [], []),
    ("Tab Azithromycin 500mg once daily for 3 days for Pharyngitis", [], []),
    ("Tab Montelukast 10mg once daily at bedtime for Asthma", [], []),
    ("Tab Sertraline 50mg once daily for Depression", [], []),
    ("Tab Ramipril 5mg once daily for Hypertension", [], []),
    ("Tab Doxycycline 100mg twice daily for 7 days for Acne", [], []),
    ("Tab Metoprolol 50mg twice daily for Hypertension", [], []),
    ("Tab Ciprofloxacin 500mg twice daily for 7 days for Urinary Tract Infection", [], []),
    ("Tab Ibuprofen 400mg three times daily after meals for 3 days for back pain", [], []),
    ("Syp Salbutamol 2mg three times daily for 5 days for cough", [], []),
    ("Tab Prednisolone 20mg once daily for 5 days for Asthma exacerbation", [], []),
    ("Tab Metformin 500mg twice daily and Tab Amlodipine 5mg once daily", [], ["Type 2 Diabetes", "Hypertension"]),
    ("Tab Atorvastatin 20mg at bedtime and Tab Amlodipine 5mg once daily", [], []),
    ("Tab Losartan 50mg once daily and Tab Metformin 500mg twice daily", [], []),
    ("Tab Pantoprazole 40mg once daily and Tab Paracetamol 500mg as needed", [], []),
    ("Tab Cetirizine 10mg at bedtime and Tab Montelukast 10mg at bedtime", [], []),
    ("Tab Amoxicillin 500mg three times daily for 5 days for tonsillitis", ["sulfa"], []),
    ("Tab Paracetamol 500mg as needed for headache", ["penicillin"], []),
    ("Tab Cetirizine 10mg once daily for Urticaria", ["penicillin"], []),
    ("Tab Levothyroxine 100mcg once daily for Hypothyroidism", [], ["Hypothyroidism"]),
    ("Tab Amlodipine 5mg once daily for Hypertension", [], ["Hypertension"]),
    ("Tab Metformin 850mg twice daily for Type 2 Diabetes", [], ["Type 2 Diabetes"]),
    ("Tab Azithromycin 250mg once daily for 5 days for bronchitis", [], []),
    ("Cap Omeprazole 20mg once daily and Tab Domperidone 10mg before meals for dyspepsia", [], []),
    ("Tab Ondansetron 4mg as needed for nausea", [], []),
    ("Tab Telmisartan 40mg once daily for Hypertension", [], []),
    # wording that used to trigger substring false alarms ('taking' -> 'aki', 'pill' -> 'ill', 'rate' -> 'ate')
    ("Patient is taking Atorvastatin 20mg at bedtime and tolerating it well", [], []),
    ("Patient will take one pill of Amlodipine 5mg once daily for Hypertension", [], []),
    ("Also taking Paracetamol 500mg as needed for pain. Heart rate normal.", [], []),
    ("Patient prescribed Lisinopril 20mg daily, Metformin 500mg for Type 2 Diabetes, Atorvastatin 40mg for cholesterol, and Omeprazole 20mg for GERD.", [], []),
    ("Patient diagnosed with hypertension. Prescribed Amlodipine 5mg once daily. Blood pressure stable.", [], []),
]


def main():
    detector = AnomalyDetector()
    rows, fp = [], 0
    for text, allergies, conditions in SAFE_CASES:
        meds = build_meds(text, allergies, conditions)
        res = detector.analyze_prescription(meds, allergies=allergies, conditions=conditions, text=text)
        serious = [f for f in res["findings"] if f["severity"] in ("CRITICAL", "HIGH")]
        advisory = [f for f in res["findings"] if f["severity"] not in ("CRITICAL", "HIGH")]
        if serious:
            fp += 1
        rows.append({
            "text": text, "false_positive": bool(serious),
            "serious": [(f["severity"], f["type"]) for f in serious],
            "advisory": [f["type"] for f in advisory],
        })
        flag = "FALSE-POSITIVE" if serious else "ok"
        print(f"[{flag:14s}] {text[:70]}")
        for f in serious:
            print(f"                 -> {f['severity']} {f['type']}: {f['description'][:90]}")

    n = len(SAFE_CASES)
    rate = fp / n
    print(f"\nSafe prescriptions: {n} | with CRITICAL/HIGH alert: {fp} | false-positive rate: {rate:.1%}")
    Path("results").mkdir(exist_ok=True)
    with open("results/false_positive_eval.json", "w") as f:
        json.dump({"n_safe_cases": n, "false_positives": fp, "false_positive_rate": round(rate, 4), "cases": rows}, f, indent=2)
    sys.exit(0 if rate <= MAX_FP_RATE else 1)


if __name__ == "__main__":
    main()

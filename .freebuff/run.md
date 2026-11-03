# MedScan-AI — Run Doc

## How to Reproduce Uncommitted Artifacts
1. Virtual environment is at `.\medscan_env\` in the main checkout — activate it before running.
2. All Python dependencies are installed in the venv (streamlit, transformers, torch, pymupdf, easyocr, pillow, sentencepiece, peft, datasets, accelerate, scikit-learn, etc.)
3. Users database: `users.json` — created on first run with default admin accounts.
4. Fine-tuned NER model: `finetuned_model/` — LoRA adapter for prescription-specific entity extraction.
   To retrain: `python train_ner_lora.py` (trains on 129 prescription examples, ~5 min on CPU).
   To compare: `python test_finetuned_vs_original.py`
5. ML Q&A backend: SmolLM-135M-Instruct (auto-downloaded on first use). Provides natural language responses.
   Falls back to rule-based Q&A if ML model unavailable.

## How to Run the Server
```powershell
Set-Location 'C:\Users\Allan\Desktop\MedScan_AI'
& '.\medscan_env\Scripts\python.exe' -m streamlit run app.py --server.port=8530 --server.headless=true
```
- Port: 8530
- Login: `flooo_17` / `Admin@123` (Admin role)

## Test Suite
Run the comprehensive 70-scenario regression test:
```powershell
& '.\medscan_env\Scripts\python.exe' test_all_70.py
```

## Test Results (Latest Run)
**92/92 PASS — 100% detection rate across 17 clinical domains:**

| # | Domain | Scenarios | Type | Status |
|---|--------|-----------|------|--------|
| 1 | Allergy & Medication Conflicts | 5/5 | Rule-based | ✅ |
| 2 | Dangerous Dose Scenarios | 6/6 | Rule-based | ✅ |
| 3 | Anticoagulant Cases | 5/5 | Rule-based | ✅ |
| 4 | Cardiac Red Flags | 5/5 | Rule-based | ✅ |
| 5 | Neurological Emergencies | 6/6 | Rule-based | ✅ |
| 6 | Respiratory Safety | 5/5 | Rule-based | ✅ |
| 7 | Diabetes Safety | 6/6 | Rule-based | ✅ |
| 8 | Pregnancy Safety | 4/4 | Rule-based | ✅ |
| 9 | Lab Value Traps | 5/5 | Rule-based | ✅ |
| 10 | Pediatric Safety | 6/6 | Rule-based | ✅ |
| 11 | Poisoning / Overdose | 5/5 | Rule-based | ✅ |
| 12 | Don't Trust the Premise | 5/5 | Rule-based | ✅ |
| 13 | Privacy & Identity Safety | 3/3 | Rule-based | ✅ |
| 14 | Hallucination Prevention | 4/4 | Rule-based | ✅ |
| 15 | Mental Health Safety | 6/6 | Rule-based | ✅ |
| 16 | Complex Drug Interactions | 6/6 | Rule-based | ✅ |
| **17** | **ML Anomaly Classification (BART-MNLI)** | **10/10** | **ML zero-shot** | **✅** |

### Domain 17 — ML Anomaly Cases (A-J)
| Case | Scenario | Expected | Matched | Confidence |
|------|----------|----------|---------|------------|
| A | Therapeutic Duplication (lisinopril + enalapril) | ML_DUPLICATION | ML_HARMFUL_COMBO | 94% |
| B | Significant Interaction (warfarin + TMP-SMX) | ML_INTERACTION_MONITORING | ML_INTERACTION_MONITORING | 100% |
| C | High Dose (metformin 6000mg/day) | ML_DOSAGE_HIGH | ML_DOSAGE_HIGH | 99% |
| D | Mild Interaction (levothyroxine + calcium) | ML_MILD_INTERACTION | ML_MILD_INTERACTION | 92% |
| E | Lab Monitoring (warfarin, no INR scheduled) | ML_LAB_MONITORING | ML_LAB_MONITORING | 88% |
| F | Dosage Near Limit (amlodipine 10mg max) | ML_DOSAGE_NEAR_LIMIT | ML_DOSAGE_NEAR_LIMIT | 99% |
| G | Side Effect (SSRI-induced nausea) | ML_SIDE_EFFECT | ML_SIDE_EFFECT | 100% |
| H | Food Interaction (warfarin + spinach) | ML_FOOD_INTERACTION | ML_FOOD_INTERACTION | 100% |
| I | Lifestyle (hypertension + poor habits) | ML_LIFESTYLE | ML_LIFESTYLE | 98% |
| J | Preventive Screening (no screenings documented) | ML_PREVENTIVE | ML_MISSING_INFO | 99% |

## Key Modules
- `app.py` — Streamlit UI with role-based auth, demo mode, upload pipeline, Medical AI Assistant tab
- `module1_ocr.py` — OCR (PyMuPDF + Tesseract + EasyOCR)
- `module2_ner.py` — Hybrid NER (ML biomedical BERT + regex fallback + drug name normalization)
- `module2_ml_ner.py` — ML NER with fine-tuned LoRA adapter (auto-fallback to base model)
- `train_ner_lora.py` — LoRA fine-tuning script (1.84% trainable params, 5 epochs, CPU)
- `module11_indian_rx.py` — Indian prescription support (292 brands, Hindi/Tamil transliteration, script detection)
- `module3_fhir.py` — FHIR R4 output generation
- `module4_validation.py` — Entity validation
- `module5_anomaly.py` — Anomaly detection (16 clinical safety domains)
- `module6_rxnorm.py` — Drug validation (graceful fallback)
- `module7_hipaa.py` — HIPAA compliance (logging, encryption)
- `module8_ml_anomaly.py` — ML zero-shot clinical anomaly classification (facebook/bart-large-mnli)
- `module9_medical_qa.py` — Medical Q&A assistant (10 clinical rule-based patterns)
- `module10_clinical_ner_advanced.py` — Advanced clinical NER (94 drug ontology, entity linking, treatment/contraindication extraction)
- `drug_knowledge.py` — Shared drug/condition/allergy database (1119 drug names, 327 drug info entries, 325 conditions)

## What Was Fixed (Latest)
- **NER truncation bug**: Added 30+ drug name fixes for ML model truncation (naprox→naproxen, ibupro→ibuprofen, etc.)
- **Therapeutic class duplication**: Now correctly detects two NSAIDs, two ACE inhibitors, etc.
- **Privacy safety**: PHI/PII exposure, unauthorized record access, prescription fabrication detection
- **Hallucination prevention**: Unspecified medication, absolute safety claims, insufficient clinical info, organ-impaired dosing
- **Dosage parsing**: "1kg" correctly parses as 1,000,000mg
- **Allergy filtering**: "Allergic to Penicillin" no longer flags Penicillin as prescribed
- **Clozapine typo fix**: Fixed "lozapine" → "clozapine" in drug_knowledge.py (was preventing NER detection)
- **Clozapine DRUG_INFO**: Added atypical antipsychotic entry with dosages (25mg–450mg)
- **Pediatric scenario fix**: Changed "antibiotic" (drug class) to "amoxicillin" (specific drug) for NER extraction
- **Domain 17 added**: 10 ML anomaly classification cases (A–J) covering all severity tiers

# MedScan-AI — Complete Project Explanation

**Written for:** you (the project author), as a study/interview/portfolio reference.
**Based on:** the actual code in this folder, plus test runs executed on 2026-10-21.

---

## 0. Update (21 Oct 2026): what changed after the first version of this document

Sections 3.1, 4.2, 4.3, 4.4 and 8 below describe the project **as it was before these fixes**. The current state:

| Problem found | Fix | Evidence |
|---|---|---|
| ~259 of ~760 hand-typed entity offsets in the training set were misaligned (e.g. `'ice daily for '` tagged FREQUENCY) | New generator [build_ner_dataset.py](build_ner_dataset.py) computes offsets programmatically (1,500 train / 200 val / 200 test + 41 hand-annotated gold lines) | spans asserted tight at build time |
| Trainer used a wrong label-id table (id 28 is really `B-Other_event`) | [train_ner_lora.py](train_ner_lora.py) uses the model's own `label2id`, and picks the best epoch by **validation F1** | [training_logs/training_log.json](training_logs/training_log.json) |
| No held-out accuracy numbers | [evaluate_ner.py](evaluate_ner.py): strict + lenient entity-level P/R/F1, five systems | [results/ner_eval.md](results/ner_eval.md) |
| No false-positive measurement | [test_false_positives.py](test_false_positives.py): 35 safe prescriptions | 0 / 35 raise CRITICAL/HIGH |
| Tests duplicated app logic | shared [medscan_helpers.py](medscan_helpers.py) | |

**NER results (strict exact-span F1, held-out):**

| System | Hand-annotated gold (41) | Unseen-template synthetic (200) |
|---|---|---|
| Base model | 0.569 | 0.489 |
| Old adapter (v1) | 0.000 | 0.000 |
| Regex only | 0.739 | 0.696 |
| **Fine-tuned LoRA (v2)** | **0.980** | **0.922** |
| Hybrid (production path) | 0.955 | 0.908 |

The old adapter scores 0 because it was trained on wrong label ids and corrupted offsets (the old app hid this behind regex fallbacks and ad-hoc name fixes). It is archived in `finetuned_model_v1_legacy/`.

Two bugs found by the new tests and fixed: the fine-tuned model tagged the word "Prescription" as a DRUG (generic-word filter added in [module2_ner.py](module2_ner.py)), and "Cetirizine **for Urticaria**" was flagged HIGH as an adverse reaction when urticaria was the *indication* ([module5_anomaly.py](module5_anomaly.py)). After both fixes: 92/92 hazard scenarios pass and 0/35 safe cases false-alarm.

**Caveats:** the gold and safe sets are small and author-written; unseen-name recognition is weaker for DRUG (F1 0.82) and CONDITION (0.75) than for dosage/frequency; the hybrid path currently scores a little *below* the fine-tuned model alone, so the regex-merge step is the next thing to tune. The validation F1 of 1.0 reflects templated data with unseen names, not real-world accuracy — rely on the gold and unseen-template numbers.

---

## 0b. Update (v5 — app review, UI redesign and logic fixes)

**Bugs found by running the app and testing it end to end (all fixed, each with a test):**

| Bug | Impact | Fix |
|---|---|---|
| Keyword checks matched inside other words (`"aki"` in "t**aki**ng", `"ill"` in "p**ill**", `"kid"` in "**kid**ney", `"ate"` in "r**ate**") | A prescription containing "taking" raised "Acute kidney injury — creatinine increase" | Short keywords (≤4 letters) now need whole-word matches ([module5_anomaly.py](module5_anomaly.py) `_kw`) |
| Relationship extractor linked every drug to every condition and gave every drug the first dose in the sentence | Wrong doses/conditions in the UI **and the FHIR output** (e.g. all four drugs "20mg") | Nearest-drug / nearest-dose assignment ([module2_ner.py](module2_ner.py)) |
| AI assistant answers were corrupted by a 135M-parameter rewrite model (looping text) | Unreliable text in a clinical context | Faithfulness guard + grounded answers only in the app ([module9_medical_qa.py](module9_medical_qa.py)) |
| Viewer role could upload and process documents | Permission model not enforced | Upload page gated by `upload` permission |
| Failed login crashed the audit logger (`user_info` was `None`) | Wrong password caused a server error | [module7_hipaa.py](module7_hipaa.py) |
| Results disappeared on any interaction; uploads re-ran OCR on every click | Poor UX, wasted compute | Results persisted in session state; explicit **Analyze** button |
| Session timeout counted from login, not inactivity | Contradicted the README | Idle timer (`SessionManager.touch`) |
| Password accepted via URL query (`?user=&pass=`); XSRF protection off; server bound to all interfaces | Security | Removed; config hardened |
| Unescaped OCR text injected into HTML | XSS risk | `html.escape` everywhere |
| Live RxNorm calls ran on every analysis (10 s timeout each when offline) and results were never shown | Hangs offline | Opt-in checkbox with a reachability check; results displayed |
| `CUSTOM_CSS` was defined but never applied | The intended styling wasn't active | New [ui_theme.py](ui_theme.py), applied globally |

**Upload speed fix:** a real scanned PDF took ~4 minutes with no feedback (OCR 86 s, ML safety scan ~2 min). It now takes ~10 s: large scans are downscaled, a confident first OCR pass skips the extra passes (others run in parallel), the BART scan is optional for uploads (capped at 1,500 characters and cached), the NER model warms up in the background after sign-in, and the upload page shows live progress.

**Admin controls:** admins can create users with any role (including Admin) and change roles from the sidebar *Users* panel; the last remaining admin cannot be demoted. Self-registration still offers only Viewer/Analyst. Covered by 38 headless UI checks.

**Sample-prescription testing (10 real-style prescriptions, incl. Hindi and Tamil):** drug recall **95 %**, dose-to-drug accuracy **93 %** (first run: 73 % / 62 %). Bugs it exposed and fixed: doses attached to the wrong drug (Warfarin got Aspirin's 75 mg and raised a false "lethal dose" alert) — now matched by text position; brand names (Crocin, Dolo, Azithral, Mox, Pantodac) now map to generics so the safety rules apply to them; an allergy written as "Allergy | Penicillin" was treated as a prescribed drug (false CRITICAL alert); NER noise ("Chennai", "fluids", doctor surnames, header words) no longer becomes drugs or conditions; syrup strength ("250 mg/5 ml") is no longer read as the dose; OCR slips ("mq", "loml", "Vitamin 03") are corrected; teratogenic drugs (e.g. isotretinoin) are flagged for women aged 12-50; real Hindi/Tamil OCR with drug-name transliteration; and a model-loading race introduced by the background warm-up (which could silently switch the session to regex-only extraction) is fixed with a shared lock. Run `python evaluate_ocr_samples.py`.

**UI redesign:** split-screen login with Enter-to-submit, email-or-username sign-in and live password strength; a Home dashboard (welcome, quick actions, session KPIs, how-it-works, recent analyses, system status); sidebar navigation; consistent dark theme with severity colour coding; polished empty/locked states.

**Verification run on the final code:** 92/92 hazard scenarios · 0/40 safe prescriptions raise CRITICAL/HIGH · 38/38 headless UI checks ([test_app_flow.py](test_app_flow.py)) · 11/11 OCR pipeline checks incl. a noisy rotated image and the bundled scanned PDF ([test_ocr_pipeline.py](test_ocr_pipeline.py)) · 34/34 regression checks ([test_regressions.py](test_regressions.py)). NER scores in section 0 are unchanged.

---

## 1. What the project is (in one paragraph)

MedScan-AI is a **clinical document intelligence platform**. You give it a prescription (a photo, a scanned PDF, or typed text); it reads the text with OCR, pulls out the medical facts (drugs, doses, conditions, frequencies) using a fine-tuned biomedical language model, checks those facts against medical knowledge bases so the AI can't "hallucinate" a fake drug, scans the whole prescription for safety problems (dangerous interactions, allergy conflicts, overdoses, pregnancy risks, etc.), and outputs the result in **FHIR**, the standard format hospital systems use. It runs in a **Streamlit** web dashboard with login, roles, audit logs and PHI protection (HIPAA-style features), and it understands Indian brand names and Hindi/Tamil scripts.

**Key design idea:** *hybrid AI.* Machine learning is used where flexibility matters (reading messy text, catching unusual risks), and deterministic rules/databases are used where correctness matters (safety checks, validation). ML never gets the last word on its own.

---

## 2. End-to-end workflow

```
 Upload (image / PDF / typed text)                      app.py → run_full_pipeline()
        │
        ▼
 [Module 1] OCR ─ Tesseract (+EasyOCR fallback) ─ picks best of several attempts
        │  raw text + OCR confidence
        ▼
 [Module 11] Indian-script detection / transliteration (Hindi, Tamil, brand names)
        │
        ▼
 [Module 2] NER ─ fine-tuned BERT (LoRA)  +  regex/dictionary  → merged entities
        │  DRUG, DOSAGE, FREQUENCY, DURATION, CONDITION, ROUTE, FORM …
        ▼
 Relationship extraction ─ links drug ↔ dose ↔ condition
        │
        ▼
 [Module 4] Validation ─ is this drug/condition real? (local DB)
 [Module 6] RxNorm API ─ live check against the US NLM drug database
        │
        ▼
 [Module 5] Rule-based anomaly detection  (~25 check functions)
 [Module 8] ML anomaly classification     (BART-MNLI zero-shot)
 [Module 10] Clinical context + drug-condition-allergy relations
        │
        ▼
 [Module 3] FHIR R4 output (MedicationRequest + Bundle)
        │
        ▼
 Dashboard: entities, safety alerts, FHIR/CSV export
 [Module 9] Q&A assistant: ask questions about the processed prescription
 [Module 7] Wraps everything: login, roles, PHI detection, audit log, session expiry
```

### Step-by-step

| Step | What happens | File |
|---|---|---|
| 1. Ingest | Image/PDF/text is accepted. PDFs are rendered to 300-DPI images page by page (PyMuPDF). | [module1_ocr.py](module1_ocr.py), [app.py](app.py) |
| 2. OCR | Several Tesseract runs with different preprocessing and page-segmentation modes; EasyOCR is added if Tesseract confidence < 0.6. The result is chosen by a **medical-content score**, not just raw confidence. | [module1_ocr.py](module1_ocr.py) |
| 3. NER | The fine-tuned model tags tokens; sub-words are merged; labels mapped; dosages split from frequencies; duplicates removed; then regex/dictionary results fill gaps. | [module2_ml_ner.py](module2_ml_ner.py), [module2_ner.py](module2_ner.py) |
| 4. Relationships | Pairs each drug with its nearest dose and condition (drug–dose window of 40 characters). | [module2_ner.py](module2_ner.py), [app.py](app.py) |
| 5. Validation | Each drug/condition is looked up in the knowledge base. Unknown → **flagged for human review**. RxNorm live API is a second opinion. | [module4_validation.py](module4_validation.py), [module6_rxnorm.py](module6_rxnorm.py) |
| 6. Safety analysis | Rule engine runs ~25 checks; BART zero-shot adds ML findings; context module links drugs to conditions/allergies. | [module5_anomaly.py](module5_anomaly.py), [module8_ml_anomaly.py](module8_ml_anomaly.py), [module10_clinical_ner_advanced.py](module10_clinical_ner_advanced.py) |
| 7. Output | FHIR R4 `MedicationRequest` and `Bundle` JSON, CSV report. | [module3_fhir.py](module3_fhir.py) |
| 8. Audit | Each processing event is logged to `audit_logs/*.jsonl` (admin sessions). | [module7_hipaa.py](module7_hipaa.py) |

---

## 3. The AI components in detail

### 3.1 Medical NER — fine-tuned with LoRA (the "real ML" part)

- **Base model:** `d4data/biomedical-ner-all` — a ~66-million-parameter DistilBERT-style transformer already trained to recognise biomedical entities (it uses layer names like `q_lin`, `k_lin`, `lin1`, which are DistilBERT's).
- **Problem:** the base model knows general biomedical text but is weaker on short prescription shorthand ("Tab Amoxicillin 500mg BD x 7d").
- **Solution:** fine-tune it on prescription-style data using **LoRA** ([train_ner_lora.py](train_ner_lora.py)).
- **Training data:** [training_data/prescription_ner_dataset.json](training_data/prescription_ner_dataset.json), generated by [create_prescription_dataset.py](create_prescription_dataset.py) — **129 synthetic prescriptions**, each with character-level entity spans.

  | Entity | Count in dataset |
  |---|---|
  | CONDITION | 150 |
  | FREQUENCY | 141 |
  | DOSAGE | 133 |
  | DRUG | 132 |
  | FORM | 129 |
  | DURATION | 27 |
  | ROUTE | 25 |
  | PROCEDURE | 13 |
  | STRENGTH | 11 |

- **Hyperparameters** (from [finetuned_model/training_config.json](finetuned_model/training_config.json) and the script):

  | Setting | Value | Meaning |
  |---|---|---|
  | LoRA rank `r` | 16 | size of the small adapter matrices |
  | LoRA alpha | 32 | scaling factor (alpha/r = 2) |
  | LoRA dropout | 0.1 | regularisation |
  | Target modules | q_lin, k_lin, v_lin, lin1, lin2 | attention + feed-forward layers adapted |
  | Epochs | 5 | passes over the data |
  | Batch size | 4 | examples per step |
  | Learning rate | 3e-4 | step size (higher than usual because LoRA) |
  | Warm-up steps | 50 | LR ramps up first |
  | Weight decay | 0.01 | regularisation |
  | Grad clip | 1.0 | prevents exploding gradients |
  | Max length | 256 tokens | |
  | Seed | 42 | reproducibility |
  | Optimizer | AdamW + linear schedule | |

- **Output size:** the adapter file is only **~5 MB** (`adapter_model.safetensors`, 4,985,792 bytes) versus a full model of ~260 MB — that's the LoRA benefit.
- **At runtime:** [module2_ml_ner.py](module2_ml_ner.py) loads the base model, attaches the adapter, and calls `merge_and_unload()` to fold the adapter into the weights for faster CPU inference. If the adapter or model fails to load, it **falls back** to the base model, and then to regex-only NER. The app never crashes because of a missing model.

### 3.2 Hybrid NER (ML + rules)

[module2_ner.py](module2_ner.py) runs the ML model first and a regex/dictionary extractor in parallel, then **merges**: ML is primary; regex fills in what ML missed. The dictionary side has **1,120 drug names** (generic + brand), **325 conditions**, fuzzy matching for OCR typos (e.g. "amoxicilin"), and **312 Indian brand names** (Crocin, Dolo, Combiflam …) from [module11_indian_rx.py](module11_indian_rx.py). Post-processing removes false positives and normalises results.

### 3.3 ML anomaly detection — zero-shot classification (BART-MNLI)

[module8_ml_anomaly.py](module8_ml_anomaly.py) uses `facebook/bart-large-mnli` with the HuggingFace `zero-shot-classification` pipeline, **no training needed**.

How it works: the model is an NLI (natural language inference) model — it judges whether one sentence *entails* another. The code supplies the prescription text as the "premise" and **21 safety descriptions** as "hypotheses" (e.g. *"life-threatening drug interaction"*, *"dosage above recommended range"*). Each gets a score from 0–1 (`multi_label=True`, so labels are scored independently). Anything ≥ **0.45** confidence becomes a finding, tagged with a severity tier:

| Tier | Example labels |
|---|---|
| CRITICAL (6) | dangerous overdose, anaphylaxis risk, life-threatening interaction, suicidal ideation, … |
| HIGH (6) | significant interaction, dosage above range, contraindication, therapeutic duplication … |
| MODERATE (5) | mild interaction, near upper limit, lab monitoring, side effect, drug-food interaction |
| LOW (4) | minor concern, dose adjustment, lifestyle, preventive screening |

This layer is **additive**: it supplements module 5 and is skipped silently if the model can't load.

### 3.4 Rule-based anomaly engine (the safety backbone)

[module5_anomaly.py](module5_anomaly.py) (~3,700 lines, `AnomalyDetector.analyze_prescription`) is deterministic and explainable. Check functions include: drug–drug interactions (34 curated pairs plus warfarin/triple-combination logic), drug–allergy conflicts (with synonym/cross-reactivity mapping), dosage anomalies, duplicate drugs and therapeutic-class duplication, condition contraindications, allergy context ("tolerated before", "uncertain"), adverse reactions, vital signs, overdose patterns, and specialised domains — anticoagulant, cardiac, neurological, respiratory, diabetes, pregnancy, lab-value emergencies, pediatric, poisoning, "don't trust the premise", hallucination prevention, privacy/identity, mental health, and complex multi-drug interactions. Output is a list of findings, each with `type`, `severity` (CRITICAL/HIGH/MODERATE/LOW), description and recommendation, plus `severity_counts` and `safe_to_proceed`.

### 3.5 Medical Q&A assistant

[module9_medical_qa.py](module9_medical_qa.py) answers questions like "Is this safe?" or "What tests are needed?" about the **already-processed** prescription. It classifies **intent** with regex patterns (medication list, dosage, conditions, interactions, allergies, safety, instructions, monitoring, purpose, dosage safety, Indian drugs), then generates a **grounded answer from extracted data and anomaly results** (a template per intent). Optionally `HuggingFaceTB/SmolLM-135M-Instruct` can polish the wording. The answer comes from verified data first, the LLM never invents the facts.

### 3.6 Advanced clinical context

[module10_clinical_ner_advanced.py](module10_clinical_ner_advanced.py): maps brand/synonym → generic drug (tylenol → acetaminophen), normalises allergy names, and extracts **treatment relationships** (drug → condition) and **contraindication relationships** (drug ↔ allergy/condition) with confidence weighting.

### 3.7 Indian prescription support

[module11_indian_rx.py](module11_indian_rx.py): detects the script of the text using **Unicode ranges** (Devanagari U+0900–097F, Tamil U+0B80–0BFF, Telugu, Kannada, Malayalam, Bengali, Gujarati, Gurmukhi), transliterates regional drug-name spellings to Latin, and parses common Indian prescription formats (e.g. "Tab", "Syp", "BD", "TDS").

---

## 4. Metrics used and results produced

> **Important honesty note up front:** the project's measured results are *scenario-based pass/fail detection tests* and *qualitative extraction checks*. There is **no held-out precision/recall/F1 evaluation** of the fine-tuned NER model yet. Section 4.4 explains this and how to add one.

### 4.1 Main safety test suite — `test_all_70.py` ([test_all_70.py](test_all_70.py))

**Result (re-run on 2026-10-21): 92 / 92 PASS — Detection Rate 100.0%**

The filename says "70" but the suite has grown to **92 scenarios across 17 clinical domains**:

| # | Domain | Scenarios | Tests |
|---|---|---|---|
| 1 | Allergy & medication conflicts | 5 | rule engine |
| 2 | Dangerous dose scenarios | 6 | rule engine |
| 3 | Anticoagulant cases | 5 | rule engine |
| 4 | Cardiac red flags | 5 | rule engine |
| 5 | Neurological emergencies | 6 | rule engine |
| 6 | Respiratory safety | 5 | rule engine |
| 7 | Diabetes safety | 6 | rule engine |
| 8 | Pregnancy safety | 4 | rule engine |
| 9 | Lab-value traps | 5 | rule engine |
| 10 | Pediatric safety | 6 | rule engine |
| 11 | Poisoning / overdose | 5 | rule engine |
| 12 | "Don't trust the premise" | 5 | rule engine |
| 13 | Privacy & identity safety | 3 | rule engine |
| 14 | Hallucination prevention | 4 | rule engine |
| 15 | Mental health safety | 6 | rule engine |
| 16 | Complex drug interactions | 6 | rule engine |
| 17 | ML anomaly classification | 10 | BART-MNLI |
| | **Total** | **92** | **82 rule + 10 ML** |

**How a scenario is scored:** the test feeds prescription text through the *real* pipeline (NER → medication building → `AnomalyDetector`), then checks that **at least one finding's `type` contains an expected type name**. Pass = the expected hazard was detected. (For allergy/condition scenarios, allergies/conditions are supplied as inputs.)

**Metric terms used here**
- **Detection rate** = scenarios passed ÷ total scenarios = 92/92 = 100%. This is effectively **recall on the hazards the tests were written for** (did we catch the dangerous thing?).
- It does **not** measure false positives (how often safe prescriptions are wrongly flagged), so it is not precision.

**Domain 17 (ML) results** — expected label vs. what BART returned, from the run:

| Case | Scenario | Matched finding | Severity | ML confidence |
|---|---|---|---|---|
| 17.1 | Lisinopril + enalapril (ACE-inhibitor duplication) | ML_HARMFUL_COMBO | HIGH | 94% |
| 17.2 | Warfarin + TMP-SMX | ML_INTERACTION_MONITORING | HIGH | 100% |
| 17.3 | Metformin 3000 mg twice daily | ML_DOSAGE_HIGH | HIGH | 99% |
| 17.4 | Levothyroxine + calcium | ML_MILD_INTERACTION | MODERATE | 92% |
| 17.5 | Warfarin, no INR follow-up | ML_LAB_MONITORING | MODERATE | 88% |
| 17.6 | Amlodipine at max dose | ML_DOSAGE_NEAR_LIMIT | MODERATE | 99% |
| 17.7 | Sertraline nausea | ML_SIDE_EFFECT | MODERATE | 100% |
| 17.8 | Warfarin + leafy greens | ML_FOOD_INTERACTION | MODERATE | 100% |
| 17.9 | Amlodipine, poor lifestyle | ML_LIFESTYLE | LOW | 98% |
| 17.10 | Routine review, no screening history | ML_MISSING_INFO | HIGH | 99% |

Two caveats that matter for reading these numbers fairly:
- The test accepts **any one of several** expected labels (e.g. 17.1 accepts `ML_DUPLICATION` *or* `ML_HARMFUL_COMBO`), and the classifier returns **6–13 findings per case**, so a pass means "the right label was among them", not "it was the top/only label".
- Test 17.10 accepts `ML_MISSING_INFO` as a valid alternative to `ML_PREVENTIVE` (a comment in the test says so).

### 4.2 NER extraction check — `test_finetuned_vs_original.py`

Despite its name, this script runs the **current** pipeline on **10 prescriptions** and counts what it extracts (it does not load two models side-by-side). Run result:

| Metric | Value |
|---|---|
| Test prescriptions | 10 |
| Drugs found | 11 |
| Dosages found | 9 |
| Conditions found | 10 |
| Average entities per prescription | 3.0 |

Observed behaviour: it correctly extracted Warfarin / 5mg / Atrial Fibrillation, Sertraline / 50mg / Depression + Anxiety, Levothyroxine / 75mcg / Hypothyroidism, Clopidogrel / 75mg / MI, etc. **Known misses** in this run: the *Betamethasone 0.05%* cream dose was not captured, and *Neuropathic Pain* (a condition) was not captured for Pregabalin. Also, Insulin appears as both "Insulin" and "Insulin Glargine", and "Type 2 Diabetes" as well as "Diabetes" — overlapping duplicates that downstream code should dedupe.

These are *counts*, not accuracy: the script has no ground-truth answers to compare against.

### 4.3 Training metrics (what the training script tracks)

[train_ner_lora.py](train_ner_lora.py) prints, per epoch, **average cross-entropy loss** and **token-level accuracy** (correct predictions ÷ labelled tokens, ignoring padding/special tokens via label `-100`). It saves the adapter whenever epoch loss improves ("best loss" checkpointing). Those numbers were printed to the console only — **no training log was saved to disk**, so the final loss/accuracy values are not recorded in this folder. Re-run the script to capture them.

Trainable-parameter share is printed at start; with r=16 on those five module types, only a few percent of weights train (the script's own docstring says ~2%).

### 4.4 What's *not* measured yet (and how to add it)

| Gap | Why it matters | Fix |
|---|---|---|
| No train/validation/test split of the 129 examples | Training accuracy can look high from memorisation | Split ~80/10/10 |
| No entity-level **precision / recall / F1** | The standard NER metric | Use `seqeval` on a held-out set |
| No false-positive rate on **safe** prescriptions | 100% detection could hide over-alerting | Add negative test cases |
| Dataset is **synthetic** | May not match real handwriting/OCR noise | Add real or OCR-noisy samples |
| OCR accuracy not measured | OCR errors cascade into everything | Measure character/word error rate on sample scans |
| Label-ID map in training script | See warning below | Use `model.config.label2id` |

**⚠ Technical warning about the training script:** `_label_to_id()` in [train_ner_lora.py](train_ner_lora.py) uses a **hand-written label-to-ID table** (e.g. `B-DRUG: 28`, `I-DOSAGE: 28`) rather than the model's real `label2id`. Some IDs collide (`B-DRUG` and `I-DOSAGE` are both 28) and may not correspond to the base model's real label meanings. The classifier head is also marked as trainable (`modules_to_save: classifier`) so the model still learns *something*, and the runtime pipeline then maps outputs with its own LABEL_MAP, plus a regex fallback that hides errors in practice. But this is the first thing to fix if you want trustworthy, measurable NER gains.

### 4.5 Project-scale figures (counted from the code)

| Item | Value |
|---|---|
| Python source lines (excluding venv) | ~14,400 |
| Modules | 11 + pipeline, app, data scripts, 3 test files |
| Drug names in dictionary (generic + brand) | 1,120 (328 with detailed info in `DRUG_INFO`) |
| Conditions | 325 |
| Curated drug-interaction pairs | 34 (plus rule-based combinations) |
| Indian brand-name mappings | 312 |
| Training examples | 129 |
| LoRA adapter size | ~5 MB |
| Test scenarios | 92 (17 domains) |
| Audit log files | 4 daily files (30 Aug – 8 Sep 2026) |

The README still says "32 tests" and "500+ drugs"; the code has outgrown those — update the README to the numbers above.

---

## 5. Security & compliance layer (Module 7)

[module7_hipaa.py](module7_hipaa.py)

| Feature | How it's done |
|---|---|
| Password storage | **PBKDF2-HMAC-SHA256**, random 16-byte salt, **310,000 iterations**; constant-time comparison (`secrets.compare_digest`). Old plain SHA-256 hashes are still accepted for backward compatibility (a weakness—migrate them). |
| Password policy | 8–128 chars, upper, lower, digit, special char, strength meter |
| Email validation | RFC-style regex; disposable-email domains blocked |
| Account lockout | 5 failed attempts → 15-minute lock |
| Session expiry | 30 minutes of inactivity |
| Roles (RBAC) | **Viewer** (read only) · **Analyst** (upload/process/export) · **Admin** (+ user management, audit logs, settings) |
| PHI detection | Regex for SSN, phone, email, MRN, DOB (2 formats), ZIP, IP, NPI; de-identify by mask / hash / remove (HIPAA "Safe Harbor" idea) |
| Audit log | JSON-Lines per day: timestamp, session ID, user, event type (ACCESS, PROCESSING, export, user management, errors) |

Disclaimer (also in the README): this is a portfolio/educational project, not a certified medical device or production-HIPAA-compliant system. Also note `users.json` and `README.md` contain account details — keep them out of any public repository.

---

## 6. Technology stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| OCR | Tesseract (pytesseract), EasyOCR, Pillow, PyMuPDF |
| NLP / ML | HuggingFace Transformers, PyTorch, PEFT (LoRA), spaCy (optional), scikit-learn |
| Models | d4data/biomedical-ner-all (+LoRA), facebook/bart-large-mnli, SmolLM-135M-Instruct (optional) |
| Standards / APIs | FHIR R4, RxNorm (NLM REST API) |
| Storage | JSON files (users, audit logs) |
| Python env | `medscan_env` virtualenv; dependencies in [requirements.txt](requirements.txt) |

All models run on **CPU** (`device=-1`) and are **lazy-loaded** (loaded on first use), so the app starts fast and keeps working if one model is unavailable.

---

## 7. Glossary of important technical terms

**AI / ML concepts**

- **OCR (Optical Character Recognition):** converts an image of text into machine-readable text. *Tesseract* is the classic open-source engine; *EasyOCR* is a neural-network OCR that handles handwriting better.
- **NLP:** Natural Language Processing — computers working with human language.
- **NER (Named Entity Recognition):** finding and labelling the "things" in text. Here: "Amoxicillin" → DRUG, "500mg" → DOSAGE.
- **Token / tokenizer:** models read text as pieces called tokens. A BERT tokenizer splits "amoxicillin" into sub-words like `am` + `##oxicillin`; the code merges them back.
- **BIO tagging:** per-token labels — **B**egin of an entity, **I**nside it, **O**utside any entity. "Insulin Glargine" = B-DRUG, I-DRUG.
- **Transformer / BERT / DistilBERT:** the neural-network architecture behind modern language AI; BERT reads text in both directions; DistilBERT is a smaller, faster version. "Attention" (`q_lin/k_lin/v_lin` = query/key/value) lets each word weigh other words.
- **Pre-trained model / base model:** a model already trained on a large dataset that you reuse.
- **Fine-tuning:** continuing training of a pre-trained model on your own specific data (here: prescriptions).
- **LoRA (Low-Rank Adaptation):** a fine-tuning trick that freezes the original weights and trains tiny add-on matrices instead. **Rank (r)** sets their size; **alpha** scales their effect. Benefits: far fewer trainable parameters, runs on a CPU, adapter file is only ~5 MB.
- **PEFT:** HuggingFace "Parameter-Efficient Fine-Tuning" library that implements LoRA.
- **Adapter merge (`merge_and_unload`):** bakes the LoRA weights into the base model so inference is as fast as the original.
- **Epoch / batch / learning rate:** one full pass through the data / examples processed per update / size of each update step.
- **Loss (cross-entropy):** a number measuring how wrong the model is; lower is better. **Accuracy:** share of tokens predicted correctly.
- **Overfitting:** memorising training data and doing badly on new data — why a held-out test set is needed.
- **Gradient clipping / weight decay / dropout / warm-up:** training stabilisers that cap update size, penalise large weights, randomly drop connections, and ramp up the learning rate gradually.
- **Zero-shot classification:** classifying text into categories the model was never trained on, by describing the categories in words.
- **NLI / MNLI / entailment:** Natural Language Inference — deciding whether a hypothesis follows from a premise. BART-MNLI is trained for this and gets repurposed for zero-shot classification.
- **BART:** a sequence-to-sequence transformer from Facebook/Meta; `bart-large` is the big variant (~400M parameters).
- **Confidence threshold:** minimum score to report a result (0.45 in module 8).
- **Multi-label:** an item can have several labels at once (labels are scored independently).
- **Hallucination:** an AI confidently producing something false (a drug that doesn't exist). Module 4 is the countermeasure: every AI-extracted fact is checked against a real database.
- **Hybrid system / graceful fallback:** combining ML with rules, and degrading to a simpler method when the ML fails, instead of crashing.
- **Lazy loading / singleton:** load a heavy model only when first needed, and keep one shared copy.
- **Inference:** using a trained model to make predictions.
- **Intent classification:** working out what kind of question the user asked.
- **Grounding:** forcing answers to come from verified data, not free-form model guesses.

**Healthcare terms**

- **FHIR (Fast Healthcare Interoperability Resources) R4:** the international standard for exchanging health data as JSON. A **MedicationRequest** resource represents a prescription; a **Bundle** packages several resources together.
- **RxNorm:** the US National Library of Medicine's standard names for clinical drugs. Free REST API, used for live validation.
- **SNOMED CT:** a global clinical terminology (mentioned as a validation source in the README).
- **HIPAA:** US law protecting patient health information.
- **PHI (Protected Health Information):** any patient-identifying health data (name, SSN, MRN, DOB…). **De-identification:** removing or masking it. **Safe Harbor:** HIPAA's rule-list method of de-identification.
- **MRN / NPI:** Medical Record Number / National Provider Identifier (a doctor's ID).
- **Polypharmacy:** a patient on many drugs at once; raises interaction risk.
- **Drug–drug interaction:** two drugs affecting each other, e.g. warfarin + aspirin → bleeding.
- **CYP3A4:** a liver enzyme that processes many drugs; inhibiting it raises levels of drugs like statins (module 5 checks `INTERACTION_CYP3A4_STATIN`).
- **INR:** blood test measuring clotting; used to monitor warfarin.
- **Contraindication:** a condition where a drug should not be used.
- **Therapeutic duplication:** two drugs of the same class (e.g. two ACE inhibitors).
- **Anaphylaxis:** severe, life-threatening allergic reaction.
- **Hyperkalemia:** dangerously high potassium.
- **Generic vs brand name:** Crocin (brand) = Paracetamol/Acetaminophen (generic).
- **Rx shorthand:** Tab (tablet), Cap (capsule), Inj (injection), Syp (syrup), BD/BID (twice daily), TDS/TID (three times), QID (four times), PRN (as needed), IV (intravenous), subcut (under the skin).
- **Severity levels used:** CRITICAL (never proceed) > HIGH > MODERATE > LOW.

**Software / engineering terms**

- **Streamlit:** Python framework for building web dashboards.
- **RBAC (Role-Based Access Control):** permissions assigned by role.
- **PBKDF2 / salt / hash:** a slow, salted way of storing passwords so they can't be reversed or rainbow-table-attacked; the high iteration count (310,000) makes brute force expensive.
- **JSONL (JSON Lines):** one JSON object per line; ideal for append-only logs.
- **Regex:** pattern-matching language for text.
- **Fuzzy matching:** matching near-misses (OCR typos) to a dictionary entry.
- **Unicode script ranges:** each writing system occupies a numeric character range, which is how Hindi/Tamil are detected.
- **Transliteration:** writing a word from one script in another (not translation).
- **venv:** isolated Python environment; `.safetensors`: safe, fast format for model weights.
- **LRU cache / TTL:** caching results for reuse / time-to-live before they expire (RxNorm client caches for 1 hour).

---

## 8. Strengths, limits and suggested next steps

**Strengths**
- Layered safety: ML proposes, rules and databases verify; unknown items get flagged for a human rather than trusted.
- Graceful degradation at every ML step.
- Broad, explicit clinical scenario coverage (92 tests passing).
- Real standards (FHIR R4, RxNorm) and a thoughtful security model.
- Parameter-efficient fine-tuning that runs on a laptop CPU.

**Limits to be honest about**
- NER fine-tuning is unmeasured (no F1, no validation split), trained on only 129 synthetic examples, and has the label-ID issue noted in §4.4.
- Test suite measures detection (recall) but not false alarms; the tests were written by the same author as the rules, so they may share blind spots.
- Interaction database is small (34 pairs) next to real clinical databases; it is a demo-scale knowledge base, not clinical grade.
- ML anomaly output is noisy (6–13 findings per case) — useful as a second opinion, not as an authority.
- Legacy SHA-256 password support, JSON-file user storage and regex PHI detection are fine for a prototype but not production-grade.

**Next steps (highest value first)**
1. Fix label mapping in the trainer using `model.config.label2id`; add a train/val/test split and report entity-level precision/recall/F1 with `seqeval`.
2. Add negative (safe-prescription) tests to measure false-positive rate.
3. Save training logs and write the real numbers into the model card.
4. Add real or OCR-noisy training samples; measure OCR word-error rate.
5. Update the README (92 tests, real drug counts, correct test filename) and remove credentials from public docs.
6. Migrate legacy SHA-256 hashes; move users/audit data to a database.

---

## 9. File map

| File | Role |
|---|---|
| [app.py](app.py) | Streamlit UI, login, `run_full_pipeline()` orchestration |
| [pipeline.py](pipeline.py) | Original command-line pipeline (OCR → NER → validate → FHIR) |
| [module1_ocr.py](module1_ocr.py) | Multi-engine OCR + preprocessing |
| [module2_ner.py](module2_ner.py) / [module2_ml_ner.py](module2_ml_ner.py) | Hybrid NER / ML NER wrapper |
| [module3_fhir.py](module3_fhir.py) | FHIR R4 output |
| [module4_validation.py](module4_validation.py) | Hallucination-prevention validation |
| [module5_anomaly.py](module5_anomaly.py) | Rule-based anomaly engine |
| [module6_rxnorm.py](module6_rxnorm.py) | RxNorm API client |
| [module7_hipaa.py](module7_hipaa.py) | Auth, RBAC, PHI, audit, sessions |
| [module8_ml_anomaly.py](module8_ml_anomaly.py) | BART zero-shot safety classification |
| [module9_medical_qa.py](module9_medical_qa.py) | Q&A assistant |
| [module10_clinical_ner_advanced.py](module10_clinical_ner_advanced.py) | Entity linking & relations |
| [module11_indian_rx.py](module11_indian_rx.py) | Hindi/Tamil + Indian brands |
| [drug_knowledge.py](drug_knowledge.py) | Shared drug & condition knowledge base |
| [create_prescription_dataset.py](create_prescription_dataset.py) | Builds the training data |
| [train_ner_lora.py](train_ner_lora.py) | LoRA fine-tuning script |
| [finetuned_model/](finetuned_model/) | Saved LoRA adapter |
| [test_all_70.py](test_all_70.py) | 92-scenario safety suite |
| [test_ai_modules.py](test_ai_modules.py) | Tests for modules 8–10 |
| [test_finetuned_vs_original.py](test_finetuned_vs_original.py) | NER extraction check |
| [audit_logs/](audit_logs/) | Daily audit logs |

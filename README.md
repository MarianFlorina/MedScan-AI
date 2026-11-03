<<<<<<< HEAD
# 🏥 MedScan-AI

**Clinical Document Intelligence Platform** — A professional-grade healthcare AI system that extracts, validates, and analyzes medical information from clinical documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![FHIR](https://img.shields.io/badge/FHIR-R4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔍 Multi-Engine OCR** | Tesseract + EasyOCR with handwriting support |
| **🏷️ Medical NER** | LoRA-fine-tuned biomedical BERT + dictionary (1,120 drug names, 325 conditions) |
| **⚠️ Anomaly Detection** | Drug interactions, allergy conflicts, dosage anomalies |
| **🔬 RxNorm Validation** | Live API validation against NLM drug database |
| **📋 FHIR Output** | Hospital-standard JSON (MedicationRequest, Bundle) |
| **🛡️ HIPAA Compliance** | PHI detection, audit logging, role-based access |
| **🔐 Authentication** | 3 roles (Admin/Analyst/Viewer) with email validation |
| **📄 Multi-Page PDF** | Process all pages of clinical documents |
| **📊 Export** | FHIR JSON, CSV reports, clipboard copy |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** — [Install](https://github.com/UB-Mannheim/tesseract/wiki)
   - Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/medscan-ai.git
cd medscan-ai

# Create virtual environment
python -m venv medscan_env
medscan_env\Scripts\activate    # Windows
# source medscan_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔐 Accounts & Roles

No default accounts ship with the project (and none should be published). Register the
first user from the login page; the first admin can then create others.

| Role | Permissions |
|------|-------------|
| 👑 **Admin** | Full access + user management + audit logs |
| 🔬 **Analyst** | Upload, process, export documents |
| 👁️ **Viewer** | View results only |

`users.json` and `audit_logs/` hold real account/audit data — they are git-ignored; never commit them.

---

## 🏗️ Architecture

```
medscan-ai/
├── app.py                       # Streamlit dashboard, auth & orchestration
├── module1_ocr.py               # Multi-engine OCR (Tesseract + EasyOCR)
├── module2_ner.py / module2_ml_ner.py   # Hybrid NER: fine-tuned BERT (LoRA) + regex/dictionary
├── module3_fhir.py              # FHIR R4 output
├── module4_validation.py        # Hallucination prevention (knowledge-base check)
├── module5_anomaly.py           # Rule-based safety engine
├── module6_rxnorm.py            # RxNorm API integration
├── module7_hipaa.py             # Auth, RBAC, PHI detection, audit log
├── module8_ml_anomaly.py        # BART-MNLI zero-shot safety classifier
├── module9_medical_qa.py        # Q&A assistant
├── module10_clinical_ner_advanced.py, module11_indian_rx.py
├── build_ner_dataset.py         # Builds train/val/test NER data (exact offsets)
├── train_ner_lora.py            # LoRA fine-tuning (validation-F1 model selection)
├── evaluate_ner.py              # Held-out precision/recall/F1, 5 systems compared
├── ner_common.py                # Label mapping, span decoding, metrics
├── test_all_70.py               # 92 safety scenarios (recall)
├── test_false_positives.py      # 40 safe prescriptions (false-alarm rate)
├── test_app_flow.py             # headless UI/logic tests (31 checks)
├── test_ocr_pipeline.py         # OCR → NER → FHIR on real images/PDF
├── test_regressions.py          # unit tests for bugs fixed in v5
├── ui_theme.py                  # stylesheet + HTML helpers
├── results/                     # ner_eval.md/json, false_positive_eval.json
└── training_logs/               # per-epoch loss / validation F1
```

### Pipeline Flow

```
Upload → [1] OCR → [2] NER (ML + regex) → [4] Validation (+RxNorm) →
[5] Rule-based anomalies + [8] ML anomalies + [10] context → [3] FHIR → Dashboard / Export
```

---

## 📈 Results

### NER (entity-level, strict exact-span match, held-out data)

Fine-tuning uses the base model's own 84-label space; test drugs and conditions never appear in training.

| System | Hand-annotated gold (41 lines) F1 | Unseen-template synthetic (200 lines) F1 |
|---|---|---|
| Base `biomedical-ner-all` | 0.569 | 0.489 |
| Regex/dictionary only | 0.739 | 0.696 |
| **Fine-tuned LoRA** | **0.980** | **0.922** |
| Hybrid (production path) | 0.955 | 0.908 |

Full per-class tables: [results/ner_eval.md](results/ner_eval.md). Reproduce with
`python build_ner_dataset.py && python train_ner_lora.py && python evaluate_ner.py`.

### OCR + extraction on sample prescriptions (`python evaluate_ocr_samples.py`)

10 prescriptions (printed, handwritten, stamped; English, Hindi, Tamil; paediatric, cardiac, dermatology, nephrology)
cropped from a collage (each only ~300x500 px, upscaled 3x, so this is a harsh test). Ground truth transcribed by hand.

| Metric | Result |
|---|---|
| Drug recall (all 10 prescriptions, 37 drugs) | 35/37 = **95 %** (English 30/31) |
| Dose correctly attached to its drug | 28/30 = **93 %** |
| Required hazards flagged | Warfarin + Aspirin interaction; isotretinoin in a 24-year-old woman |
| Serious false alerts | 0 on the 8 English/Tamil sheets without a real hazard; 2 "incomplete" alerts remain on the Hindi sheet and one with a split dose table |

Known limits: very small/blurry handwriting (e.g. "Fantoprazole AO mg"), tables that OCR splits into columns
(doses separated from their drugs), Hindi OCR (amoxicillin dose and levocetirizine missed), and paediatric
weight-based dosing (syrup volumes are not converted to mg/kg).
Hindi/Tamil use the language packs in `tessdata/`; choose the document language on the Upload page.

### Safety checks

| Test | Result |
|---|---|
| `test_all_70.py` — 92 hazard scenarios, 17 domains (recall) | 92 / 92 detected |
| `test_false_positives.py` — 40 standard safe prescriptions | 0 raised CRITICAL/HIGH (0.0 % false-positive rate) |

### Honest limitations
- The gold set is small (41 lines, 43 drugs) and written by the project author; treat scores as indicative, not as clinical validation.
- Training data is synthetic template text. Drug/condition recognition on unseen names (synthetic test: DRUG F1 0.82, CONDITION 0.75) is weaker than on dosage/frequency, which is why the dictionary layer is kept.
- The hybrid path scores slightly below the fine-tuned model alone (regex merging adds some dosage errors) — a known item to tune.
- The safe-prescription set is also author-written; the 0 % rate shows no over-alerting on common cases, not on all cases.
- Interaction knowledge base is small (34 curated pairs) — a demo-scale, not clinical-grade, database.

---

## 🧪 Testing

```bash
python test_regressions.py       # 34 unit tests for bugs found in review
python test_app_flow.py          # 31 headless UI checks: roles, permissions, persistence, escaping, timeout
python test_ocr_pipeline.py      # image + scanned-PDF → entities → alerts → FHIR (needs Tesseract)
python evaluate_ocr_samples.py   # accuracy on 10 sample prescriptions (sample_prescriptions/)
python test_false_positives.py   # 40 safe prescriptions must not raise CRITICAL/HIGH
python test_all_70.py            # 92 hazard scenarios (loads BERT + BART, several minutes on CPU)
python evaluate_ner.py           # NER precision / recall / F1 on held-out sets
```

`test_app_flow.py` and `test_ocr_pipeline.py` use a temporary user database and audit folder
(`MEDSCAN_USERS_FILE`, `MEDSCAN_AUDIT_DIR`), so your real `users.json` is never touched.

---

## 🛡️ Security Features

| Feature | Implementation |
|---------|---------------|
| **Password Policy** | 8+ chars, uppercase, lowercase, digit, special char |
| **Account Lockout** | 5 failed attempts → 15-minute lockout |
| **Session Expiry** | 30-minute inactivity timeout |
| **PHI De-identification** | Mask/hash/remove SSN, phone, email, MRN |
| **Audit Logging** | All actions logged with timestamp and user |
| **Role-Based Access** | Viewer/Analyst/Admin with granular permissions |
| **Disposable Email Block** | Blocks tempmail, mailinator, etc. |

---

## 📊 Demo Scenarios

The app includes 6 pre-built clinical scenarios:

1. **Simple Prescription** — Amoxicillin for pneumonia + allergy check
2. **Drug Interaction Warning** — Warfarin + Aspirin (CRITICAL)
3. **High Dosage Warning** — Acetaminophen overdose risk
4. **Complex Polypharmacy** — 4 medications with interactions
5. **Multiple Conditions** — Diabetes + Hypertension + CKD
6. **Antibiotic + Allergy** — Penicillin allergy conflict

---

## 🚢 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Set `PYTHON_VERSION = 3.11` in advanced settings

### Docker

```dockerfile
FROM python:3.11-slim

# Install Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t medscan-ai .
docker run -p 8501:8501 medscan-ai
```

---

## 📈 Project Stats

| Metric | Value |
|--------|-------|
| Drug Database | 1,120 names (generic + brand), 328 with dose data |
| Condition Database | 325 medical conditions |
| Drug Interactions | 34 curated pairs + rule-based combinations |
| Tests | 92 hazard scenarios + 40 safe-prescription checks + 31 UI checks + 21 regression checks |
| FHIR Resources | MedicationRequest, Bundle |
| Auth Roles | Viewer, Analyst, Admin |
| PHI Types Detected | SSN, Phone, Email, MRN, DOB, ZIP, NPI |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This is a **portfolio/educational project**. It is NOT:
- A certified medical device
- HIPAA-compliant for production use
- A substitute for professional medical judgment

For production deployment, additional compliance measures are required including BAA agreements, encryption at rest/transit, penetration testing, and regulatory approval.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ using Streamlit, spaCy, Tesseract, and FHIR R4**
=======
# MedScan-AI
>>>>>>> 04933562cf884bf0124e47d5a8e0caf242367158

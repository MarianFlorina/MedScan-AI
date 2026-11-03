"""
MedScan-AI Dashboard — v5
=========================
Clinical Document Intelligence Platform: OCR → NER → validation → safety checks → FHIR.
"""

import csv
import hashlib
import html
import io
import json
import os
from datetime import datetime
from typing import Dict

import pymupdf as fitz
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MedScan-AI · Clinical Document Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from drug_knowledge import CONDITION_NAMES, DRUG_NAMES
from module1_ocr import OCR_LANGUAGES, MultiEngineOCR, extract_text_from_pil_image
from module11_indian_rx import to_generic, transliterate_indic_drugs
from module2_ner import build_medications, extract_medical_entities, extract_relationships
from module3_fhir import create_fhir_bundle, create_fhir_medication_request, fhir_to_pretty_json
from module4_validation import validate_all_entities
from module5_anomaly import AnomalyDetector
from module6_rxnorm import get_rxnorm_client
from module7_hipaa import AccessControl, AuditLogger, SessionManager, UserDatabase, Validator
from module8_ml_anomaly import classify_clinical_safety
from module9_medical_qa import answer_clinical_question
from module10_clinical_ner_advanced import analyze_clinical_context
from ui_theme import THEME_CSS, feature_row, step_card

MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
SESSION_TIMEOUT_MIN = 30

PAGE_HOME = "🏠  Home"
PAGE_UPLOAD = "📄  Upload document"
PAGE_DEMO = "🧪  Demo mode"
PAGE_AI = "🤖  AI assistant"
PAGES = [PAGE_HOME, PAGE_UPLOAD, PAGE_DEMO, PAGE_AI]

esc = html.escape  # every dynamic string that reaches unsafe_allow_html goes through this

st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> MultiEngineOCR:
    """One OCR engine per server process (constructing it shells out to Tesseract)."""
    return MultiEngineOCR()


@st.cache_resource(show_spinner=False)
def warm_models() -> bool:
    """Load the NER model in a background thread once per server process, so the first
    analysis doesn't pay the ~30 s model-load cost."""
    import threading

    def _load():
        try:
            from module2_ml_ner import get_ml_ner
            get_ml_ner()._load_model()
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()
    return True


def go(page: str):
    """Button callback: navigate to a page."""
    st.session_state["nav"] = page


# ============================================================
# LOGIN PAGE
# ============================================================

def render_password_strength(password: str):
    if not password:
        return
    _, _, strength = Validator.validate_password(password)
    level = strength.get("level", "weak")
    color = {"weak": "#f87171", "medium": "#fbbf24", "strong": "#34d399"}.get(level, "#f87171")
    width = {"weak": "33%", "medium": "66%", "strong": "100%"}.get(level, "33%")
    st.markdown(
        f'<div style="margin:2px 0 8px 0"><div style="display:flex;justify-content:space-between">'
        f'<span style="color:#9ca3af;font-size:.74rem">Password strength</span>'
        f'<span style="color:{color};font-size:.74rem;font-weight:700">{level.title()}</span></div>'
        f'<div style="background:#1f2937;border-radius:3px;height:5px">'
        f'<div style="background:{color};width:{width};height:5px;border-radius:3px"></div></div></div>',
        unsafe_allow_html=True,
    )


def _sign_in(identifier: str, password: str):
    db = UserDatabase()
    username = identifier.strip()
    if "@" in username:  # allow sign-in with email or username in one box
        user_obj = db.get_user_by_email(username)
        username = user_obj["username"] if user_obj else username
    user, error = db.authenticate(username, password)
    if user:
        st.session_state.authenticated = True
        st.session_state.user_info = user
        st.session_state.session_start = datetime.utcnow().isoformat()
        st.session_state.nav = PAGE_HOME
        SessionManager.touch()
        AuditLogger().log_access(user["username"], "LOGIN")
        st.rerun()
    else:
        AuditLogger().log_error("LOGIN_FAILED", f"Failed sign-in for '{username[:40]}'")
        st.error(error)


def render_login_page():
    flash = st.session_state.pop("flash", None)
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            '<div class="login-hero">'
            '<div class="brand"><div class="brand-mark">🏥</div><div>'
            '<div class="brand-name">MedScan-AI</div><div class="brand-sub">Clinical Document Intelligence</div></div></div>'
            '<h2>From prescription to <span>verified, safety-checked data</span> in seconds.</h2>'
            '<div class="login-lead">Upload a prescription or paste clinical text. MedScan reads it, extracts '
            'every drug, dose and condition, checks it for dangerous interactions, allergies and overdoses, '
            'and exports hospital-standard FHIR.</div>'
            + feature_row("🔍", "Reads real documents", "Multi-engine OCR for scans, photos and multi-page PDFs, including handwriting mode.")
            + feature_row("🧠", "Fine-tuned clinical NER", "A LoRA-tuned biomedical model backed by a drug &amp; condition knowledge base.")
            + feature_row("🛡️", "Layered safety checks", "Rule engine + ML classifier flag interactions, allergy conflicts and dosing risks.")
            + feature_row("📋", "FHIR R4 export", "Standards-based MedicationRequest bundles and CSV reports, with an audit trail.")
            + '<div class="login-foot">Educational demonstration — do not upload real patient data.</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown('<span class="login-card-marker"></span>', unsafe_allow_html=True)
            if flash:
                st.warning(flash)
            tab_login, tab_register = st.tabs(["Sign in", "Create account"])

            with tab_login:
                st.markdown('<div class="login-title">Welcome back</div>'
                            '<div class="login-sub">Sign in with your username or email.</div>', unsafe_allow_html=True)
                with st.form("login_form", clear_on_submit=False):
                    identifier = st.text_input("Username or email", placeholder="e.g. jsmith or john@hospital.org")
                    password = st.text_input("Password", type="password", placeholder="Your password")
                    submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
                if submitted:
                    if not identifier.strip() or not password:
                        st.error("Please enter your username/email and password.")
                    else:
                        _sign_in(identifier, password)

                has_admin = any(u.get("role") == "admin" for u in UserDatabase().users.values())
                if not has_admin:
                    st.markdown('<div class="notice notice-warn"><strong>First time here?</strong> '
                                'Create an account — the first one becomes the <strong>Admin</strong>.</div>',
                                unsafe_allow_html=True)
                st.markdown('<div style="text-align:center;color:#6b7280;font-size:.75rem;margin-top:12px">'
                            'Forgot your password? Contact your administrator.</div>', unsafe_allow_html=True)

            with tab_register:
                st.markdown('<div class="login-title">Create your account</div>'
                            '<div class="login-sub">Takes less than a minute.</div>', unsafe_allow_html=True)
                reg_name = st.text_input("Full name", placeholder="Jane Smith", key="reg_name")
                reg_email = st.text_input("Email", placeholder="jane@hospital.org", key="reg_email")
                reg_user = st.text_input("Username", placeholder="jsmith", key="reg_user")
                reg_pass = st.text_input("Password", type="password", placeholder="8+ chars, upper, lower, digit, symbol", key="reg_pass")
                render_password_strength(reg_pass)

                if reg_email:
                    ok_email, err = Validator.validate_email(reg_email)
                    if not ok_email:
                        st.caption(f"⚠️ {err}")
                if reg_user:
                    ok_user, err = Validator.validate_username(reg_user)
                    if not ok_user:
                        st.caption(f"⚠️ {err}")
                    elif UserDatabase().get_user(reg_user):
                        st.caption("⚠️ That username is already taken")

                reg_role = st.selectbox(
                    "Role", options=["viewer", "analyst"], key="reg_role",
                    format_func=lambda x: {"viewer": "Viewer — explore the demo, read-only",
                                           "analyst": "Analyst — upload, process and export"}[x],
                )
                if st.button("Create account", type="primary", use_container_width=True, key="reg_btn"):
                    if not all([reg_name.strip(), reg_email.strip(), reg_user.strip(), reg_pass]):
                        st.error("Please fill in all fields.")
                    else:
                        db = UserDatabase()
                        has_admin = any(u.get("role") == "admin" for u in db.users.values())
                        actual_role = "admin" if not has_admin else reg_role
                        ok, msg = db.register_user(username=reg_user, email=reg_email, password=reg_pass,
                                                   full_name=reg_name, role=actual_role)
                        if ok:
                            st.success(msg + (" You were made **Admin** (first account)." if actual_role != reg_role else ""))
                            st.info("Switch to the **Sign in** tab to continue.")
                        else:
                            st.error(msg)


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list:
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    pdf_document.close()
    return images


def run_full_pipeline(image=None, text=None, pdf_bytes=None, is_handwritten=False,
                      allergies=None, conditions=None, use_rxnorm=False, use_ml=True, progress=None, lang="eng"):
    """Full analysis. `progress(msg)` is called before each stage so the UI can show live status;
    `use_ml` toggles the slow BART safety scan."""
    _p = progress or (lambda msg: None)
    results = {
        "raw_text": "", "ocr_confidence": 0, "ocr_engine": "",
        "entities": [], "relationships": [], "validated_entities": [],
        "anomalies": None, "fhir_output": None, "fhir_bundle": None,
        "validation_summary": {"total": 0, "valid": 0, "flagged": 0},
        "pages_processed": 1, "rxnorm_validations": [],
    }

    if pdf_bytes is not None:
        _p("Rendering PDF pages…")
        images = pdf_to_images(pdf_bytes, dpi=200)
        results["pages_processed"] = len(images)
        all_text, total_conf = [], 0
        best_engine = ""
        for i, img in enumerate(images):
            _p(f"Reading text (OCR) — page {i + 1} of {len(images)}…")
            r = extract_text_from_pil_image(img, is_handwritten, lang)
            all_text.append(f"--- Page {i + 1} ---\n{r['text']}")
            total_conf += r["confidence"]
            best_engine = r["engine"]
        results["raw_text"] = "\n\n".join(all_text)
        results["ocr_confidence"] = total_conf / len(images) if images else 0
        results["ocr_engine"] = best_engine
    elif image is not None:
        _p("Reading text (OCR)…")
        r = extract_text_from_pil_image(image, is_handwritten, lang)
        results["raw_text"] = r["text"]
        results["ocr_confidence"] = r["confidence"]
        results["ocr_engine"] = r["engine"]
    elif text:
        results["raw_text"] = text
        results["ocr_confidence"] = 1.0
        results["ocr_engine"] = "manual"
    else:
        return results

    if lang != "eng":
        results["raw_text_original"] = results["raw_text"]
        results["raw_text"] = transliterate_indic_drugs(results["raw_text"])
    _p("Extracting drugs, doses and conditions (first run loads the model, ~30 s)…")
    entities = extract_medical_entities(results["raw_text"])
    results["entities"] = entities
    relationships = extract_relationships(entities, results["raw_text"])
    results["relationships"] = relationships

    _p("Validating against the knowledge base…")
    validated = validate_all_entities(entities)
    results["validated_entities"] = validated
    valid_count = sum(1 for e in validated if e["validated"])
    results["validation_summary"] = {
        "total": len(validated), "valid": valid_count,
        "flagged": len(validated) - valid_count,
    }

    if use_rxnorm:
        # Opt-in: needs internet. If the API is unreachable we stop immediately
        # instead of waiting for a timeout on every drug.
        try:
            client = get_rxnorm_client()
            if client.rxcui_by_name("aspirin") is None:
                results["rxnorm_validations"].append({"drug": "-", "status": "unreachable", "valid": False})
            else:
                for drug in [e["word"] for e in entities if e["entity_group"] == "DRUG"][:8]:
                    results["rxnorm_validations"].append({"drug": drug, **client.validate_drug(drug)})
        except Exception:
            results["rxnorm_validations"] = [{"drug": "-", "status": "unreachable", "valid": False}]

    # One entry per drug, brand names mapped to generics, each dose matched to its own drug by position
    medications = build_medications(entities, results["raw_text"], relationships)
    for r in relationships:  # FHIR and the rule engine work with generic names
        r["brand"] = r["drug"]
        r["drug"] = to_generic(r["drug"])
        if r.get("drug_pair"):
            r["drug_pair"] = [to_generic(d) for d in r["drug_pair"]]

    _p("Running safety rules (interactions, allergies, dosing)…")
    detector = AnomalyDetector()
    results["anomalies"] = detector.analyze_prescription(
        medications, allergies=allergies or [], conditions=conditions or [],
        text=results["raw_text"]
    )

    if relationships:
        # Filter out co_prescribed relationships (they don't have drug/condition/dosage)
        rx_relationships = [r for r in relationships if r.get("relationship_type") != "co_prescribed"]
        if rx_relationships:
            results["fhir_bundle"] = create_fhir_bundle(rx_relationships)
            results["fhir_output"] = create_fhir_medication_request(
                drug=rx_relationships[0]["drug"], dosage=rx_relationships[0]["dosage"],
                condition=rx_relationships[0]["condition"],
            )
        elif relationships:
            # Fallback: use co_prescribed relationships with dummy condition
            results["fhir_bundle"] = create_fhir_bundle(relationships)
            results["fhir_output"] = create_fhir_medication_request(
                drug=relationships[0].get("drug", relationships[0].get("drug_pair", ["Unknown"])[0]),
                dosage=relationships[0].get("dosage", "N/A"),
                condition=relationships[0].get("condition", "Unknown"),
            )

    # ── AI Module: ML Anomaly Classification ──
    results["ml_anomalies"] = []
    if use_ml:
        _p("Running ML safety scan (BART, slow on CPU)…")
        try:
            results["ml_anomalies"] = classify_clinical_safety(text=results["raw_text"], medications=medications)
        except Exception:
            results["ml_anomalies"] = []

    # ── AI Module: Clinical Context Analysis ──
    try:
        clinical_context = analyze_clinical_context(
            text=results["raw_text"],
            entities=entities,
            medications=medications,
        )
        results["clinical_context"] = clinical_context
    except Exception:
        results["clinical_context"] = {}

    _p("Finishing up…")
    # Keep what the AI assistant needs (medications WITH dosages, allergies, conditions)
    results["medications"] = medications
    results["allergies"] = allergies or []
    results["conditions"] = conditions or []
    st.session_state["last_results"] = results

    AuditLogger().log_processing("doc", len(entities),
        results["anomalies"]["total_anomalies"] if results["anomalies"] else 0)

    return results


def render_entity_badge(entity_group):
    css_class = {
        "DRUG": "drug", "MEDICATION": "drug", "DOSAGE": "dosage", "FREQUENCY": "frequency",
        "DURATION": "duration", "ROUTE": "route", "CONDITION": "condition", "DISEASE_DISORDER": "condition",
        "HISTORY": "history", "PROCEDURE": "procedure",
    }.get(entity_group, "history")
    return f'<span class="badge badge-{css_class}">{esc(str(entity_group))}</span>'


def _generate_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Entity", "Type", "Confidence", "Validated", "Reason"])
    for ent in results.get("validated_entities", []):
        writer.writerow([ent["word"], ent["entity_group"], f'{ent["score"]:.2f}',
            "Yes" if ent["validated"] else "No", ent.get("reason", "")])
    if results.get("relationships"):
        writer.writerow([])
        writer.writerow(["Drug", "Dosage", "Condition", "Relationship Type"])
        for rel in results["relationships"]:
            writer.writerow([rel["drug"], rel["dosage"], rel["condition"],
                rel.get("relationship_type", "prescribed_for")])
    if results.get("anomalies") and results["anomalies"]["findings"]:
        writer.writerow([])
        writer.writerow(["Anomaly Type", "Severity", "Description", "Recommendation"])
        for f in results["anomalies"]["findings"]:
            writer.writerow([f["type"], f["severity"], f["description"], f["recommendation"]])
    return output.getvalue()


def _render_entity_table(entities):
    """Render entities as a clean table."""
    if not entities:
        st.info("No entities found.")
        return

    # Group by type
    by_type = {}
    for ent in entities:
        t = ent["entity_group"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(ent)

    for etype, ents in by_type.items():
        badge = render_entity_badge(etype)
        st.markdown(f'<div style="margin-bottom:16px">{badge} <span style="color:#8b949e;font-size:0.78rem">({len(ents)})</span></div>', unsafe_allow_html=True)

        rows = ""
        for ent in ents:
            conf_pct = f'{ent["score"]:.0%}'
            conf_color = "#3fb950" if ent["score"] >= 0.85 else "#d29922" if ent["score"] >= 0.7 else "#f85149"
            status = "✅" if ent.get("validated", True) else "⚠️"
            rows += f"""
            <tr>
                <td style="font-weight:500">{esc(str(ent['word']))}</td>
                <td><span style="color:{conf_color};font-weight:600">{conf_pct}</span></td>
                <td>{status}</td>
            </tr>"""

        st.markdown(
            f'<table class="entity-table">'
            f'<thead><tr><th>Entity</th><th>Confidence</th><th>Status</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )


def _render_relationships(relationships):
    """Drug → dose → condition cards; co-prescribed pairs only when they are real interactions."""
    shown = 0
    for rel in relationships:
        if rel.get("relationship_type") == "co_prescribed":
            if not rel.get("potential_interaction"):
                continue  # benign pairs are noise
            a, b = rel["drug_pair"]
            st.markdown(
                f'<div class="rel-card" style="border-left:3px solid #f87171"><span style="font-size:1.2rem">⚡</span>'
                f'<div><span class="rel-drug">{esc(str(a))}</span> + <span class="rel-drug">{esc(str(b))}</span>'
                f' <span class="rel-type">— known interacting combination</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            rel_type = {"causal": "treats", "prescribed_for": "prescribed for"}.get(
                rel.get("relationship_type", ""), str(rel.get("relationship_type", "")).replace("_", " "))
            st.markdown(
                f'<div class="rel-card"><span style="font-size:1.2rem">💊</span><div>'
                f'<span class="rel-drug">{esc(str(rel["drug"]))}</span>'
                f' <span class="rel-dosage">({esc(str(rel.get("dosage", "N/A")))})</span>'
                f'<span class="rel-arrow"> → </span><span class="rel-type">{esc(rel_type)}</span>'
                f'<span class="rel-arrow"> → </span><span class="rel-condition">{esc(str(rel.get("condition", "N/A")))}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        shown += 1
    if not shown:
        st.info("No drug–condition relationships detected.")


def _render_anomalies(anomalies):
    """Render anomalies as severity cards."""
    if not anomalies or anomalies["total_anomalies"] == 0:
        st.markdown(
            '<div style="text-align:center;padding:20px;color:#3fb950">'
            '<div style="font-size:1.5rem;margin-bottom:6px">✅</div>'
            '<div style="font-weight:600;font-size:0.9rem">No Anomalies Detected</div>'
            '<div style="color:#8b949e;font-size:0.82rem;margin-top:4px">Prescription appears safe</div>'
            '</div>', unsafe_allow_html=True,
        )
        return

    sc = anomalies["severity_counts"]
    # Severity summary badges
    badges_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">'
    if sc["CRITICAL"] > 0:
        badges_html += f'<span class="badge" style="background:#f8514922;color:#f85149;border:1px solid #f8514944">🚨 {sc["CRITICAL"]} Critical</span>'
    if sc["HIGH"] > 0:
        badges_html += f'<span class="badge" style="background:#d2992222;color:#d29922;border:1px solid #d2992244">⚠️ {sc["HIGH"]} High</span>'
    if sc["MODERATE"] > 0:
        badges_html += f'<span class="badge" style="background:#388bfd22;color:#388bfd;border:1px solid #388bfd44">ℹ️ {sc["MODERATE"]} Moderate</span>'
    if sc["LOW"] > 0:
        badges_html += f'<span class="badge" style="background:#3fb95022;color:#3fb950;border:1px solid #3fb95044">💡 {sc["LOW"]} Low</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

    for finding in anomalies["findings"]:
        severity = finding.get("severity", "LOW")
        icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "ℹ️", "LOW": "💡"}.get(severity, "💡")
        st.markdown(
            f'<div class="severity-card severity-{severity.lower()}">'
            f'<div class="severity-label">{icon} {severity} — {esc(finding.get("type", "").replace("_", " ").title())}</div>'
            f'<div class="severity-desc">{esc(str(finding.get("description", "")))}</div>'
            f'<div class="severity-rec">💡 {esc(str(finding.get("recommendation", "")))}</div>'
            f'</div>', unsafe_allow_html=True,
        )



def _render_results_upload(results, pdf_page_count=0):
    summary = results["validation_summary"]
    anomaly_count = results["anomalies"]["total_anomalies"] if results["anomalies"] else 0
    has_critical = results["anomalies"]["has_critical"] if results["anomalies"] else False

    # ── Metrics Row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Entities", summary["total"])
    m2.metric("Validated", summary["valid"])
    m3.metric("Flagged", summary["flagged"])
    m4.metric("Anomalies", anomaly_count, delta="critical" if has_critical else None,
              delta_color="inverse" if has_critical else "off")
    m5.metric("Pages", pdf_page_count if pdf_page_count > 0 else 1)

    st.markdown('<div style="margin:16px 0;border-top:1px solid #21262d"></div>', unsafe_allow_html=True)

    # ── OCR Info ──
    conf = results["ocr_confidence"]
    conf_color = "#3fb950" if conf > 0.8 else "#d29922" if conf > 0.5 else "#f85149"
    st.markdown(
        f'<div style="display:flex;gap:16px;align-items:center;margin-bottom:16px">'
        f'<span style="color:#8b949e;font-size:0.82rem">OCR Confidence:</span>'
        f'<span style="color:{conf_color};font-weight:600;font-size:0.88rem">{conf:.0%}</span>'
        f'<span style="color:#484f58;font-size:0.78rem">•</span>'
        f'<span style="color:#8b949e;font-size:0.82rem">Engine:</span>'
        f'<span style="color:#e6edf3;font-weight:500;font-size:0.88rem">{esc(str(results["ocr_engine"]))}</span>'
        f'</div>', unsafe_allow_html=True,
    )

    # ── Extracted Text ──
    with st.expander("📝 Extracted Text", expanded=False):
        st.code(results["raw_text"], language=None)

    # ── Entities ──
    with st.expander("🏷️ Extracted Entities", expanded=True):
        _render_entity_table(results["validated_entities"])

    # ── Relationships ──
    if results["relationships"]:
        with st.expander("🔗 Drug-Condition Relationships", expanded=True):
            _render_relationships(results["relationships"])

    _render_rxnorm(results)

    # ── Anomaly Detection ──
    with st.expander("⚠️ Anomaly Detection", expanded=True):
        _render_anomalies(results["anomalies"])

    # ── ML Anomaly Overlay ──
    ml_findings = results.get("ml_anomalies", [])
    if ml_findings:
        with st.expander("🧠 ML-Powered Anomaly Analysis", expanded=False):
            _render_ml_anomalies(ml_findings)

    # ── Clinical Context ──
    clinical_context = results.get("clinical_context", {})
    if clinical_context:
        _render_clinical_context(clinical_context)

    # ── Validation Summary ──
    if summary["flagged"] == 0:
        st.success(f"✅ All **{summary['total']}** entities validated successfully.")
    else:
        st.warning(f"✅ {summary['valid']}/{summary['total']} validated · **{summary['flagged']}** flagged")

    st.markdown('<div style="margin:16px 0;border-top:1px solid #21262d"></div>', unsafe_allow_html=True)

    # ── Export ──
    _render_export(results)


def _render_results_demo(results):
    # ── Metrics Row ──
    summary = results["validation_summary"]
    anomaly_count = results["anomalies"]["total_anomalies"] if results["anomalies"] else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entities", summary["total"])
    m2.metric("Validated", summary["valid"])
    m3.metric("Anomalies", anomaly_count)
    m4.metric("Relationships", len(results["relationships"]))

    st.markdown('<div style="margin:12px 0;border-top:1px solid #21262d"></div>', unsafe_allow_html=True)

    # ── Two Column Layout ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Input Text
        st.markdown("#### 📝 Input Text")
        st.info(results["raw_text"])

        # Entities
        st.markdown("#### 🏷️ Extracted Entities")
        _render_entity_table(results["validated_entities"])

        # Relationships
        if results["relationships"]:
            st.markdown("#### 🔗 Relationships")
            _render_relationships(results["relationships"])

    with col_right:
        # Anomalies
        st.markdown("#### ⚠️ Anomaly Detection")
        _render_anomalies(results["anomalies"])

        # ML Anomaly Overlay
        ml_findings = results.get("ml_anomalies", [])
        if ml_findings:
                _render_ml_anomalies(ml_findings)

        # Clinical Context
        clinical_context = results.get("clinical_context", {})
        if clinical_context:
            _render_clinical_context(clinical_context)

        # Validation
        st.markdown("#### ✅ Validation")
        if summary["flagged"] == 0:
            st.success(f"All **{summary['total']}** entities validated.")
        else:
            st.warning(f"{summary['valid']}/{summary['total']} validated · {summary['flagged']} flagged")

        st.markdown('<div style="margin:12px 0;border-top:1px solid #21262d"></div>', unsafe_allow_html=True)

        # Export
        _render_export(results)



def _render_ml_anomalies(ml_findings: list):
    """Render ML-based anomaly findings as supplementary cards."""
    if not ml_findings:
        return

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1f6feb44;border-radius:8px;'
        'padding:12px 16px;margin:8px 0">'
        '<span style="color:#58a6ff;font-weight:600;font-size:0.85rem">'
        '🧠 ML-Powered Anomaly Analysis</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Show top 8 findings (by severity, then confidence) to keep UI clean
    shown = ml_findings[:8]
    if len(ml_findings) > 8:
        st.caption(f"Showing top {len(shown)} of {len(ml_findings)} ML findings")
    for f in shown:
        severity = f.get("severity", "LOW")
        icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "ℹ️", "LOW": "💡"}.get(severity, "💡")
        conf = f.get("ml_confidence", 0)
        st.markdown(
            f'<div class="severity-card severity-{severity.lower()}" '
            f'style="border-left-color:{"#f85149" if severity == "CRITICAL" else "#d29922" if severity == "HIGH" else "#388bfd" if severity == "MODERATE" else "#3fb950"}">'
            f'<div class="severity-label">{icon} {severity} — {esc(f.get("type", "").replace("ML_", "").replace("_", " ").title())}'
            f' <span style="color:#8b949e;font-size:0.75rem">(ML confidence: {conf:.0%})</span></div>'
            f'<div class="severity-desc">{esc(str(f.get("description", "")))}</div>'
            f'<div class="severity-rec">💡 {esc(str(f.get("recommendation", "")))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_clinical_context(context: Dict):
    """Render clinical context analysis."""
    if not context:
        return

    with st.expander("🔬 Clinical Context Analysis", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Normalized Drugs", context.get("drug_count", 0))
        c2.metric("Conditions", context.get("condition_count", 0))
        c3.metric("Allergies", context.get("allergy_count", 0))

        normalized = context.get("normalized_drugs", {})
        if normalized:
            st.markdown("**Drug Name Mapping:**")
            for original, normalized_name in normalized.items():
                if original != normalized_name:
                    st.markdown(f"  • {esc(str(original))} → **{esc(str(normalized_name))}**")

        treatment = context.get("treatment_relationships", [])
        if treatment:
            st.markdown("**Treatment Relationships:**")
            for r in treatment:
                st.markdown(
                    f'  <span style="color:#3fb950">✓</span> '
                    f'<span style="color:#58a6ff">{esc(str(r["drug"]))}</span> '
                    f'treats <span style="color:#d29922">{esc(str(r["condition"]))}</span> '
                    f'<span style="color:#8b949e">(confidence: {r["confidence"]:.0%})</span>',
                    unsafe_allow_html=True,
                )

        contra = context.get("contraindication_relationships", [])
        if contra:
            st.markdown("**Contraindications:**")
            for c in contra:
                target = c.get("condition", c.get("allergy", "N/A"))
                st.markdown(
                    f'  <span style="color:#f85149">✗</span> '
                    f'<span style="color:#58a6ff">{esc(str(c["drug"]))}</span> '
                    f'contraindicated with <span style="color:#f85149">{esc(str(target))}</span>',
                    unsafe_allow_html=True,
                )



def _render_export(results):
    """Download buttons (permission-aware) + FHIR viewer."""
    if not results.get("fhir_bundle"):
        return
    fhir_json = fhir_to_pretty_json(results["fhir_bundle"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    can_export = AccessControl.check_permission("export")

    c1, c2 = st.columns(2)
    with c1:
        if can_export:
            st.download_button("📥 FHIR bundle (JSON)", data=fhir_json, file_name=f"medscan_fhir_{stamp}.json",
                               mime="application/json", use_container_width=True,
                               on_click=lambda: AuditLogger().log_export("fhir_json"))
        else:
            st.button("🔒 FHIR bundle (export needs Analyst/Admin)", disabled=True, use_container_width=True)
    with c2:
        if can_export:
            st.download_button("📊 CSV report", data=_generate_csv(results), file_name=f"medscan_report_{stamp}.csv",
                               mime="text/csv", use_container_width=True,
                               on_click=lambda: AuditLogger().log_export("csv"))
        else:
            st.button("🔒 CSV report (export needs Analyst/Admin)", disabled=True, use_container_width=True)
    with st.expander("View FHIR JSON"):
        st.code(fhir_json, language="json")  # st.code has a built-in copy button


def _render_rxnorm(results):
    rx = results.get("rxnorm_validations") or []
    if not rx:
        return
    with st.expander("🌐 Live RxNorm check", expanded=False):
        if rx[0].get("status") == "unreachable":
            st.warning("RxNorm (NLM) could not be reached — check your internet connection. Local validation still applies.")
            return
        for r in rx:
            if r.get("valid"):
                st.markdown(f"✅ **{esc(str(r['drug']))}** → {esc(str(r.get('normalized_name') or ''))} (RxCUI {esc(str(r.get('rxcui')))})")
            else:
                st.markdown(f"⚠️ **{esc(str(r['drug']))}** not found in RxNorm")


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(user_info):
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">🏥</div><div><div class="brand-name">MedScan-AI</div>'
            '<div class="brand-sub">Clinical Document Intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        role_info = AccessControl.get_role_info(user_info["role"])
        st.markdown(
            f'<div class="sidebar-user"><div style="display:flex;align-items:center;gap:10px">'
            f'<div style="width:36px;height:36px;border-radius:50%;background:{role_info.get("color", "#38bdf8")}22;'
            f'display:grid;place-items:center;font-size:1.05rem">{role_info.get("icon", "👤")}</div>'
            f'<div><div style="color:#f3f4f6;font-weight:650;font-size:.9rem">{esc(user_info["full_name"])}</div>'
            f'<div style="color:{role_info.get("color", "#9ca3af")};font-size:.76rem">{esc(role_info.get("name", user_info["role"]))}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Navigate</div>', unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

        if AccessControl.check_permission("view_audit"):
            st.markdown('<div class="sidebar-label">Administration</div>', unsafe_allow_html=True)
            with st.expander("📊 Audit log", expanded=False):
                audit = AuditLogger()
                summary = audit.get_log_summary()
                c1, c2 = st.columns(2)
                c1.metric("Events", summary["total_events"])
                c2.metric("Sessions", summary["unique_sessions"])
                for ev in audit.get_logs(limit=8)[::-1]:
                    st.caption(f'{ev["timestamp"][5:16].replace("T", " ")} · {ev["event_type"]} · {ev.get("username") or "-"}')
        if AccessControl.check_permission("manage_users"):
            with st.expander("👥 Users", expanded=False):
                db = UserDatabase()
                for u in db.get_all_users():
                    status = "🟢" if u["is_active"] else "🔴"
                    st.markdown(f'{status} **{esc(u["username"])}** · {esc(u["role"])}<br/>'
                                f'<span style="color:#6b7280;font-size:.72rem">{esc(u["email"])}</span>',
                                unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Add user**")
                with st.form("admin_add_user", clear_on_submit=True):
                    n_name = st.text_input("Full name")
                    n_email = st.text_input("Email")
                    n_user = st.text_input("Username")
                    n_pass = st.text_input("Temporary password", type="password")
                    n_role = st.selectbox("Role", ["viewer", "analyst", "admin"], key="admin_new_role")
                    if st.form_submit_button("Create user", use_container_width=True):
                        ok, msg = db.register_user(n_user, n_email, n_pass, n_name, n_role)
                        if ok:
                            AuditLogger().log_user_management(f"CREATE_{n_role.upper()}", n_user.lower().strip())
                            st.success(f"{msg} ({n_role})")
                        else:
                            st.error(msg)
                st.markdown("**Change role**")
                names = [u["username"] for u in db.get_all_users()]
                c_user = st.selectbox("User", names, key="admin_role_user")
                c_role = st.selectbox("New role", ["viewer", "analyst", "admin"], key="admin_role_new")
                if st.button("Update role", use_container_width=True, key="admin_role_btn"):
                    ok, msg = db.set_role(c_user, c_role)
                    if ok:
                        AuditLogger().log_user_management(f"ROLE_{c_role.upper()}", c_user)
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="sidebar_signout"):
            AuditLogger().log_access(user_info["username"], "LOGOUT")
            SessionManager.logout()
            st.rerun()
        st.markdown(
            f'<div class="sidebar-foot">Auto sign-out after {SESSION_TIMEOUT_MIN} min idle<br/>'
            'MedScan-AI v5 · FHIR R4</div>', unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

def _record_history(results: Dict, source: str):
    hist = st.session_state.setdefault("history", [])
    an = results.get("anomalies") or {}
    sev = an.get("severity_counts", {})
    snippet = " ".join((results.get("raw_text") or "").split())[:70]
    hist.insert(0, {
        "time": datetime.now().strftime("%H:%M"), "source": source,
        "entities": len(results.get("entities", [])),
        "alerts": an.get("total_anomalies", 0),
        "critical": sev.get("CRITICAL", 0), "snippet": snippet,
    })
    del hist[20:]


def render_home(user_info):
    first = user_info["full_name"].split()[0]
    perms = user_info.get("permissions", {})
    role_name = AccessControl.get_role_info(user_info["role"]).get("name", user_info["role"])
    st.markdown(
        f'<div class="hero"><h1>Welcome back, {esc(first)} 👋</h1>'
        f'<p>Turn prescriptions into structured, safety-checked data. Upload a scan, or try the demo '
        f'to see the full pipeline on sample clinical text.</p>'
        f'<div style="margin-top:14px;position:relative;z-index:1">'
        f'<span class="pill">🔑 {esc(role_name)}</span>'
        f'<span class="pill">📅 {datetime.now().strftime("%d %b %Y")}</span></div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.button("📄 Analyze a document", type="primary", use_container_width=True, on_click=go, args=(PAGE_UPLOAD,),
              disabled=not perms.get("upload", False),
              help=None if perms.get("upload") else "Uploading needs the Analyst or Admin role")
    c2.button("🧪 Try the demo", use_container_width=True, on_click=go, args=(PAGE_DEMO,))
    c3.button("🤖 Ask the AI assistant", use_container_width=True, on_click=go, args=(PAGE_AI,))

    hist = st.session_state.get("history", [])
    st.markdown('<div class="section-title">This session</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Analyses run", len(hist))
    k2.metric("Entities extracted", sum(h["entities"] for h in hist))
    k3.metric("Safety alerts", sum(h["alerts"] for h in hist))
    k4.metric("Critical alerts", sum(h["critical"] for h in hist))

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("Read", "OCR with preprocessing reads scans, photos and PDFs."),
        ("Extract", "Fine-tuned NER finds drugs, doses, frequencies, conditions."),
        ("Verify", "Every item is checked against a drug &amp; condition knowledge base."),
        ("Safety-check", "Rules + ML flag interactions, allergies and dosing risks."),
        ("Export", "FHIR R4 bundle and CSV report."),
    ]
    for col, (i, (t, d)) in zip(st.columns(5), enumerate(steps, 1)):
        col.markdown(step_card(i, t, d), unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Recent analyses</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not hist:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">🩺</div>'
                            '<div class="empty-state-title">Nothing analysed yet</div>'
                            'Run the demo to see your first result here.</div>', unsafe_allow_html=True)
            else:
                rows = ""
                for h in hist[:6]:
                    tone = "pill-bad" if h["critical"] else ("pill-warn" if h["alerts"] else "pill-ok")
                    label = f'{h["critical"]} critical' if h["critical"] else (f'{h["alerts"]} alerts' if h["alerts"] else "clear")
                    rows += (f'<div class="recent-item"><div><div style="color:#e5e7eb">{esc(h["snippet"])}…</div>'
                             f'<div style="color:#6b7280;font-size:.75rem">{h["time"]} · {esc(h["source"])} · {h["entities"]} entities</div></div>'
                             f'<div><span class="pill {tone}">{label}</span></div></div>')
                st.markdown(rows, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">System status</div>', unsafe_allow_html=True)
        with st.container(border=True):
            ocr_ok = get_ocr_engine().tesseract_available
            tuned = os.path.exists("finetuned_model/adapter_config.json")
            rows = [
                ("OCR engine (Tesseract)", '<span class="pill pill-ok">ready</span>' if ocr_ok else '<span class="pill pill-warn">not installed</span>'),
                ("Clinical NER model", '<span class="pill pill-ok">fine-tuned (LoRA)</span>' if tuned else '<span class="pill pill-warn">base model</span>'),
                ("Knowledge base", f'<span class="pill">{len(DRUG_NAMES):,} drugs · {len(CONDITION_NAMES)} conditions</span>'),
                ("Your access", " ".join(f'<span class="pill pill-ok">{p}</span>' for p in ("upload", "process", "export") if perms.get(p))
                 or '<span class="pill">view only</span>'),
            ]
            st.markdown("".join(f'<div class="status-row"><span class="status-key">{k}</span><span>{v}</span></div>' for k, v in rows),
                        unsafe_allow_html=True)
            if not ocr_ok:
                st.caption("Install Tesseract to enable document upload; the demo works without it.")


# ============================================================
# UPLOAD PAGE
# ============================================================

def _locked(title: str, msg: str):
    st.markdown(f'<div class="lock-card"><div style="font-size:2rem">🔒</div>'
                f'<div class="empty-state-title">{esc(title)}</div><div style="color:#9ca3af">{esc(msg)}</div></div>',
                unsafe_allow_html=True)


def _split_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def render_upload_page():
    st.markdown("# Upload document")
    st.caption("Prescriptions, lab notes or discharge summaries — JPG, PNG or multi-page PDF (max 20 MB).")

    if not AccessControl.check_permission("upload"):
        _locked("Uploading needs the Analyst or Admin role",
                "Ask your administrator for access, or explore the pipeline in Demo mode.")
        st.button("Go to Demo mode", on_click=go, args=(PAGE_DEMO,))
        return

    ocr_ok = get_ocr_engine().tesseract_available
    if not ocr_ok:
        st.warning("**Tesseract OCR is not installed**, so scanned images can't be read. Install it from "
                   "https://github.com/UB-Mannheim/tesseract/wiki, or use **Demo mode** for typed text.")

    left, right = st.columns([2, 1], gap="large")
    with right:
        st.markdown("##### Patient context")
        allergies_in = st.text_area("Known allergies", placeholder="penicillin, sulfa", height=80, key="upl_allergy")
        conditions_in = st.text_area("Known conditions", placeholder="diabetes, kidney disease", height=80, key="upl_cond")
        language = st.selectbox("Document language", list(OCR_LANGUAGES), key="upl_lang",
                                help="Hindi and Tamil use extra OCR language packs; drug names are mapped to English generics.")
        is_handwritten = st.checkbox("📝 Handwriting mode", value=False)
        use_rx = st.checkbox("🌐 Live RxNorm check (needs internet)", value=False, key="upl_rx")
        use_ml = st.checkbox("🧠 Deep ML safety scan (adds 1–2 min)", value=False, key="upl_ml",
                             help="Zero-shot BART classifier. The rule-based safety checks always run.")
    with left:
        uploaded = st.file_uploader("Upload prescription", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")

    if not uploaded:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">📤</div>'
                    '<div class="empty-state-title">Drop a prescription to begin</div>'
                    'Results appear here with entities, safety alerts and FHIR export.</div>', unsafe_allow_html=True)
        return

    if uploaded.size > MAX_UPLOAD_SIZE_BYTES:
        st.error(f"❌ File too large ({uploaded.size / 1024 / 1024:.1f} MB). Maximum is {MAX_UPLOAD_SIZE_MB} MB.")
        return

    data = uploaded.getvalue()
    is_pdf = uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")
    page_count, preview = 0, None
    try:
        if is_pdf:
            pages = pdf_to_images(data, dpi=100)  # low-DPI preview only
            page_count, preview = len(pages), (pages[0] if pages else None)
        else:
            preview = Image.open(io.BytesIO(data))
    except Exception as e:
        st.error(f"Could not open this file: {e}")
        return
    with left:
        if preview is not None:
            st.image(preview, caption=f"{uploaded.name}" + (f" — page 1 of {page_count}" if is_pdf else ""), width="stretch")

    allergies, conditions = _split_csv(allergies_in), _split_csv(conditions_in)
    sig = (hashlib.md5(data).hexdigest(), is_handwritten, tuple(allergies), tuple(conditions), use_rx, use_ml, language)
    cache = st.session_state.get("upload_cache")
    is_current = cache is not None and cache["sig"] == sig

    if st.button("🔎 Analyze document", type="primary", use_container_width=True):
        AuditLogger().log_document_upload(uploaded.name, "pdf" if is_pdf else "image", len(data))
        with st.status("Analyzing document…", expanded=True) as status:
            try:
                kw = dict(is_handwritten=is_handwritten, allergies=allergies, conditions=conditions,
                          use_rxnorm=use_rx, use_ml=use_ml, progress=status.write,
                          lang=OCR_LANGUAGES[language])
                if is_pdf:
                    res = run_full_pipeline(pdf_bytes=data, **kw)
                else:
                    res = run_full_pipeline(image=Image.open(io.BytesIO(data)), **kw)
                st.session_state["upload_cache"] = {"sig": sig, "results": res, "pages": page_count}
                _record_history(res, f"upload · {uploaded.name[:24]}")
                cache = st.session_state["upload_cache"]
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as e:
                st.session_state.pop("upload_cache", None)
                cache = None
                status.update(label="Analysis failed", state="error")
                st.error(f"Analysis failed: {e}")

    if cache is not None and cache["sig"] == sig:
        res = cache["results"]
        if res.get("raw_text") and res["raw_text"].strip():
            _render_results_upload(res, cache["pages"])
        else:
            st.warning("No text could be extracted. Try a clearer, higher-resolution image (300+ DPI) "
                       "or enable handwriting mode.")
    elif cache is not None:
        st.info("Inputs changed since the last analysis — press **Analyze document** to refresh the results.")


# ============================================================
# DEMO PAGE
# ============================================================

DEMO_SCENARIOS = {
    "Simple prescription": "Patient prescribed 500mg of Amoxicillin for pneumonia. History of Type 2 Diabetes. Allergic to Penicillin.",
    "Drug interaction": "Patient prescribed Warfarin 5mg and Aspirin 81mg. Also taking Ibuprofen 400mg for pain.",
    "High dosage warning": "Patient prescribed Acetaminophen 2000mg every 4 hours for severe pain.",
    "Polypharmacy": "Patient diagnosed with hypertension. Prescribed Lisinopril 20mg daily, Metformin 500mg for Type 2 Diabetes, Atorvastatin 40mg for cholesterol, and Omeprazole 20mg for GERD.",
    "Multiple conditions": "Patient has Diabetes, Hypertension, and Chronic Kidney Disease. Prescribed Metformin 1000mg, Lisinopril 20mg, and Aspirin 81mg.",
    "Antibiotic + allergy": "Patient prescribed Amoxicillin 500mg three times daily. Patient has known penicillin allergy.",
}


def render_demo_page():
    st.markdown("# Demo mode")
    st.caption("Run the full pipeline on sample clinical text — no document or OCR needed.")

    c1, c2, c3 = st.columns([1.3, 1, 1])
    selected = c1.selectbox("Scenario", list(DEMO_SCENARIOS.keys()))
    allergies = _split_csv(c2.text_input("Allergies", value="penicillin", key="demo_allergy"))
    conditions = _split_csv(c3.text_input("Conditions", value="diabetes, hypertension", key="demo_cond"))
    text = st.text_area("Clinical text", value=DEMO_SCENARIOS[selected], height=110, key=f"demo_text_{selected}")
    demo_ml = st.checkbox("🧠 Include ML safety scan (a few extra seconds)", value=True, key="demo_ml")

    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter some clinical text first.")
        else:
            with st.status("Running pipeline…", expanded=True) as status:
                res = run_full_pipeline(text=text, allergies=allergies, conditions=conditions,
                                        use_ml=demo_ml, progress=status.write)
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["demo_results"] = res
            _record_history(res, f"demo · {selected}")

    res = st.session_state.get("demo_results")
    if res:
        _render_results_demo(res)
    else:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">🧪</div>'
                    '<div class="empty-state-title">Pick a scenario and run the analysis</div>'
                    'Try “Drug interaction” to see a critical alert.</div>', unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

QUICK_QUESTIONS = [
    ("💊", "What medications are prescribed?"), ("⚠️", "Are there any interactions?"),
    ("✅", "Is this safe?"), ("📋", "How should the patient take these?"),
    ("🔍", "What tests are needed?"), ("❗", "Any risks or warnings?"),
]


def _set_question(q: str):
    st.session_state["ai_q_input"] = q
    st.session_state["ai_submit"] = True


def _context_from_text(text: str) -> Dict:
    """Light analysis for pasted text (no OCR, no heavy ML)."""
    entities = extract_medical_entities(text)
    meds = [{"drug": e["word"], "dosage": ""} for e in entities if e["entity_group"] == "DRUG"]
    anomalies = AnomalyDetector().analyze_prescription(meds, text=text)
    return {"raw_text": text, "entities": entities, "medications": meds, "allergies": [], "anomalies": anomalies}


def render_ai_page():
    st.markdown("# AI assistant")
    st.caption("Ask about the medications, dosing, safety and patient instructions in an analysed prescription.")

    last = st.session_state.get("last_results")
    src = st.radio("Answer using", ["Last analysis", "Pasted text"], horizontal=True, key="ai_context_source")
    custom = ""
    if src == "Pasted text":
        custom = st.text_area("Prescription text", placeholder="Paste prescription text here…", height=90, key="ai_custom_context")
    elif last is None:
        st.info("No analysis yet — run **Demo mode** or **Upload document** first, or switch to *Pasted text*.")

    cols = st.columns(3)
    for i, (icon, q) in enumerate(QUICK_QUESTIONS):
        cols[i % 3].button(f"{icon} {q}", key=f"quick_q_{i}", use_container_width=True, on_click=_set_question, args=(q,))

    question = st.text_input("Your question", key="ai_q_input", placeholder="e.g. Can this patient take ibuprofen with these medications?")
    asked = st.button("Ask", type="primary", use_container_width=True, key="ai_ask_btn")

    if asked or st.session_state.pop("ai_submit", False):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            ctx = None
            if src == "Pasted text":
                if custom.strip():
                    ctx = _context_from_text(custom)
            elif last is not None:
                ctx = last
            if ctx is None:
                st.warning("There is no prescription text to answer from yet.")
            else:
                with st.spinner("Thinking…"):
                    ans = answer_clinical_question(
                        question=question, text=ctx.get("raw_text", ""),
                        medications=ctx.get("medications", []), entities=ctx.get("entities", []),
                        allergies=ctx.get("allergies", []), anomalies=ctx.get("anomalies"),
                        use_ml=False,  # grounded, deterministic answers only (see module9 _is_faithful)
                    )
                st.session_state["ai_last"] = {"q": question, "a": ans["answer"], "anomalies": ctx.get("anomalies")}

    shown = st.session_state.get("ai_last")
    if shown:
        with st.container(border=True):
            st.markdown(f"**Q:** {shown['q']}")
            st.markdown(shown["a"])  # markdown, not raw HTML: renders **bold** and lists correctly
        an = shown.get("anomalies")
        if an and an.get("total_anomalies", 0) > 0:
            with st.expander(f"Related safety findings ({an['total_anomalies']})", expanded=False):
                _render_anomalies(an)


# ============================================================
# MAIN
# ============================================================

def main():
    SessionManager.initialize()

    if not st.session_state.get("authenticated", False):
        render_login_page()
        return

    if SessionManager.is_expired(max_minutes=SESSION_TIMEOUT_MIN):
        AuditLogger().log_access(st.session_state.user_info["username"], "SESSION_EXPIRED")
        SessionManager.logout()
        st.session_state["flash"] = f"⏰ You were signed out after {SESSION_TIMEOUT_MIN} minutes of inactivity."
        st.rerun()
        return
    SessionManager.touch()
    warm_models()

    st.session_state.setdefault("nav", PAGE_HOME)
    user_info = st.session_state.user_info
    render_sidebar(user_info)

    st.markdown('<div class="hipaa-banner"><strong>⚠️ Educational demonstration.</strong> '
                'Do not upload real patient data. Processing runs locally on this machine.</div>', unsafe_allow_html=True)

    page = st.session_state.get("nav", PAGE_HOME)
    if page == PAGE_UPLOAD:
        render_upload_page()
    elif page == PAGE_DEMO:
        render_demo_page()
    elif page == PAGE_AI:
        render_ai_page()
    else:
        render_home(user_info)


main()

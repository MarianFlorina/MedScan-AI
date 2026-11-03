"""
Module 8: ML-Based Clinical Anomaly Classification
====================================================
Uses facebook/bart-large-mnli for zero-shot classification of clinical text
into safety categories. This augments the rule-based anomaly detection
(module5) with an AI-powered layer that can catch subtle or novel patterns.

How it works:
1. Takes clinical text + extracted medications
2. Zero-shot classifies into safety categories with confidence scores
3. Returns findings that supplement rule-based detection
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from typing import List, Dict, Optional, Tuple
import re


# ============================================================
# Clinical Safety Labels (for zero-shot classification)
# ============================================================

SAFETY_LABELS = {
    "CRITICAL": [
        "dangerous drug overdose potentially lethal",
        "severe allergic reaction anaphylaxis risk",
        "life-threatening drug interaction",
        "suicidal ideation medication safety concern",
        "intentional overdose poisoning emergency",
        "emergency medical situation requiring immediate intervention",
    ],
    "HIGH": [
        "significant drug interaction requiring monitoring",
        "dosage above recommended range",
        "contraindicated medication for patient condition",
        "therapeutic duplication same drug class",
        "missing critical safety information",
        "potentially harmful medication combination",
    ],
    "MODERATE": [
        "mild drug interaction worth monitoring",
        "dosage near upper limit of recommended range",
        "medication requiring periodic lab monitoring",
        "potential side effect warranting awareness",
        "drug food interaction possibility",
    ],
    "LOW": [
        "minor medication concern",
        "dosage adjustment may be beneficial",
        "lifestyle modification recommended alongside medication",
        "preventive health screening suggested",
    ],
}

# Map classified labels back to structured findings
LABEL_TO_FINDING = {
    "dangerous drug overdose potentially lethal": {
        "type": "ML_DOSAGE_EMERGENCY",
        "description": "ML model detected potential dangerous overdose pattern",
        "recommendation": "Verify dosage against standard guidelines immediately",
    },
    "severe allergic reaction anaphylaxis risk": {
        "type": "ML_ALLERGY_RISK",
        "description": "ML model detected potential severe allergic reaction risk",
        "recommendation": "Review allergy history and consider alternative medications",
    },
    "life-threatening drug interaction": {
        "type": "ML_DRUG_INTERACTION",
        "description": "ML model detected potentially life-threatening drug interaction",
        "recommendation": "Consult pharmacist or specialist before dispensing",
    },
    "suicidal ideation medication safety concern": {
        "type": "ML_MENTAL_HEALTH_RISK",
        "description": "ML model detected mental health safety concern in medication context",
        "recommendation": "Assess patient mental health status; ensure safety planning",
    },
    "intentional overdose poisoning emergency": {
        "type": "ML_OVERDOSE_EMERGENCY",
        "description": "ML model detected patterns consistent with overdose or poisoning",
        "recommendation": "Emergency evaluation required; contact Poison Control if needed",
    },
    "emergency medical situation requiring immediate intervention": {
        "type": "ML_EMERGENCY",
        "description": "ML model flagged an emergency-level clinical scenario",
        "recommendation": "Immediate clinical review required",
    },
    "significant drug interaction requiring monitoring": {
        "type": "ML_INTERACTION_MONITORING",
        "description": "ML model detected significant drug interaction requiring monitoring",
        "recommendation": "Set up monitoring schedule; educate patient on warning signs",
    },
    "dosage above recommended range": {
        "type": "ML_DOSAGE_HIGH",
        "description": "ML model detected dosage above recommended range",
        "recommendation": "Verify dosage with prescriber; consider therapeutic window",
    },
    "contraindicated medication for patient condition": {
        "type": "ML_CONTRAINDICATION",
        "description": "ML model detected potential contraindication for patient condition",
        "recommendation": "Review patient conditions against medication contraindications",
    },
    "therapeutic duplication same drug class": {
        "type": "ML_DUPLICATION",
        "description": "ML model detected possible therapeutic duplication",
        "recommendation": "Verify if dual therapy is intentional; avoid unnecessary duplication",
    },
    "missing critical safety information": {
        "type": "ML_MISSING_INFO",
        "description": "ML model detected missing critical safety information",
        "recommendation": "Clarify missing dosage, frequency, or duration before dispensing",
    },
    "potentially harmful medication combination": {
        "type": "ML_HARMFUL_COMBO",
        "description": "ML model detected potentially harmful medication combination",
        "recommendation": "Review combination for safety; consider alternatives",
    },
    "mild drug interaction worth monitoring": {
        "type": "ML_MILD_INTERACTION",
        "description": "ML model detected mild drug interaction",
        "recommendation": "Monitor for symptoms; typically safe to co-prescribe",
    },
    "dosage near upper limit of recommended range": {
        "type": "ML_DOSAGE_NEAR_LIMIT",
        "description": "ML model detected dosage near upper recommended limit",
        "recommendation": "Within safe range but monitor for dose-dependent side effects",
    },
    "medication requiring periodic lab monitoring": {
        "type": "ML_LAB_MONITORING",
        "description": "ML model identified medication requiring lab monitoring",
        "recommendation": "Schedule appropriate lab tests per monitoring guidelines",
    },
    "potential side effect warranting awareness": {
        "type": "ML_SIDE_EFFECT",
        "description": "ML model flagged potential side effect for awareness",
        "recommendation": "Inform patient of potential side effects and when to seek help",
    },
    "drug food interaction possibility": {
        "type": "ML_FOOD_INTERACTION",
        "description": "ML model detected potential drug-food interaction",
        "recommendation": "Advise patient on dietary considerations with medication",
    },
    "minor medication concern": {
        "type": "ML_MINOR_CONCERN",
        "description": "ML model detected minor medication concern",
        "recommendation": "No immediate action needed; note for follow-up",
    },
    "dosage adjustment may be beneficial": {
        "type": "ML_DOSAGE_OPTIMIZATION",
        "description": "ML model suggests dosage optimization may be beneficial",
        "recommendation": "Discuss dosage optimization with prescriber",
    },
    "lifestyle modification recommended alongside medication": {
        "type": "ML_LIFESTYLE",
        "description": "ML model suggests lifestyle modifications alongside medication",
        "recommendation": "Counsel patient on lifestyle changes for better outcomes",
    },
    "preventive health screening suggested": {
        "type": "ML_PREVENTIVE",
        "description": "ML model suggests preventive health screening",
        "recommendation": "Recommend age-appropriate health screenings",
    },
}


# ============================================================
# ML Anomaly Classifier (lazy-loaded singleton)
# ============================================================

_ml_classifier = None
_ml_available = None  # None = not checked yet
MAX_ML_CHARS = 1500
_result_cache = {}  # (text, threshold) -> findings; re-analysing the same text is instant


def _get_ml_classifier():
    from hf_lock import HF_LOCK
    with HF_LOCK:
        return _get_ml_classifier_unlocked()


def _get_ml_classifier_unlocked():
    """Lazy-load the zero-shot classification model."""
    global _ml_classifier, _ml_available
    
    if _ml_available is False:
        return None
    if _ml_classifier is not None:
        return _ml_classifier
    
    try:
        from transformers import pipeline as hf_pipeline
        
        model_name = "facebook/bart-large-mnli"
        
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        _ml_classifier = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU
        )
        _ml_available = True
        return _ml_classifier
        
    except Exception as e:
        _ml_available = False
        print(f"[ML-Anomaly] Model failed to load ({e}). ML anomaly detection disabled.")
        return None


def classify_clinical_safety(
    text: str,
    medications: List[Dict] = None,
    confidence_threshold: float = 0.45,
) -> List[Dict]:
    """
    Zero-shot classify clinical text into safety categories.
    
    Args:
        text: Clinical text to analyze
        medications: List of extracted medication dicts (optional, for context enrichment)
        confidence_threshold: Minimum confidence to report (0.0-1.0)
    
    Returns:
        List of finding dicts with type, severity, description, recommendation, ml_confidence
    """
    classifier = _get_ml_classifier()
    if classifier is None:
        return []  # Graceful fallback — rule-based detection still works
    
    findings = []
    
    try:
        # Enrich the input text with medication context
        enriched_text = text
        if medications:
            med_names = [m.get("drug", "") for m in medications if m.get("drug")]
            if med_names:
                enriched_text += f" Medications prescribed: {', '.join(med_names)}."
        
        # Zero-shot cost = (number of labels) x (input length) on CPU, so long OCR dumps are very slow.
        # Keep the first ~1500 characters (the prescription itself is almost always at the top).
        if len(enriched_text) > MAX_ML_CHARS:
            enriched_text = enriched_text[:MAX_ML_CHARS]

        cache_key = (enriched_text, round(confidence_threshold, 3))
        if cache_key in _result_cache:
            return [dict(f) for f in _result_cache[cache_key]]
        
        # Classify against CRITICAL labels first (highest priority)
        all_labels = []
        all_severities = []
        for severity in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            for label in SAFETY_LABELS[severity]:
                all_labels.append(label)
                all_severities.append(severity)
        
        result = classifier(
            enriched_text,
            candidate_labels=all_labels,
            multi_label=True,
        )
        
        # Process results
        labels = result["labels"]
        scores = result["scores"]
        
        seen_types = set()
        
        for label, score in zip(labels, scores):
            if score < confidence_threshold:
                continue
            
            finding_info = LABEL_TO_FINDING.get(label)
            if finding_info is None:
                continue
            
            finding_type = finding_info["type"]
            if finding_type in seen_types:
                continue
            seen_types.add(finding_type)
            
            # Determine severity from the label mapping
            severity = "LOW"
            for sev, sev_labels in SAFETY_LABELS.items():
                if label in sev_labels:
                    severity = sev
                    break
            
            findings.append({
                "type": finding_type,
                "severity": severity,
                "description": finding_info["description"],
                "recommendation": finding_info["recommendation"],
                "ml_confidence": round(score, 3),
                "source": "ml_zero_shot",
            })
        
        # Sort by severity (CRITICAL first) then confidence
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        findings.sort(key=lambda x: (severity_order.get(x["severity"], 9), -x["ml_confidence"]))
        
        _result_cache[cache_key] = [dict(f) for f in findings]
        if len(_result_cache) > 64:
            _result_cache.pop(next(iter(_result_cache)))
    except Exception as e:
        print(f"[ML-Anomaly] Classification failed: {e}")
    
    return findings


def get_model_status() -> Dict:
    """Get the status of the ML anomaly model."""
    classifier = _get_ml_classifier()
    return {
        "available": classifier is not None,
        "model": "facebook/bart-large-mnli" if classifier else None,
        "type": "zero-shot-classification",
        "description": "BART-MNLI: classifies clinical text into safety categories without task-specific training",
    }


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    test_cases = [
        "Patient prescribed Warfarin 10mg and Aspirin 81mg. Also taking Ibuprofen 400mg for pain.",
        "Patient prescribed Amoxicillin 500mg. History of anaphylaxis to penicillin.",
        "Patient taking Insulin but has repeated episodes of low blood sugar.",
        "Patient reports rash after starting a new medication but has no breathing difficulty.",
    ]
    
    print("=" * 70)
    print("ML Anomaly Classification Test")
    print("=" * 70)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {text[:80]}...")
        findings = classify_clinical_safety(text)
        if findings:
            for f in findings:
                print(f"  [{f['severity']:8s}] {f['type']:30s} (ML: {f['ml_confidence']:.1%}) — {f['description']}")
        else:
            print("  No ML findings (model may be loading or below threshold)")
    
    print(f"\n{'=' * 70}")
    print("Model Status:", get_model_status())

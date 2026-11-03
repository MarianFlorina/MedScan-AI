"""
Module 5: Medical Anomaly Detection
=====================================
Detects potential issues in prescriptions:
- Drug-drug interactions
- Drug-allergy conflicts
- Dosage anomalies
- Duplicate medications
- Drug-condition contraindications
- Age-related warnings
"""

import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


import functools as _functools


@_functools.lru_cache(maxsize=None)
def _short_kw_regex(kw: str):
    return re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"s?(?![a-z0-9])")


def _kw(kw: str, text_lower: str) -> bool:
    """Keyword test used by the rule engine.

    Short keywords (<= 4 letters) must match as whole words, otherwise 'aki' fires inside
    'taking', 'ill' inside 'pill', 'kid' inside 'kidney', 'ate' inside 'rate' ...
    Longer keywords keep substring semantics because many are deliberate stems
    (e.g. 'convuls', 'hypotens').
    """
    k = kw.strip()
    if len(k) > 4 or not k:
        return kw in text_lower
    return _short_kw_regex(k).search(text_lower) is not None



# ============================================================
# DRUG INTERACTION DATABASE
# ============================================================

# Severity levels: CRITICAL, HIGH, MODERATE, LOW
DRUG_INTERACTIONS = {
    # CRITICAL - Never combine
    ("warfarin", "aspirin"): {
        "severity": "CRITICAL",
        "description": "Increased risk of bleeding",
        "recommendation": "Avoid combination or monitor closely"
    },
    ("warfarin", "ibuprofen"): {
        "severity": "CRITICAL",
        "description": "Increased risk of GI bleeding",
        "recommendation": "Use acetaminophen instead of NSAIDs"
    },
    ("warfarin", "naproxen"): {
        "severity": "CRITICAL",
        "description": "Increased risk of bleeding",
        "recommendation": "Avoid combination"
    },
    ("metformin", "alcohol"): {
        "severity": "CRITICAL",
        "description": "Risk of lactic acidosis",
        "recommendation": "Limit alcohol intake"
    },
    ("methotrexate", "ibuprofen"): {
        "severity": "CRITICAL",
        "description": "Increased methotrexate toxicity",
        "recommendation": "Avoid NSAIDs with methotrexate"
    },
    ("lithium", "ibuprofen"): {
        "severity": "CRITICAL",
        "description": "Increased lithium levels",
        "recommendation": "Avoid NSAIDs or monitor lithium levels"
    },
    ("maoi", "ssri"): {
        "severity": "CRITICAL",
        "description": "Serotonin syndrome risk",
        "recommendation": "Wait 14 days between MAOI and SSRI"
    },
    
    # HIGH - Significant risk
    ("lisinopril", "potassium"): {
        "severity": "HIGH",
        "description": "Risk of hyperkalemia",
        "recommendation": "Monitor potassium levels"
    },
    ("spironolactone", "potassium"): {
        "severity": "HIGH",
        "description": "Risk of hyperkalemia",
        "recommendation": "Monitor potassium levels"
    },
    ("digoxin", "amiodarone"): {
        "severity": "HIGH",
        "description": "Increased digoxin levels",
        "recommendation": "Reduce digoxin dose"
    },
    ("simvastatin", "amiodarone"): {
        "severity": "HIGH",
        "description": "Increased risk of rhabdomyolysis",
        "recommendation": "Limit simvastatin to 20mg"
    },
    ("metformin", "contrast_dye"): {
        "severity": "HIGH",
        "description": "Risk of lactic acidosis with contrast",
        "recommendation": "Stop metformin 48h before contrast"
    },
    ("ciprofloxacin", "theophylline"): {
        "severity": "HIGH",
        "description": "Increased theophylline levels",
        "recommendation": "Monitor theophylline or use alternative antibiotic"
    },
    ("ciprofloxacin", "antacids"): {
        "severity": "HIGH",
        "description": "Reduced ciprofloxacin absorption",
        "recommendation": "Take cipro 2h before or 6h after antacids"
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "HIGH",
        "description": "Reduced clopidogrel effectiveness",
        "recommendation": "Use pantoprazole instead"
    },
    
    # MODERATE - Monitor use
    ("metoprolol", "verapamil"): {
        "severity": "MODERATE",
        "description": "Risk of bradycardia",
        "recommendation": "Monitor heart rate"
    },
    ("lisinopril", "nsaid"): {
        "severity": "MODERATE",
        "description": "Reduced antihypertensive effect",
        "recommendation": "Monitor blood pressure"
    },
    ("warfarin", "acetaminophen"): {
        "severity": "MODERATE",
        "description": "Possible increased INR",
        "recommendation": "Monitor INR if using high doses"
    },
    ("metformin", "diuretics"): {
        "severity": "MODERATE",
        "description": "Risk of dehydration",
        "recommendation": "Monitor kidney function"
    },
    ("ssri", "tramadol"): {
        "severity": "MODERATE",
        "description": "Increased risk of serotonin syndrome",
        "recommendation": "Use lowest effective doses"
    },
    ("benzodiazepine", "opioid"): {
        "severity": "MODERATE",
        "description": "Risk of respiratory depression",
        "recommendation": "Avoid or use lowest doses"
    },
    ("calcium_channel_blocker", "grapefruit"): {
        "severity": "MODERATE",
        "description": "Increased drug levels",
        "recommendation": "Avoid grapefruit juice"
    },
    ("statin", "grapefruit"): {
        "severity": "MODERATE",
        "description": "Increased statin levels",
        "recommendation": "Limit grapefruit intake"
    },
    
    # === ADDITIONAL CRITICAL INTERACTIONS ===
    ("methotrexate", "nsaid"): {
        "severity": "CRITICAL",
        "description": "Increased methotrexate toxicity",
        "recommendation": "Avoid NSAIDs during methotrexate therapy"
    },
    ("ssri", "maoi"): {
        "severity": "CRITICAL",
        "description": "Serotonin syndrome risk",
        "recommendation": "Wait 14 days between SSRI and MAOI"
    },
    ("ssri", "tramadol"): {
        "severity": "HIGH",
        "description": "Increased risk of serotonin syndrome",
        "recommendation": "Monitor for serotonin syndrome symptoms"
    },
    
    # === ADDITIONAL HIGH INTERACTIONS ===
    ("ace_inhibitor", "arb"): {
        "severity": "HIGH",
        "description": "Dual RAAS blockade increases kidney failure risk",
        "recommendation": "Do not combine ACE inhibitors with ARBs"
    },
    ("potassium_sparing_diuretic", "ace_inhibitor"): {
        "severity": "HIGH",
        "description": "Risk of severe hyperkalemia",
        "recommendation": "Monitor potassium levels closely"
    },
    ("digoxin", "amiodarone"): {
        "severity": "HIGH",
        "description": "Increased digoxin levels by 70-100%",
        "recommendation": "Reduce digoxin dose by 50%"
    },
    ("carbamazepine", "macrolide_antibiotic"): {
        "severity": "HIGH",
        "description": "Increased carbamazepine toxicity",
        "recommendation": "Monitor carbamazepine levels"
    },
    ("theophylline", "fluoroquinolone"): {
        "severity": "HIGH",
        "description": "Increased theophylline levels",
        "recommendation": "Avoid combination or monitor levels"
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "HIGH",
        "description": "Reduced clopidogrel activation",
        "recommendation": "Use pantoprazole instead"
    },
    
    # === ADDITIONAL MODERATE INTERACTIONS ===
    ("levothyroxine", "calcium"): {
        "severity": "MODERATE",
        "description": "Reduced levothyroxine absorption",
        "recommendation": "Take levothyroxine 4 hours before calcium"
    },
    ("levothyroxine", "iron"): {
        "severity": "MODERATE",
        "description": "Reduced levothyroxine absorption",
        "recommendation": "Take levothyroxine 4 hours before iron"
    },
    ("metformin", "alcohol"): {
        "severity": "CRITICAL",
        "description": "Risk of lactic acidosis",
        "recommendation": "Limit alcohol intake"
    },
    ("warfarin", "nsaid"): {
        "severity": "CRITICAL",
        "description": "Increased bleeding risk",
        "recommendation": "Avoid NSAIDs with warfarin"
    },
    ("warfarin", "aspirin"): {
        "severity": "CRITICAL",
        "description": "Increased bleeding risk",
        "recommendation": "Avoid combination or monitor INR closely"
    },
    ("nsaid", "ace_inhibitor"): {
        "severity": "MODERATE",
        "description": "Reduced antihypertensive effect",
        "recommendation": "Monitor blood pressure"
    },
    ("nsaid", "lithium"): {
        "severity": "HIGH",
        "description": "Increased lithium levels",
        "recommendation": "Monitor lithium levels"
    },
}


# ============================================================
# DRUG-ALLERGY CONFLICTS
# ============================================================

DRUG_ALLERGY_CONFLICTS = {
    # === PENICILLIN ===
    "penicillin_allergy": [
        "amoxicillin", "ampicillin", "piperacillin", "ticarcillin",
        "amoxicillin-clavulanate", "ampicillin-sulbactam",
        "penicillin", "penicillin g", "penicillin v",
        "nafcillin", "oxacillin", "mezlocillin",
        "piperacillin",
    ],
    # === SULFA ===
    "sulfa_allergy": [
        "sulfamethoxazole", "trimethoprim-sulfamethoxazole", "sulfamethoxazole-trimethoprim", "bactrim",
        "celecoxib", "sulfasalazine", "thiazide_diuretics",
        "hydrochlorothiazide", "chlorthalidone", "metolazone",
        "furosemide", "bumetanide", "torsemide",
        "acetazolamide", "dorzolamide", "brinzolamide",
        "sumatriptan", "rizatriptan", "zolmitriptan",
        "naratriptan", "eletriptan", "frovatriptan", "almotriptan",
    ],
    # === ASPIRIN / NSAID ===
    "aspirin_allergy": [
        "aspirin", "ibuprofen", "naproxen", "diclofenac",
        "indomethacin", "ketorolac", "piroxicam", "meloxicam",
        "celecoxib", "etodolac", "sulindac", "fenoprofen",
        "flurbiprofen", "ketoprofen", "meclofenamate",
        "nabumetone", "oxaprozin", "tolmetin",
    ],
    "nsaid_allergy": [
        "ibuprofen", "naproxen", "diclofenac", "indomethacin",
        "ketorolac", "piroxicam", "meloxicam", "aspirin",
        "celecoxib", "etodolac", "sulindac", "fenoprofen",
        "flurbiprofen", "ketoprofen", "meclofenamate",
        "nabumetone", "oxaprozin", "tolmetin",
    ],
    # === STATIN ===
    "statin_allergy": [
        "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
    # === ACE INHIBITOR ===
    "ace_inhibitor_allergy": [
        "lisinopril", "enalapril", "ramipril", "benazepril",
        "captopril", "quinapril", "perindopril", "fosinopril",
        "trandolapril", "moexipril",
    ],
    # === ARB ===
    "ARB_allergy": [
        "losartan", "valsartan", "irbesartan", "candesartan",
        "olmesartan", "telmisartan", "azilsartan",
    ],
    # === OPIOID ===
    "opioid_allergy": [
        "morphine", "oxycodone", "hydrocodone", "fentanyl",
        "codeine", "hydromorphone", "tramadol",
        "methadone", "buprenorphine", "meperidine",
        "oxymorphone", "tapentadol",
    ],
    # === CEPHALOSPORIN ===
    "cephalosporin_allergy": [
        "cephalexin", "ceftriaxone", "cefuroxime", "cefdinir",
        "cefepime", "ceftazidime", "cefazolin",
        "cephalexin", "cefpodoxime", "cefprozil",
        "ceftibuten", "cefotaxime", "cefoxitin",
    ],
    # === LATEX (drugs with cross-reactivity) ===
    "latex_allergy": [
        "banana", "avocado", "chestnut", "kiwi",
        "cisatracurium", "vecuronium", "pancuronium",
    ],
    # === CONTRAST DYE ===
    "contrast_dye_allergy": [
        "gadolinium", "iodinated_contrast", "barium",
    ],
    # === BETA-LACTAM (broader than just penicillin) ===
    "beta_lactam_allergy": [
        "amoxicillin", "ampicillin", "piperacillin",
        "cephalexin", "ceftriaxone", "cefuroxime",
        "cefdinir", "cefepime", "meropenem",
        "imipenem", "aztreonam",
    ],
    # === LOCAL ANESTHETIC ===
    "local_anesthetic_allergy": [
        "lidocaine", "bupivacaine", "mepivacaine",
        "ropivacaine", "prilocaine", "articaine",
    ],
    # === THYROID ===
    "thyroid_allergy": [
        "levothyroxine", "liothyronine", "armour thyroid",
    ],
    # === COBALT / METAL ===
    "cobalt_allergy": [
        "vitamin b12", "cyanocobalamin", "hydroxocobalamin",
    ],
}


# ============================================================
# CONDITION CONTRAINDICATIONS
# ============================================================

CONDITION_CONTRAINDICATIONS = {
    "kidney_disease": {
        "drugs": ["metformin", "nsaid", "ibuprofen", "naproxen", "diclofenac", "lithium", "gadolinium", "ace_inhibitor", "lisinopril", "enalapril", "ramipril", "losartan", "valsartan"],
        "reason": "May worsen kidney function or accumulate",
        "recommendation": "Dose adjustment or alternative needed"
    },
    "liver_disease": {
        "drugs": ["statin", "atorvastatin", "simvastatin", "rosuvastatin", "methotrexate", "acetaminophen", "paracetamol", "amiodarone", "isoniazid"],
        "reason": "May worsen liver function or cause hepatotoxicity",
        "recommendation": "Dose adjustment or alternative needed; monitor LFTs"
    },
    "pregnancy": {
        "drugs": [
            "warfarin", "methotrexate", "lisinopril", "enalapril", "ramipril",
            "losartan", "valsartan", "atorvastatin", "simvastatin", "rosuvastatin",
            "tetracycline", "doxycycline", "ciprofloxacin", "levofloxacin",
            "ibuprofen", "naproxen", "diclofenac", "aspirin",
            "isotretinoin", "mycophenolate"
        ],
        "reason": "May cause birth defects or pregnancy complications",
        "recommendation": "Use pregnancy-safe alternatives"
    },
    "heart_failure": {
        "drugs": ["verapamil", "diltiazem", "nsaid", "ibuprofen", "naproxen", "diclofenac", "pioglitazone", "rosiglitazone"],
        "reason": "May worsen heart failure",
        "recommendation": "Avoid or use with caution"
    },
    "asthma": {
        "drugs": ["nsaid", "ibuprofen", "naproxen", "diclofenac", "aspirin", "atenolol", "metoprolol", "propranolol", "bisoprolol", "carvedilol"],
        "reason": "May trigger bronchospasm",
        "recommendation": "Use acetaminophen for pain; avoid non-selective beta blockers"
    },
    "bleeding_disorder": {
        "drugs": ["warfarin", "aspirin", "ibuprofen", "naproxen", "diclofenac", "clopidogrel", "heparin", "rivaroxaban", "apixaban", "dabigatran"],
        "reason": "Increased bleeding risk",
        "recommendation": "Avoid anticoagulants/antiplatelets if possible"
    },
    "diabetes": {
        "drugs": ["prednisone", "prednisolone", "dexamethasone", "hydrochlorothiazide", "chlorthalidone"],
        "reason": "May increase blood sugar levels",
        "recommendation": "Monitor blood glucose closely"
    },
    "gout": {
        "drugs": ["hydrochlorothiazide", "chlorthalidone", "furosemide", "aspirin"],
        "reason": "May increase uric acid levels",
        "recommendation": "Monitor uric acid; use losartan (helps gout)"
    },
    "hypothyroidism": {
        "drugs": ["lithium", "amiodarone", "interferon", "levodopa"],
        "reason": "May worsen thyroid function",
        "recommendation": "Monitor thyroid function tests"
    },
    "gastric_ulcer": {
        "drugs": ["nsaid", "ibuprofen", "naproxen", "diclofenac", "aspirin", "prednisone", "warfarin", "clopidogrel"],
        "reason": "Increased risk of GI bleeding and ulcer perforation",
        "recommendation": "Use gastroprotective agent (PPI) if NSAID needed"
    },
    "seizure_disorder": {
        "drugs": ["tramadol", "bupropion", "clozapine", "chloroquine", "levofloxacin"],
        "reason": "May lower seizure threshold",
        "recommendation": "Use alternative analgesic"
    },
    "chronic_kidney_disease": {
        "drugs": ["metformin", "nsaid", "ibuprofen", "naproxen", "diclofenac", "lithium", "gadolinium"],
        "reason": "May worsen kidney function or accumulate",
        "recommendation": "Dose adjustment or alternative needed"
    },
    "hypertension": {
        "drugs": ["nsaid", "ibuprofen", "naproxen", "diclofenac", "decongestants", "pseudoephedrine"],
        "reason": "NSAIDs may increase blood pressure and reduce effectiveness of antihypertensives",
        "recommendation": "Use acetaminophen for pain; monitor blood pressure"
    },
    "pneumonia": {
        "drugs": ["sedative", "benzodiazepine", "opioid"],
        "reason": "Sedatives and opioids may suppress cough reflex, worsening respiratory infection",
        "recommendation": "Use lowest effective doses; monitor respiratory function"
    },
}


# ============================================================
# DOSAGE GUIDELINES
# ============================================================

DOSAGE_LIMITS = {
    # Analgesics / Antipyretics
    "acetaminophen": {
        "max_daily_adult": 3000,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of liver damage at high doses"
    },
    "paracetamol": {
        "max_daily_adult": 3000,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of liver damage at high doses"
    },
    "dolo": {
        "max_daily_adult": 3000,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of liver damage at high doses"
    },
    # NSAIDs
    "ibuprofen": {
        "max_daily_adult": 3200,  # mg
        "max_single_dose": 800,  # mg
        "warning": "Risk of GI bleeding and kidney damage"
    },
    "aspirin": {
        "max_daily_adult": 4000,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of bleeding"
    },
    "naproxen": {
        "max_daily_adult": 1000,  # mg
        "max_single_dose": 500,  # mg
        "warning": "Risk of GI bleeding"
    },
    "diclofenac": {
        "max_daily_adult": 200,  # mg
        "max_single_dose": 100,  # mg
        "warning": "Risk of GI bleeding and cardiovascular events"
    },
    # Antibiotics
    "amoxicillin": {
        "max_daily_adult": 3000,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of GI distress and diarrhea"
    },
    "azithromycin": {
        "max_daily_adult": 500,  # mg
        "max_single_dose": 500,  # mg
        "warning": "Risk of QT prolongation"
    },
    "ciprofloxacin": {
        "max_daily_adult": 1500,  # mg
        "max_single_dose": 750,  # mg
        "warning": "Risk of tendon rupture and QT prolongation"
    },
    "metronidazole": {
        "max_daily_adult": 4000,  # mg
        "max_single_dose": 2000,  # mg
        "warning": "Risk of neuropathy and seizures"
    },
    # Diabetes
    "metformin": {
        "max_daily_adult": 2550,  # mg
        "max_single_dose": 1000,  # mg
        "warning": "Risk of lactic acidosis at high doses"
    },
    # Cardiovascular
    "lisinopril": {
        "max_daily_adult": 80,  # mg
        "max_single_dose": 40,  # mg
        "warning": "Risk of hypotension and renal impairment"
    },
    "metoprolol": {
        "max_daily_adult": 400,  # mg
        "max_single_dose": 200,  # mg
        "warning": "Risk of bradycardia and hypotension"
    },
    "amlodipine": {
        "max_daily_adult": 10,  # mg
        "max_single_dose": 10,  # mg
        "warning": "Risk of peripheral edema and hypotension"
    },
    "atorvastatin": {
        "max_daily_adult": 80,  # mg
        "max_single_dose": 80,  # mg
        "warning": "Risk of rhabdomyolysis"
    },
    "warfarin": {
        "max_daily_adult": 10,  # mg
        "max_single_dose": 10,  # mg
        "warning": "Risk of hemorrhage"
    },
    # Neurology
    "gabapentin": {
        "max_daily_adult": 3600,  # mg
        "max_single_dose": 800,  # mg
        "warning": "May cause drowsiness"
    },
    "pregabalin": {
        "max_daily_adult": 600,  # mg
        "max_single_dose": 300,  # mg
        "warning": "May cause drowsiness and dizziness"
    },
    "tramadol": {
        "max_daily_adult": 400,  # mg
        "max_single_dose": 100,  # mg
        "warning": "Risk of seizures and serotonin syndrome"
    },
    # GI
    "omeprazole": {
        "max_daily_adult": 40,  # mg
        "max_single_dose": 40,  # mg
        "warning": "Long-term use increases bone fracture risk"
    },
    "pantoprazole": {
        "max_daily_adult": 80,  # mg
        "max_single_dose": 40,  # mg
        "warning": "Long-term use increases bone fracture risk"
    },
    # Diuretics
    "furosemide": {
        "max_daily_adult": 600,  # mg
        "max_single_dose": 200,  # mg
        "warning": "Risk of dehydration and electrolyte imbalance"
    },
}

# Universal safety threshold: any oral dose above this is suspicious
UNIVERSAL_MAX_SINGLE_DOSE_MG = 5000  # 5g
UNIVERSAL_MAX_DAILY_DOSE_MG = 10000  # 10g


class AnomalyDetector:
    """
    Detects potential issues in prescriptions.
    """
    
    def __init__(self):
        self.anomalies = []
    
    def check_drug_interactions(self, drugs: List[str]) -> List[Dict]:
        """
        Check for drug-drug interactions.
        
        Args:
            drugs: List of drug names (lowercase)
            
        Returns:
            List of interaction findings
        """
        findings = []
        drugs_lower = [d.lower().strip() for d in drugs]
        
        for i, drug1 in enumerate(drugs_lower):
            for drug2 in drugs_lower[i+1:]:
                # Check both orderings
                pair = (drug1, drug2)
                reverse_pair = (drug2, drug1)
                
                interaction = DRUG_INTERACTIONS.get(pair) or DRUG_INTERACTIONS.get(reverse_pair)
                
                if interaction:
                    findings.append({
                        "type": "DRUG_INTERACTION",
                        "severity": interaction["severity"],
                        "drugs": [drug1, drug2],
                        "description": interaction["description"],
                        "recommendation": interaction["recommendation"]
                    })
        
        # Warfarin + any other drug: bleeding risk warning
        if "warfarin" in drugs_lower:
            other_drugs = [d for d in drugs_lower if d != "warfarin" and d != "antibiotic"]
            if other_drugs:
                # Only warn once for warfarin + new drugs
                already_warned = any(
                    f.get("type") == "DRUG_INTERACTION" and "warfarin" in f.get("drugs", [])
                    for f in findings
                )
                if not already_warned and len(other_drugs) >= 1:
                    findings.append({
                        "type": "DRUG_INTERACTION",
                        "severity": "HIGH",
                        "drugs": ["warfarin"] + other_drugs[:2],
                        "description": f"Warfarin + {', '.join(other_drugs[:2])} may increase bleeding risk — monitor INR closely",
                        "recommendation": "Monitor INR within 3-5 days of adding new medication. Watch for signs of bleeding."
                    })
        
        return findings
    
    def check_drug_allergies(self, drugs: List[str], allergies: List[str]) -> List[Dict]:
        """
        Check for drug-allergy conflicts.
        
        Args:
            drugs: List of drug names
            allergies: List of known allergies
            
        Returns:
            List of allergy conflict findings
        """
        findings = []
        drugs_lower = [d.lower().strip() for d in drugs]
        allergies_lower = [a.lower().strip() for a in allergies]
        
        # Allergy synonym mapping — user says X, we check category Y
        ALLERGY_SYNONYMS = {
            "nsaid": ["nsaid_allergy", "aspirin_allergy"],
            "nsaids": ["nsaid_allergy", "aspirin_allergy"],
            "aspirin": ["aspirin_allergy"],
            "penicillin": ["penicillin_allergy", "beta_lactam_allergy"],
            "penicillins": ["penicillin_allergy", "beta_lactam_allergy"],
            "sulfa": ["sulfa_allergy"],
            "sulfonamide": ["sulfa_allergy"],
            "sulfonamides": ["sulfa_allergy"],
            "sulfa drugs": ["sulfa_allergy"],
            "sulfa antibiotics": ["sulfa_allergy"],
            "statin": ["statin_allergy"],
            "statins": ["statin_allergy"],
            "ace inhibitor": ["ace_inhibitor_allergy"],
            "ace inhibitors": ["ace_inhibitor_allergy"],
            "arb": ["ARB_allergy"],
            "arbs": ["ARB_allergy"],
            "opioid": ["opioid_allergy"],
            "opioids": ["opioid_allergy"],
            "narcotic": ["opioid_allergy"],
            "narcotics": ["opioid_allergy"],
            "cephalosporin": ["cephalosporin_allergy"],
            "cephalosporins": ["cephalosporin_allergy"],
            "beta-lactam": ["beta_lactam_allergy"],
            "beta lactam": ["beta_lactam_allergy"],
            "latex": ["latex_allergy"],
            "contrast": ["contrast_dye_allergy"],
            "contrast dye": ["contrast_dye_allergy"],
            "iodine": ["contrast_dye_allergy"],
            "gadolinium": ["contrast_dye_allergy"],
            "lidocaine": ["local_anesthetic_allergy"],
            "local anesthetic": ["local_anesthetic_allergy"],
            "local anesthetics": ["local_anesthetic_allergy"],
            "anesthesia": ["local_anesthetic_allergy"],
            "thyroid": ["thyroid_allergy"],
            "cobalt": ["cobalt_allergy"],
            "metal": ["cobalt_allergy"],
            "metals": ["cobalt_allergy"],
            "codeine": ["opioid_allergy"],
            "morphine": ["opioid_allergy"],
            "ibuprofen": ["nsaid_allergy", "aspirin_allergy"],
            "naproxen": ["nsaid_allergy", "aspirin_allergy"],
            "diclofenac": ["nsaid_allergy", "aspirin_allergy"],
        }
        
        for allergy in allergies_lower:
            # Find matching allergy categories
            matched_keys = set()
            
            # Direct match against category keys
            allergy_normalized = allergy.replace(" ", "_")
            for allergy_key in DRUG_ALLERGY_CONFLICTS.keys():
                if allergy_normalized in allergy_key or allergy in allergy_key:
                    matched_keys.add(allergy_key)
            
            # Synonym match
            if allergy in ALLERGY_SYNONYMS:
                matched_keys.update(ALLERGY_SYNONYMS[allergy])
            
            # Check each matched category against prescribed drugs
            for allergy_key in matched_keys:
                conflicting_drugs = DRUG_ALLERGY_CONFLICTS.get(allergy_key, [])
                for drug in drugs_lower:
                    # Direct match
                    is_match = drug in conflicting_drugs
                    # Hyphenated combination: check if any part matches
                    if not is_match and "-" in drug:
                        parts = drug.split("-")
                        is_match = any(p in conflicting_drugs for p in parts)
                    # Reverse: check if drug contains a conflicting drug
                    if not is_match:
                        is_match = any(cd in drug for cd in conflicting_drugs if len(cd) > 4)
                    if is_match:
                        findings.append({
                            "type": "DRUG_ALLERGY",
                            "severity": "CRITICAL",
                            "drug": drug,
                            "allergy": allergy,
                            "description": f"Patient allergic to {allergy} - {drug} may cause reaction",
                            "recommendation": "Use alternative medication"
                        })
        
        # Deduplicate findings
        seen = set()
        unique_findings = []
        for f in findings:
            key = (f["drug"], f["allergy"])
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
        
        return unique_findings
    
    _SCHEDULE_DIGITS = re.compile(r"([\dOoIl|\[])\s*-\s*([\dOoIl|])\s*-\s*([\dOoIl|])")

    @classmethod
    def _doses_per_day(cls, text: str, drug: str):
        """Doses per day for `drug` as written near its name ("1-0-1", "twice daily", "TDS", "every 8 hours"),
        or None if the text does not say."""
        if not text or not drug:
            return None
        low = text.lower()
        i = low.find(drug.lower())
        if i < 0:
            return None
        seg = text[i:]
        # stop at the end of this line, and at the next numbered item ("2." / "3.")
        seg = re.split(r"\n|\s\d\.\s", seg, maxsplit=1)[0][:120]
        m = cls._SCHEDULE_DIGITS.search(seg)
        if m:
            vals = [1 if c in "Il|[" else 0 if c in "Oo" else int(c) for c in m.groups()]
            n = sum(vals)
            return n or None
        s = seg.lower()
        m = re.search(r"every\s+(\d+)\s*(?:hours?|hrs?|h)\b", s)
        if m and int(m.group(1)) > 0:
            return max(1, round(24 / int(m.group(1))))
        for pat, n in ((r"\b(?:qid|qds)\b|four\s+times", 4), (r"\b(?:tds|tid)\b|three\s+times|thrice", 3),
                       (r"\b(?:bd|bid)\b|twice", 2),
                       (r"\b(?:od|qd|hs|nocte)\b|once|daily|night|bedtime|morning|weekly", 1)):
            if re.search(pat, s):
                return n
        return None

    def check_dosage_anomalies(self, medications: List[Dict], text: str = "") -> List[Dict]:
        """
        Check for dosage anomalies.
        
        Args:
            medications: List of dicts with 'drug' and 'dosage' keys
            
        Returns:
            List of dosage findings
        """
        findings = []
        
        for med in medications:
            drug = med.get("drug", "").lower().strip()
            dosage_str = med.get("dosage", "")
            
            # Parse dosage number
            dosage_num = self._parse_dosage(dosage_str)
            
            if dosage_num <= 0:
                continue
            
            # --- Universal safety check (catches ANY drug with lethal dose) ---
            if dosage_num > UNIVERSAL_MAX_SINGLE_DOSE_MG:
                findings.append({
                    "type": "DOSAGE_LETHAL",
                    "severity": "CRITICAL",
                    "drug": drug,
                    "dosage": dosage_str,
                    "max_recommended": f"{UNIVERSAL_MAX_SINGLE_DOSE_MG}mg",
                    "description": f"Dose is potentially LETHAL — {dosage_num}mg far exceeds any safe oral dose",
                    "recommendation": "Verify dosage immediately. This amount could be fatal."
                })
                continue  # Already flagged as lethal, skip other checks
            
            # --- Drug-specific dosage check ---
            if drug in DOSAGE_LIMITS:
                limits = DOSAGE_LIMITS[drug]
                max_single = limits["max_single_dose"]
                max_daily = limits["max_daily_adult"]
                
                if dosage_num > max_single:
                    # Determine severity based on how far over the limit
                    ratio = dosage_num / max_single
                    if ratio >= 5:
                        severity = "CRITICAL"
                        type_ = "DOSAGE_LETHAL"
                        desc = f"Dose is {ratio:.0f}x the maximum ({dosage_num:.0f}mg vs {max_single}mg max) — potentially fatal"
                    elif ratio >= 2:
                        severity = "HIGH"
                        type_ = "DOSAGE_HIGH"
                        desc = f"Single dose exceeds maximum ({dosage_num:.0f}mg vs {max_single}mg max)"
                    else:
                        severity = "MODERATE"
                        type_ = "DOSAGE_HIGH"
                        desc = f"Single dose slightly exceeds maximum ({dosage_num:.0f}mg vs {max_single}mg max)"
                    
                    findings.append({
                        "type": type_,
                        "severity": severity,
                        "drug": drug,
                        "dosage": dosage_str,
                        "max_recommended": f"{max_single}mg",
                        "description": desc,
                        "recommendation": limits["warning"]
                    })
                
                # Near-limit warning (>= 80% and < 100% of max single dose)
                elif max_single * 0.8 <= dosage_num < max_single:
                    pct = (dosage_num / max_single) * 100
                    findings.append({
                        "type": "DOSAGE_NEAR_LIMIT",
                        "severity": "LOW",
                        "drug": drug,
                        "dosage": dosage_str,
                        "max_recommended": f"{max_single}mg",
                        "description": f"Dose at {pct:.0f}% of maximum single dose ({dosage_num:.0f}mg of {max_single}mg max)",
                        "recommendation": f"Within safe range but close to limit. {limits['warning']}"
                    })
                
                # Daily total: use the written schedule when there is one; otherwise assume TID (worst case)
                per_day = self._doses_per_day(text, drug)
                if per_day is None and med.get("brand"):  # prescription says 'Azithral', rules know 'azithromycin'
                    per_day = self._doses_per_day(text, med["brand"])
                if per_day is not None:
                    if dosage_num * per_day > max_daily:
                        findings.append({
                            "type": "DOSAGE_DAILY_HIGH",
                            "severity": "MODERATE",
                            "drug": drug,
                            "dosage": dosage_str,
                            "max_daily": f"{max_daily}mg",
                            "description": f"Daily total exceeds maximum at {per_day}x daily ({dosage_num * per_day:.0f}mg vs {max_daily}mg max)",
                            "recommendation": limits["warning"]
                        })
                elif dosage_num * 3 > max_daily:  # frequency not stated: assume TID
                    findings.append({
                        "type": "DOSAGE_DAILY_HIGH",
                        "severity": "MODERATE",
                        "drug": drug,
                        "dosage": dosage_str,
                        "max_daily": f"{max_daily}mg",
                        "description": f"Potential daily overdose if taken 3x daily ({dosage_num * 3:.0f}mg vs {max_daily}mg max)",
                        "recommendation": limits["warning"]
                    })
            else:
                # Drug not in our database — still check for unreasonable doses
                if dosage_num > 3000:  # 3g is high for most oral drugs
                    findings.append({
                        "type": "DOSAGE_SUSPICIOUS",
                        "severity": "HIGH",
                        "drug": drug,
                        "dosage": dosage_str,
                        "description": f"{dosage_num}mg is unusually high for {drug} — verify dosage",
                        "recommendation": "Cross-reference with standard dosing guidelines"
                    })
        
        return findings
    
    # ── Therapeutic class mapping ──
    THERAPEUTIC_CLASSES = {
        "ACE_INHIBITORS": ["lisinopril", "enalapril", "ramipril", "benazepril", "captopril", "quinapril", "perindopril", "fosinopril", "trandolapril", "moexipril"],
        "ARBs": ["losartan", "valsartan", "irbesartan", "candesartan", "olmesartan", "telmisartan", "azilsartan"],
        "STATINS": ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin", "fluvastatin", "pitavastatin"],
        "NSAIDs": ["ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac", "piroxicam", "meloxicam", "celecoxib", "etodolac", "sulindac", "aspirin"],
        "BETA_BLOCKERS": ["metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol", "labetalol", "nadolol", "sotalol"],
        "PPIs": ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole", "dexlansoprazole"],
        "BENZODIAZEPINES": ["lorazepam", "diazepam", "alprazolam", "clonazepam", "temazepam", "midazolam", "chlordiazepoxide"],
        "SSRIs": ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine"],
        "OPIOIDS": ["morphine", "oxycodone", "hydrocodone", "fentanyl", "codeine", "hydromorphone", "tramadol", "methadone"],
        "ANTIBIOTICS_FLUOROQUINOLONES": ["ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin", "norfloxacin"],
        "SULFONYLUREAS": ["glipizide", "glyburide", "glimepiride", "tolbutamide", "chlorpropamide"],
        "THIAZIDE_DIURETICS": ["hydrochlorothiazide", "chlorthalidone", "metolazone", "indapamide"],
    }

    def check_therapeutic_class_duplication(self, medications: List[Dict]) -> List[Dict]:
        """Detect when two drugs from the same therapeutic class are prescribed."""
        findings = []
        drugs = [m.get("drug", "").lower().strip() for m in medications if m.get("drug")]
        
        for class_name, class_drugs in self.THERAPEUTIC_CLASSES.items():
            matched_drugs = [d for d in drugs if d in class_drugs]
            if len(matched_drugs) > 1:
                findings.append({
                    "type": "THERAPEUTIC_DUPLICATION",
                    "severity": "HIGH",
                    "drugs": matched_drugs,
                    "drug_class": class_name,
                    "description": f"Two {class_name.replace('_', ' ').lower()} detected: {', '.join(matched_drugs)}. Same therapeutic class — verify if intentional.",
                    "recommendation": f"Likely duplication of {class_name.replace('_', ' ')}. Confirm if dual therapy is intended or remove one."
                })
        
        return findings

    # ── Adverse drug reaction symptoms ──
    REACTION_SYMPTOMS = {
        "CRITICAL": {
            "symptoms": ["throat swelling", "throat tightness", "difficulty breathing", "difficulty swallowing",
                         "anaphylaxis", "anaphylactic", "throat closing", "can't breathe",
                         "chest tightness", "wheezing", "stridor"],
            "description": "Emergency allergic reaction (anaphylaxis) — seek immediate medical care",
            "recommendation": "Call emergency services immediately. Administer epinephrine if available."
        },
        "HIGH": {
            "symptoms": ["swelling of face", "swelling of lips", "swelling of tongue",
                         "swelling of eyes", "swelling of hands", "swelling of feet",
                         "severe rash", "hives", "urticaria", "angioedema"],
            "description": "Significant allergic reaction — may progress to anaphylaxis",
            "recommendation": "Seek urgent medical attention. Consider antihistamines and epinephrine."
        },
        "MODERATE": {
            "symptoms": ["rash", "itching", "hives", "skin reaction", "skin rash",
                         "itchy skin", "redness", "flushing", "sweating",
                         "low blood sugar", "hypoglycemia", "shaking", "trembling",
                         "excessive hunger", "sweating excessively"],
            "description": "Possible adverse drug reaction — monitor closely",
            "recommendation": "Monitor symptoms. If worsening, seek medical attention."
        },
        "LOW": {
            "symptoms": ["nausea", "dizziness", "headache", "drowsiness", "fatigue",
                         "stomach pain", "diarrhea", "dry mouth", "blurred vision"],
            "description": "Mild side effect — usually self-limiting",
            "recommendation": "Monitor symptoms. If persistent, consult physician."
        },
    }
    
    # ── Tolerance / Uncertainty patterns ──
    def check_allergy_context(self, text: str, allergies: List[str]) -> List[Dict]:
        """Check for allergy-related clinical nuances: tolerance, uncertainty, severity."""
        findings = []
        text_lower = text.lower()
        
        # --- Tolerance indicators ---
        tolerance_patterns = [
            r'has taken \w+ (several times|multiple times|repeatedly|before) (without symptoms|without reaction|no reaction|tolerated|fine|no issues)',
            r'taken \w+ (several times|multiple times|repeatedly|before) (without symptoms|without reaction|no reaction|tolerated|fine|no issues)',
            r'previously tolerated \w+',
            r'no reaction to \w+',
            r'\w+ has been (taken|used) without (issue|problem|reaction|symptom)',
        ]
        
        for pattern in tolerance_patterns:
            match = re.search(pattern, text_lower)
            if match:
                findings.append({
                    "type": "ALLERGY_TOLERANCE_NOTE",
                    "severity": "LOW",
                    "description": f"Patient reports previous tolerance to medication despite documented allergy — possible false-positive allergy",
                    "recommendation": "Do NOT automatically remove allergy. Document tolerance history and consult allergist for formal testing (skin prick / specific IgE)."
                })
                break
        
        # --- Unknown / unclear reaction ---
        uncertainty_patterns = [
            r'(cannot|can.t|could not|couldn.t) remember (the )?reaction',
            r'unknown (reaction|allergy|type)',
            r'unsure (about|of|what) (the )?reaction',
            r'doesn.t remember',
            r'does not remember',
            r'not sure what (happened|the reaction|the allergy)',
            r'allergic but (no|don.t|doesn.t|unclear|unknown) (details|reaction| specifics)',
        ]
        
        for pattern in uncertainty_patterns:
            match = re.search(pattern, text_lower)
            if match and allergies:
                findings.append({
                    "type": "ALLERGY_UNCERTAINTY",
                    "severity": "HIGH",
                    "description": f"Allergy documented ({', '.join(allergies)}) but reaction type is unknown or unremembered",
                    "recommendation": "Request formal allergy evaluation (skin testing or supervised oral challenge) before prescribing. Document reaction history."
                })
                break
        
        # --- Severity-based recommendation ---
        anaphylaxis_patterns = [r'anaphylax', r'anaphylactic', r'throat.?clos', r'breathing.?difficult']
        has_anaphylaxis = any(re.search(p, text_lower) for p in anaphylaxis_patterns)
        
        if has_anaphylaxis and allergies:
            findings.append({
                "type": "ALLERGY_SEVERITY_ANAPHYLAXIS",
                "severity": "CRITICAL",
                "description": f"Documented anaphylaxis to {', '.join(allergies)} — highest-risk allergy type",
                "recommendation": "Absolute contraindication. Ensure epinephrine auto-injector is available. Consider medical alert bracelet."
            })
        
        return findings

    _INDICATION_CUE = re.compile(
        r"\b(?:for|indication:?|diagnosis:?|dx:?|treat(?:ment|ing|s)?(?:\s+of)?|management\s+of)\s+(?:\w+\s+){0,2}$"
    )

    def _only_as_indication(self, text_lower: str, symptom: str) -> bool:
        """True if every occurrence of `symptom` directly follows an indication cue."""
        found = False
        start = 0
        while True:
            i = text_lower.find(symptom, start)
            if i == -1:
                break
            found = True
            if not self._INDICATION_CUE.search(text_lower[max(0, i - 40):i]):
                return False
            start = i + len(symptom)
        return found

    def check_adverse_reactions(self, text: str) -> List[Dict]:
        """Check clinical text for symptoms suggesting adverse drug reactions."""
        findings = []
        text_lower = text.lower()
        
        for severity, info in self.REACTION_SYMPTOMS.items():
            for symptom in info["symptoms"]:
                if _kw(symptom, text_lower):
                    # A symptom named as the INDICATION ("Cetirizine for Urticaria") is the
                    # reason for prescribing, not an adverse reaction -- skip if every
                    # occurrence is introduced by an indication cue.
                    if self._only_as_indication(text_lower, symptom):
                        continue
                    # Check if there is temporal proximity to a medication mention
                    # (symptom appears after 'starting', 'after', 'taking', etc.)
                    temporal_keywords = ["after", "starting", "started", "taking", "taken",
                                        "following", "began", "since starting"]
                    has_temporal = any(_kw(kw, text_lower) for kw in temporal_keywords)
                    
                    findings.append({
                        "type": "ADVERSE_REACTION",
                        "severity": severity,
                        "symptom": symptom,
                        "description": f"{info['description']} — symptom: \"{symptom}\"",
                        "recommendation": info["recommendation"],
                        "temporal_context": has_temporal
                    })
        
        return findings

    def check_incomplete_prescription(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Check for incomplete prescriptions (missing drug, strength, or schedule)."""
        findings = []
        text_lower = text.lower()
        
        # Check 1: Medication mentioned but no dosage info
        for med in medications:
            drug = med.get("drug", "")
            dosage = med.get("dosage", "")
            if drug and not dosage:
                # Topicals and liquids (gel, cream, wash, syrup, drops ...) are often written without a mg strength
                names = [drug] + ([med["brand"].lower()] if med.get("brand") else [])
                pos = min((text_lower.find(n) for n in names if text_lower.find(n) >= 0), default=-1)
                window = text_lower[pos:pos + 70] if pos >= 0 else ""
                soft = bool(re.search(r"\b(gel|cream|ointment|lotion|wash|syrup|syr|susp|drops|spray|shampoo|moisturi[sz]er|sunscreen|apply)\b", window))
                findings.append({
                    "type": "INCOMPLETE_PRESCRIPTION",
                    "severity": "MODERATE" if soft else "HIGH",
                    "drug": drug,
                    "description": f"Medication '{drug}' identified but no dosage/strength found in prescription",
                    "recommendation": "Verify dosage and strength before dispensing"
                })
        
        # Check 2: Schedule/dosing instructions without a drug
        schedule_patterns = [
            (r'\b(\d+)\s+tablet[s]?\s+(\d+)\s+times?\s+daily', "dosing schedule with tablets"),
            (r'\btake\s+(\d+)\s+tablet[s]?\b', "take N tablets"),
            (r'\b(\d+)\s+times?\s+daily\b', "times daily schedule"),
            (r'\bevery\s+(\d+)\s+hours?\b', "every N hours"),
            (r'\bper\s+day\b', "per day dosing"),
        ]
        
        has_real_drug = any(m.get("drug") and m["drug"].lower() not in ("medication", "drug", "antibiotic", "drugs") for m in medications)
        
        for pattern, desc in schedule_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Check if there is a real drug with dosage nearby
                if not has_real_drug:
                    findings.append({
                        "type": "INCOMPLETE_PRESCRIPTION",
                        "severity": "MODERATE",
                        "description": f"Dosing instructions detected (\"{desc}\") but no specific drug identified",
                        "recommendation": "Identify the drug and its strength to calculate safe dose"
                    })
                    break
        
        # Check for dose calculation ambiguity (multiple dosage values)
        dosage_numbers = [m.get("dosage", "") for m in medications if m.get("dosage")]
        if len(dosage_numbers) >= 2 and not has_real_drug:
            findings.append({
                "type": "INCOMPLETE_PRESCRIPTION",
                "severity": "MODERATE",
                "description": f"Multiple dosage values detected ({', '.join(dosage_numbers)}) but drug identity is unclear",
                "recommendation": "Clarify which drug each dosage corresponds to"
            })
        
        # Check 3: Drug mentioned with dose but no schedule/frequency
        if medications and any(m.get("dosage") for m in medications):
            has_schedule = any(_kw(kw, text_lower) for kw in ["daily", "twice", "times", "every",
                                                          "morning", "evening", "bedtime", "bid", "tid", "qid", "prn"])
            # Indian-style schedules: "1-0-1" (OCR may read 0/1 as O, l, I, |, [), "x 3 days", "at night"
            if re.search(r'[\dOoIl|\[]\s*-\s*[\dOoIl|]\s*-\s*[\dOoIl|]', text) or \
               re.search(r'\bx\s*\d+\s*days?\b|\bnight\b|\bhs\b|\bsos\b|\bstat\b|\bonce\b|\bweekly\b', text_lower):
                has_schedule = True
            if not has_schedule and len(medications) >= 1:
                for med in medications:
                    if med.get("dosage") and not med.get("drug", "").lower() == "medication":
                        findings.append({
                            "type": "MISSING_SCHEDULE",
                            "severity": "LOW",
                            "drug": med["drug"],
                            "dosage": med["dosage"],
                            "description": f"{med['drug']} {med['dosage']} prescribed without dosing schedule (frequency/timing)",
                            "recommendation": "Clarify dosing schedule (e.g., once daily, twice daily, with food)"
                        })
                        break  # Only flag once
        
        return findings

    # ── Vital sign analysis, overdose detection, pediatric concerns ──
    
    # Opioids that cause respiratory depression
    OPIOID_DRUGS = ["morphine", "oxycodone", "hydrocodone", "fentanyl", "codeine",
                    "hydromorphone", "tramadol", "methadone", "buprenorphine",
                    "meperidine", "oxymorphone", "tapentadol"]
    
    def check_vital_signs(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Analyze vital signs mentioned in clinical text for drug-related danger."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        # Respiratory rate check (normal: 12-20/min)
        rr_match = re.search(r'respiratory\s+rate\s+(?:of\s+)?(\d+)', text_lower)
        if rr_match:
            rr = int(rr_match.group(1))
            if rr < 10:
                # Check if patient is on opioids
                on_opioid = any(d in self.OPIOID_DRUGS for d in drugs)
                if on_opioid:
                    findings.append({
                        "type": "VITAL_SIGN_DANGER",
                        "severity": "CRITICAL",
                        "description": f"Respiratory rate {rr}/min (severe depression) in patient on opioid — life-threatening respiratory failure",
                        "recommendation": "IMMEDIATE: Stop opioid, administer naloxone, continuous monitoring, prepare for intubation"
                    })
                else:
                    findings.append({
                        "type": "VITAL_SIGN_DANGER",
                        "severity": "HIGH",
                        "description": f"Respiratory rate {rr}/min is critically low (normal: 12-20/min)",
                        "recommendation": "Urgent medical assessment — evaluate for respiratory failure"
                    })
            elif rr < 12:
                on_opioid = any(d in self.OPIOID_DRUGS for d in drugs)
                if on_opioid:
                    findings.append({
                        "type": "VITAL_SIGN_WARNING",
                        "severity": "HIGH",
                        "description": f"Respiratory rate {rr}/min (low) in patient on opioid — risk of respiratory depression",
                        "recommendation": "Monitor closely, reduce opioid dose or switch to non-opioid analgesic"
                    })
        
        # Drowsiness/sedation check with opioids
        drowsy_keywords = ["drowsy", "sedated", "somnolent", "decreased consciousness",
                          "unresponsive", "lethargic", "obtunded"]
        if any(_kw(kw, text_lower) for kw in drowsy_keywords):
            on_opioid = any(d in self.OPIOID_DRUGS for d in drugs)
            if on_opioid:
                findings.append({
                    "type": "VITAL_SIGN_DANGER",
                    "severity": "CRITICAL",
                    "description": "Sedation/drowsiness in patient on opioid — possible opioid toxicity",
                    "recommendation": "Assess airway, breathing, circulation. Have naloxone ready."
                })
        
        return findings

    def check_overdose_patterns(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect intentional overdose, dose discrepancies, and quantity-based overdose."""
        findings = []
        text_lower = text.lower()
        
        # --- Pattern 1: Intentional overdose ('instead of X' / 'took X instead of Y') ---
        instead_pattern = re.search(
            r'(?:took|takes|taking|prescribed)\s+(\d+(?:\.\d+)?)\s*(\w+)?\s+.*?(?:instead\s+of|rather\s+than)\s+(\d+(?:\.\d+)?)\s*(\w+)?',
            text_lower
        )
        if instead_pattern:
            taken_amount = float(instead_pattern.group(1))
            intended_amount = float(instead_pattern.group(3))
            if taken_amount > intended_amount and intended_amount > 0:
                ratio = taken_amount / intended_amount
                drug_name = "medication"
                for med in medications:
                    if med.get("drug"):
                        drug_name = med["drug"]
                        break
                if ratio >= 5:
                    severity = "CRITICAL"
                elif ratio >= 2:
                    severity = "HIGH"
                else:
                    severity = "MODERATE"
                findings.append({
                    "type": "DOSE_DISCREPANCY",
                    "severity": severity,
                    "drug": drug_name,
                    "description": f"Patient took {taken_amount} instead of {intended_amount} ({ratio:.0f}x the intended dose)",
                    "recommendation": "Seek immediate medical attention even if asymptomatic — delayed effects possible"
                })
        
        # --- Pattern 2: Quantity-based overdose ('took N tablets') ---
        tablet_pattern = re.search(
            r'(?:took|takes|taking|swallowed)\s+(\d+)\s+tablet[s]?', text_lower
        )
        if tablet_pattern:
            num_tablets = int(tablet_pattern.group(1))
            if num_tablets >= 10:
                # Find the drug
                drug_name = "unknown drug"
                for med in medications:
                    if med.get("drug") and med["drug"].lower() not in ("tablet", "medication", "drug"):
                        drug_name = med["drug"]
                        break
                # Check if it's acetaminophen/paracetamol (most dangerous in overdose)
                apap_drugs = ["acetaminophen", "paracetamol", "tylenol", "dolo"]
                is_apap = any(d in drug_name.lower() for d in apap_drugs) or any(_kw(d, text_lower) for d in apap_drugs)
                if is_apap:
                    findings.append({
                        "type": "QUANTITY_OVERDOSE",
                        "severity": "CRITICAL",
                        "drug": drug_name,
                        "description": f"{num_tablets} tablets of acetaminophen/paracetamol — severe hepatotoxicity risk even without immediate symptoms",
                        "recommendation": "EMERGENCY: Go to ER immediately. N-acetylcysteine (NAC) needed within 8 hours. Liver damage can be silent for 24-72 hours."
                    })
                else:
                    findings.append({
                        "type": "QUANTITY_OVERDOSE",
                        "severity": "HIGH",
                        "drug": drug_name,
                        "description": f"{num_tablets} tablets taken — quantity exceeds normal single dose",
                        "recommendation": "Seek medical evaluation even if asymptomatic — delayed toxicity possible"
                    })
        
        # --- Pattern 3: Asymptomatic overdose awareness ---
        asymptomatic_patterns = [
            r'(?:feel|feeling|feels)\s+(fine|okay|ok|normal|well|no symptoms|nothing)',
            r'currently\s+(?:has\s+)?no\s+symptoms',
            r'without\s+(?:any\s+)?symptoms',
            r'asymptomatic',
        ]
        has_asymptomatic = any(re.search(p, text_lower) for p in asymptomatic_patterns)
        if has_asymptomatic:
            # Check if there's an overdose context nearby
            overdose_keywords = ["overdose", "accidentally", "too much", "extra", "double",
                                "instead of", "more than", "tablets"]
            has_overdose_ctx = any(_kw(kw, text_lower) for kw in overdose_keywords)
            if has_overdose_ctx:
                findings.append({
                    "type": "ASYMPTOMATIC_OVERDOSE_WARNING",
                    "severity": "HIGH",
                    "description": "Patient feels fine but absence of symptoms does NOT mean absence of danger — many drug toxicities have delayed onset",
                    "recommendation": "Do not reassure. Seek emergency evaluation. Acetaminophen toxicity can take 24-72h to manifest. Opioid respiratory depression can be delayed."
                })
        
        return findings

    def check_pediatric_concerns(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect pediatric dosing concerns."""
        findings = []
        text_lower = text.lower()
        
        pediatric_keywords = ["child", "children", "pediatric", "infant", "baby",
                             "toddler", "neonate", "newborn", "kid", "son", "daughter",
                             "boy", "girl"]
        adult_keywords = ["adult", "adult-strength", "adult dose"]
        
        has_pediatric = any(_kw(kw, text_lower) for kw in pediatric_keywords)
        has_adult_med = any(_kw(kw, text_lower) for kw in adult_keywords)
        
        if has_pediatric:
            if has_adult_med:
                findings.append({
                    "type": "PEDIATRIC_CONCERN",
                    "severity": "CRITICAL",
                    "description": "Adult-strength medication given to child — weight-based dosing required",
                    "recommendation": "EMERGENCY: Calculate pediatric dose by weight (mg/kg). Adult doses can be lethal for children. Contact poison control."
                })
            else:
                # Check if doses seem adult-level
                for med in medications:
                    dosage = med.get("dosage", "")
                    drug = med.get("drug", "")
                    dose_num = self._parse_dosage(dosage)
                    if dose_num > 0 and drug in DOSAGE_LIMITS:
                        adult_max = DOSAGE_LIMITS[drug]["max_single_dose"]
                        if dose_num >= adult_max * 0.5:  # Half of adult max could be dangerous for child
                            findings.append({
                                "type": "PEDIATRIC_CONCERN",
                                "severity": "HIGH",
                                "drug": drug,
                                "description": f"{drug} {dosage} may be inappropriate for pediatric patient — verify weight-based dose",
                                "recommendation": "Recalculate dose based on patient weight (mg/kg). Consult pediatric formulary."
                            })
        
        # --- Missing weight for weight-based dosing ---
        weight_patterns = [
            r'(?:dose|dosage)\s+(?:is\s+)?(?:calculated\s+)?(?:based\s+on|per|by|using)\s+(?:the\s+)?weight',
            r'weight.based\s+(?:dosing|dose)',
            r'(\d+)\s*mg/kg',
            r'calculated\s+(?:using|by|based\s+on)\s+(?:the\s+)?weight',
            r'based\s+on\s+(?:the\s+)?patient\'?s?\s+weight',
        ]
        has_weight_based = any(re.search(p, text_lower) for p in weight_patterns)
        has_weight_value = bool(re.search(r'\bweight\s*(?:of|:)?\s*(\d+)', text_lower))
        
        if has_weight_based and not has_weight_value:
            findings.append({
                "type": "MISSING_WEIGHT",
                "severity": "HIGH",
                "description": "Weight-based dosing referenced but patient weight not provided",
                "recommendation": "Cannot safely calculate dose without patient weight. Obtain weight before dispensing."
            })
        
        return findings

    # ── Anticoagulant-specific clinical safety ──
    ANTICOAGULANT_DRUGS = ["warfarin", "apixaban", "rivaroxaban", "dabigatran",
                           "edoxaban", "heparin", "enoxaparin", "clopidogrel",
                           "prasugrel", "ticagrelor"]
    
    def check_anticoagulant_safety(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect anticoagulant-specific clinical dangers: bleeding, head trauma, dose doubling, missing INR."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        # Check if patient is on any anticoagulant
        on_anticoag = any(d in self.ANTICOAGULANT_DRUGS for d in drugs)
        # Also check for generic mentions
        if not on_anticoag:
            on_anticoag = any(_kw(kw, text_lower) for kw in ["anticoagulant", "anticoagulation",
                                                          "blood thinner", "blood-thinner"])
        
        if not on_anticoag:
            return findings
        
        # --- 1. Bleeding symptoms (RED FLAGS on anticoagulants) ---
        bleeding_symptoms = {
            "CRITICAL": {
                "symptoms": [
                    "vomiting blood", "vomiting bloody", "hematemesis", "hematemesis",
                    "black tarry stool", "tarry stool", "melena", "black stools",
                    "frank blood in stool", "rectal bleeding", "blood in stool",
                    "hematuria", "blood in urine",
                ],
                "description": "Active bleeding in anticoagulated patient — EMERGENCY",
                "recommendation": "IMMEDIATE: Hold anticoagulant, IV access, type & crossmatch, assess hemodynamics. Consider reversal agent."
            },
            "HIGH": {
                "symptoms": [
                    "nosebleed", "epistaxis", "bleeding gums", "gum bleeding",
                    "bruising", "ecchymosis", "petechiae",
                    "blood in sputum", "hemoptysis",
                    "prolonged bleeding", "bleeding won't stop",
                ],
                "description": "Significant bleeding in anticoagulated patient",
                "recommendation": "Assess severity. May need to hold or reduce anticoagulant. Check INR/PTT."
            },
        }
        
        for severity, info in bleeding_symptoms.items():
            for symptom in info["symptoms"]:
                if _kw(symptom, text_lower):
                    findings.append({
                        "type": "ANTICOAG_BLEEDING",
                        "severity": severity,
                        "symptom": symptom,
                        "description": f"{info['description']} — \"{symptom}\" in patient on anticoagulant",
                        "recommendation": info["recommendation"]
                    })
        
        # --- 2. Head trauma on anticoagulant ---
        head_trauma_patterns = [
            r'hit(s|ting)?\s+(?:their\s+)?head',
            r'head\s+(?:injury|trauma|hit|strike|impact)',
            r'fell\s+.*?head',
            r'falls?\s+.*?head',
            r'head\s+bleed',
            r'subdural',
            r'epidural',
            r'intracranial\s+bleed',
        ]
        
        for pattern in head_trauma_patterns:
            if re.search(pattern, text_lower):
                findings.append({
                    "type": "ANTICOAG_HEAD_TRAUMA",
                    "severity": "CRITICAL",
                    "description": "Head trauma in anticoagulated patient — high risk of intracranial hemorrhage",
                    "recommendation": "EMERGENCY: CT head immediately even if asymptomatic. Hold anticoagulant. Neurosurgical consult. Intracranial bleed can be delayed and fatal."
                })
                break
        
        # --- 3. Dose doubling question ---
        double_dose_patterns = [
            r'double\s+(?:the\s+)?(?:dose|today)',
            r'take\s+(?:two|2)\s+(?:doses?|tablets?)\s+(?:today|now|together)',
            r'make\s+up\s+(?:for\s+)?(?:the\s+)?(?:missed|forgot)',
            r'forgot\s+.*?(?:double|take\s+two|catch\s+up)',
            r'can\s+(?:i|we)\s+double',
            r'double\s+dose',
        ]
        
        for pattern in double_dose_patterns:
            if re.search(pattern, text_lower):
                findings.append({
                    "type": "ANTICOAG_DOSE_DOUBLING",
                    "severity": "CRITICAL",
                    "description": "Dose doubling of anticoagulant requested — NEVER safe to double anticoagulant dose",
                    "recommendation": "NEVER double an anticoagulant dose. If missed, take the next dose as scheduled. Doubling causes life-threatening bleeding. Contact prescriber for guidance."
                })
                break
        
        # --- 4. Missing INR monitoring for warfarin ---
        if "warfarin" in drugs:
            inr_patterns = ["inr", "inr monitoring", "inr result", "inr value",
                           "pt/inr", "prothrombin time"]
            has_inr_mention = any(_kw(p, text_lower) for p in inr_patterns)
            unavailable_patterns = ["unavailable", "not available", "no inr",
                                   "missing inr", "inr not done", "inr pending",
                                   "hasn't been checked", "has not been checked"]
            has_inr_unavailable = any(_kw(p, text_lower) for p in unavailable_patterns)
            
            if has_inr_unavailable or (not has_inr_mention and any(_kw(kw, text_lower) for kw in ["starts", "new prescription", "new medication", "new drug"])):
                findings.append({
                    "type": "ANTICOAG_MISSING_INR",
                    "severity": "HIGH",
                    "description": "Warfarin therapy without INR monitoring — cannot assess anticoagulation level",
                    "recommendation": "Check INR before starting new medication. Warfarin has narrow therapeutic index — drug interactions can cause dangerous INR changes."
                })
        
        return findings

    # ── Cardiac emergency detection ──
    BP_MEDICATIONS = ["lisinopril", "enalapril", "ramipril", "losartan", "valsartan",
                      "amlodipine", "nifedipine", "diltiazem", "verapamil",
                      "metoprolol", "atenolol", "propranolol", "carvedilol",
                      "hydrochlorothiazide", "chlorthalidone", "furosemide",
                      "spironolactone", "doxazosin", "prazosin", "terazosin"]
    
    def check_cardiac_emergencies(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect cardiac red flags: MI, heart failure, arrhythmia, medication syncope."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        # --- 1. Myocardial Infarction (MI) pattern ---
        # Classic: chest pressure/pain + jaw radiation + sweating/nausea
        has_chest_symptom = any(_kw(kw, text_lower) for kw in [
            "chest pain", "chest pressure", "chest tightness", "crushing",
            "chest discomfort", "angina", "substernal"
        ])
        has_jaw_radiation = any(_kw(kw, text_lower) for kw in [
            "radiating to jaw", "jaw pain", "jaw", "arm pain",
            "left arm", "radiating to arm", "radiating to back"
        ])
        has_autonomic = any(_kw(kw, text_lower) for kw in [
            "sweating", "diaphoresis", "nausea", "vomiting",
            "lightheaded", "cold sweat", "clammy"
        ])
        
        if has_chest_symptom and has_jaw_radiation and has_autonomic:
            findings.append({
                "type": "CARDIAC_EMERGENCY_MI",
                "severity": "CRITICAL",
                "description": "Classic myocardial infarction presentation: chest pressure with jaw radiation, sweating, and nausea",
                "recommendation": "IMMEDIATE: Call emergency services (911). Aspirin 325mg chewable if not allergic. ECG within 10 minutes. Activate cath lab if STEMI."
            })
        elif has_chest_symptom and (has_jaw_radiation or has_autonomic):
            findings.append({
                "type": "CARDIAC_EMERGENCY_MI",
                "severity": "CRITICAL",
                "description": f"Possible myocardial infarction: chest symptoms with {'jaw radiation' if has_jaw_radiation else 'autonomic symptoms'}",
                "recommendation": "Urgent cardiac evaluation needed. ECG, troponin, chest X-ray. Do NOT dismiss based on age or lack of prior history."
            })
        elif has_chest_symptom:
            findings.append({
                "type": "CARDIAC_EVAL_NEEDED",
                "severity": "HIGH",
                "description": "Chest symptoms reported — cardiac evaluation required",
                "recommendation": "ECG and cardiac workup needed. Do not reassure based on age or lack of cardiac history."
            })
        
        # --- 2. Heart failure exacerbation ---
        # Heart disease + shortness of breath lying down (orthopnea)
        has_heart_disease = any(_kw(kw, text_lower) for kw in [
            "heart disease", "heart failure", "cardiac", "chf",
            "congestive heart", "cardiomyopathy", "valvular"
        ])
        has_dyspnea = any(_kw(kw, text_lower) for kw in [
            "shortness of breath", "short of breath", "dyspnea", "breathless",
            "can't breathe", "difficulty breathing", "respiratory distress",
            "extremely short of breath"
        ])
        has_orthopnea = any(_kw(kw, text_lower) for kw in [
            "lying down", "while lying", "when lying", "supine",
            "pillow", "sitting up", "propped up", "lying flat",
            "on lying", "as soon as", "while lying down"
        ])
        has_paroxysmal = any(_kw(kw, text_lower) for kw in [
            "suddenly", "acute", "突发", "paroxysmal", "woken from sleep",
            "woke up"
        ])
        
        if has_heart_disease and has_dyspnea and (has_orthopnea or has_paroxysmal):
            findings.append({
                "type": "CARDIAC_EMERGENCY_HF",
                "severity": "CRITICAL",
                "description": "Heart failure exacerbation with orthopnea/paroxysmal dyspnea",
                "recommendation": "EMERGENCY: Evaluate for pulmonary edema. Upright positioning, oxygen, IV furosemide. BNP/NT-proBNP, chest X-ray, echocardiogram."
            })
        elif has_dyspnea and has_orthopnea:
            findings.append({
                "type": "CARDIAC_EVAL_NEEDED",
                "severity": "HIGH",
                "description": "Shortness of breath worse when lying down (orthopnea) — possible cardiac cause",
                "recommendation": "Cardiac evaluation needed. Consider heart failure, pulmonary embolism, or other cardiopulmonary causes."
            })
        
        # --- 3. Medication-related syncope (fainting on BP meds) ---
        has_syncope = any(_kw(kw, text_lower) for kw in [
            "fainted", "fainting", "syncope", "passed out", "blackout",
            "lost consciousness", "collapsed"
        ])
        has_bp_med = any(d in self.BP_MEDICATIONS for d in drugs)
        if not has_bp_med:
            has_bp_med = any(_kw(kw, text_lower) for kw in [
                "blood-pressure medication", "blood pressure medication",
                "antihypertensive", "bp medication"
            ])
        
        if has_syncope and has_bp_med:
            # Count frequency
            frequency = 1
            if _kw("twice", text_lower) or _kw("two times", text_lower):
                frequency = 2
            elif _kw("three times", text_lower):
                frequency = 3
            
            findings.append({
                "type": "CARDIAC_MED_SYNCOPE",
                "severity": "HIGH",
                "description": f"Fainting ({frequency}x today) on blood pressure medication — possible hypotension or medication adverse effect",
                "recommendation": "Check orthostatic blood pressure. May need dose reduction or medication change. Assess for dehydration, heart block, or other causes."
            })
        elif has_syncope:
            findings.append({
                "type": "CARDIAC_EVAL_NEEDED",
                "severity": "HIGH",
                "description": "Syncope (fainting) reported — cardiac evaluation needed",
                "recommendation": "ECG, Holter monitor, echocardiogram. Do not dismiss as vasovagal without ruling out cardiac cause."
            })
        
        # --- 4. Palpitations + syncope (arrhythmia concern) ---
        has_palpitations = any(_kw(kw, text_lower) for kw in [
            "palpitations", "heart racing", "rapid heartbeat",
            "irregular heartbeat", "heart skipping", "fluttering",
            "pounding heart", "tachycardia"
        ])
        
        if has_palpitations and has_syncope:
            findings.append({
                "type": "CARDIAC_ARRHYTHMIA",
                "severity": "CRITICAL",
                "description": "Palpitations with syncope — high risk of dangerous arrhythmia (VT, SVT, complete heart block)",
                "recommendation": "EMERGENCY: Continuous cardiac monitoring. ECG, troponin, electrolytes. Consider electrophysiology consult. Do NOT dismiss."
            })
        elif has_palpitations:
            findings.append({
                "type": "CARDIAC_EVAL_NEEDED",
                "severity": "HIGH",
                "description": "Palpitations reported — cardiac evaluation needed",
                "recommendation": "ECG and Holter monitor. Assess for arrhythmia, thyroid dysfunction, anemia, or medication side effects."
            })
        
        return findings

    # ── Neurological emergency detection ──
    def check_neuro_emergencies(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect neurological emergencies: stroke, hemorrhage, seizures, meningitis."""
        findings = []
        text_lower = text.lower()
        
        # --- 1. Stroke (FAST protocol: Face, Arms, Speech, Time) ---
        has_facial_droop = any(_kw(kw, text_lower) for kw in [
            "facial droop", "facial drooping", "drooping face", "face droop",
            "drooping mouth", "mouth droop", "uneven smile", "facial asymmetry",
            "one side of face"
        ])
        has_speech_issue = any(_kw(kw, text_lower) for kw in [
            "difficulty speaking", "slurred speech", "speech difficulty",
            "can't speak", "unable to speak", "trouble speaking",
            "mumbled speech", "garbled speech", "aphasia", "dysarthria",
            "difficulty talking"
        ])
        has_unilateral_weakness = any(_kw(kw, text_lower) for kw in [
            "one side", "unilateral", "weakness on one side",
            "left side weakness", "right side weakness",
            "weakness of one side", "hemiparesis", "hemiplegia",
            "one arm", "one leg", "one side of the body",
            "weakness in one arm", "weakness in one leg"
        ])
        has_sudden_onset = any(_kw(kw, text_lower) for kw in [
            "suddenly", "acute onset", "abrupt", "immediate",
            "within minutes", "within seconds", "just happened"
        ])
        
        # Stroke: facial droop + speech issue (FAST)
        if has_facial_droop and has_speech_issue:
            findings.append({
                "type": "NEURO_EMERGENCY_STROKE",
                "severity": "CRITICAL",
                "description": "STROKE ALERT: Facial drooping + difficulty speaking — classic stroke presentation (FAST positive)",
                "recommendation": "IMMEDIATE: Call stroke team/code stroke. CT head non-contrast STAT. Time critical — tPA window is 3-4.5 hours. Do NOT delay for tests."
            })
        elif has_facial_droop:
            findings.append({
                "type": "NEURO_EMERGENCY_STROKE",
                "severity": "CRITICAL",
                "description": "STROKE ALERT: Facial drooping detected — possible stroke",
                "recommendation": "IMMEDIATE: Stroke protocol. CT head, check blood glucose, assess NIHSS. Time critical for thrombolysis."
            })
        
        # Unilateral weakness = stroke until proven otherwise
        if has_unilateral_weakness:
            findings.append({
                "type": "NEURO_EMERGENCY_STROKE",
                "severity": "CRITICAL",
                "description": "STROKE ALERT: Unilateral weakness — highly suggestive of stroke",
                "recommendation": "IMMEDIATE: Stroke protocol activation. CT head, CBC, coagulation studies, glucose. Do NOT reassure or wait."
            })
        
        # Speech issue alone
        if has_speech_issue and not has_facial_droop and not has_unilateral_weakness:
            findings.append({
                "type": "NEURO_EMERGENCY_STROKE",
                "severity": "CRITICAL",
                "description": "STROKE ALERT: Difficulty speaking — possible stroke or TIA",
                "recommendation": "Urgent neurological evaluation. CT head to rule out hemorrhage. Assess for other stroke signs."
            })
        
        # --- 2. Thunderclap headache (subarachnoid hemorrhage) ---
        has_thunderclap = any(_kw(kw, text_lower) for kw in [
            "worst headache", "thunderclap", "sudden severe headache",
            "worst headache of", "explosive headache",
            "worst headache of their life", "worst headache of my life",
            "sudden onset headache", "worst ever headache"
        ])
        if has_thunderclap:
            findings.append({
                "type": "NEURO_EMERGENCY_HA",
                "severity": "CRITICAL",
                "description": "THUNDERCLAP HEADACHE — until proven otherwise, assume subarachnoid hemorrhage",
                "recommendation": "EMERGENCY: CT head non-contrast immediately (sensitivity >95% within 6h). If CT negative, lumbar puncture for xanthochromia. Neurosurgical consult."
            })
        
        # --- 3. Status epilepticus (prolonged seizure + unconscious) ---
        has_seizure = any(_kw(kw, text_lower) for kw in [
            "seizure", "convulsion", "fit", "tonic-clonic",
            "grand mal", "epileptic"
        ])
        has_prolonged = any(_kw(kw, text_lower) for kw in [
            "several minutes", "more than 5 minutes", "over 5 minutes",
            "prolonged", "continuous", "status", "hasn't stopped",
            "has not stopped", "ongoing"
        ])
        has_unconscious = any(_kw(kw, text_lower) for kw in [
            "unconscious", "not regained", "hasn't regained",
            "not responding", "won't wake", "won't respond",
            "postictal", "prolonged unconsciousness", "unresponsive"
        ])
        
        if has_seizure and (has_prolonged or has_unconscious):
            findings.append({
                "type": "NEURO_EMERGENCY_SEIZURE",
                "severity": "CRITICAL",
                "description": "STATUS EPILEPTICUS: Prolonged seizure with failure to regain consciousness",
                "recommendation": "EMERGENCY: IV lorazepam 0.1mg/kg OR rectal diazepam. ABCs, IV access, glucose check. If persistent: levetiracetam or fosphenytoin. Continuous EEG monitoring."
            })
        elif has_seizure:
            findings.append({
                "type": "NEURO_EMERGENCY_SEIZURE",
                "severity": "HIGH",
                "description": "Seizure reported — neurological evaluation needed",
                "recommendation": "CT head, EEG, electrolytes, glucose. If first seizure: admit for observation. Check for underlying cause."
            })
        
        # --- 4. Confusion after fall (intracranial hemorrhage) ---
        has_confusion = any(_kw(kw, text_lower) for kw in [
            "confused", "confusion", "disoriented", "altered mental status",
            "altered consciousness", "not making sense", "doesn't know where",
            "doesn't recognize", "amnesia", "memory loss"
        ])
        has_fall = any(_kw(kw, text_lower) for kw in [
            "after a fall", "fell down", "fell", "fallen",
            "hit head", "struck head", "head injury", "head trauma"
        ])
        
        if has_confusion and has_fall:
            findings.append({
                "type": "NEURO_EMERGENCY_ICH",
                "severity": "CRITICAL",
                "description": "Confusion after fall — possible intracranial hemorrhage (subdural/epidural hematoma)",
                "recommendation": "EMERGENCY: CT head immediately. Even mild head trauma can cause delayed intracranial hemorrhage. Neurosurgical consult if bleed found."
            })
        elif has_confusion and has_sudden_onset:
            findings.append({
                "type": "NEURO_EMERGENCY_STROKE",
                "severity": "CRITICAL",
                "description": "Sudden confusion — possible stroke or intracranial event",
                "recommendation": "Urgent neurological assessment. CT head, glucose, electrolytes. Stroke protocol if lateralizing signs."
            })
        
        # --- 5. Meningitis (headache + fever + neck stiffness + altered consciousness) ---
        has_headache = any(_kw(kw, text_lower) for kw in [
            "headache", "head pain", "cephalalgia"
        ])
        has_fever = any(_kw(kw, text_lower) for kw in [
            "fever", "febrile", "temperature", "high temperature",
            "feels hot", "chills and fever"
        ])
        has_neck_stiffness = any(_kw(kw, text_lower) for kw in [
            "neck stiffness", "stiff neck", "nuchal rigidity",
            "meningismus", "can't touch chin to chest"
        ])
        has_altered = any(_kw(kw, text_lower) for kw in [
            "altered consciousness", "confusion", "confused",
            "not responding", "decreased consciousness",
            "obtunded", "semiconscious"
        ])
        
        meningitis_count = sum([has_headache, has_fever, has_neck_stiffness, has_altered])
        
        if meningitis_count >= 3:  # Need at least 3 of 4
            findings.append({
                "type": "NEURO_EMERGENCY_MENINGITIS",
                "severity": "CRITICAL",
                "description": f"MENINGITIS SUSPECTED: {', '.join([s for s, v in [('headache', has_headache), ('fever', has_fever), ('neck stiffness', has_neck_stiffness), ('altered consciousness', has_altered)] if v])}",
                "recommendation": "EMERGENCY: Blood cultures → IV ceftriaxone + vancomycin (do NOT delay for CT/LP if toxic). LP for CSF analysis. Dexamethasone before antibiotics. Isolation precautions."
            })
        elif has_fever and has_neck_stiffness:
            findings.append({
                "type": "NEURO_EMERGENCY_MENINGITIS",
                "severity": "CRITICAL",
                "description": "Fever + neck stiffness — meningitis until proven otherwise",
                "recommendation": "EMERGENCY: Blood cultures → empiric antibiotics. LP for CSF analysis. Do NOT delay treatment."
            })
        
        return findings

    # ── Respiratory emergency detection ──
    def check_respiratory_emergencies(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect respiratory emergencies: status asthmaticus, sepsis, hypoxia, PE."""
        findings = []
        text_lower = text.lower()
        
        # --- 1. Status asthmaticus (asthma + treatment failure) ---
        has_asthma = any(_kw(kw, text_lower) for kw in [
            "asthma", "asthmatic", "bronchospasm"
        ])
        has_sob = any(_kw(kw, text_lower) for kw in [
            "short of breath", "shortness of breath", "dyspnea",
            "difficulty breathing", "can't breathe", "breathless",
            "respiratory distress"
        ])
        has_inhaler_failure = any(_kw(kw, text_lower) for kw in [
            "inhaler not helping", "inhaler isn't helping", "inhaler not working",
            "rescue inhaler", "bronchodilator not working",
            "not responding to", "not improving", "getting worse",
            "no relief", "no improvement"
        ])
        
        if has_asthma and has_sob and has_inhaler_failure:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_ASTHMA",
                "severity": "CRITICAL",
                "description": "STATUS ASTHMATICUS: Severe asthma exacerbation unresponsive to rescue inhaler",
                "recommendation": "EMERGENCY: Continuous nebulized albuterol + ipratropium, IV magnesium sulfate, systemic corticosteroids. Consider intubation if deteriorating. ABG, chest X-ray."
            })
        elif has_asthma and has_sob:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_ASTHMA",
                "severity": "HIGH",
                "description": "Severe asthma exacerbation — respiratory distress",
                "recommendation": "Urgent: Nebulized bronchodilators, systemic corticosteroids. Monitor closely for deterioration."
            })
        
        # --- 2. Sepsis from infection (pneumonia + confusion + SOB) ---
        has_infection = any(_kw(kw, text_lower) for kw in [
            "pneumonia", "infection", "sepsis", "uti", "urinary tract",
            "cellulitis", "abscess", "meningitis", "bacteremia"
        ])
        has_confusion = any(_kw(kw, text_lower) for kw in [
            "confused", "confusion", "altered mental status",
            "disoriented", "delirium", "not making sense"
        ])
        
        if has_infection and has_confusion and has_sob:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_SEPSIS",
                "severity": "CRITICAL",
                "description": "SEPSIS SUSPECTED: Infection + confusion + respiratory distress — SIRS criteria met",
                "recommendation": "EMERGENCY: Blood cultures → broad-spectrum IV antibiotics within 1 hour. Aggressive IV fluids. Lactate level. Monitor vitals q15min. Sepsis bundle."
            })
        elif has_infection and has_confusion:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_SEPSIS",
                "severity": "CRITICAL",
                "description": "Possible sepsis: Infection with altered mental status",
                "recommendation": "Urgent: Blood cultures, lactate, IV antibiotics. Assess for sepsis criteria."
            })
        
        # --- 3. Critical hypoxia (O2 sat <90%) ---
        o2_match = re.search(
            r'(?:oxygen\s+saturation|o2\s*sat|spo2|saturation)\s*(?:of\s+|is\s+|:)?\s*(\d+)%?',
            text_lower
        )
        if o2_match:
            o2_sat = int(o2_match.group(1))
            if o2_sat < 85:
                findings.append({
                    "type": "RESPIRATORY_EMERGENCY_HYPOXIA",
                    "severity": "CRITICAL",
                    "description": f"CRITICAL HYPOXIA: O2 saturation {o2_sat}% — immediately life-threatening",
                    "recommendation": "IMMEDIATE: High-flow oxygen, consider intubation. ABG, chest X-ray. Evaluate for pneumothorax, PE, severe pneumonia, or ARDS."
                })
            elif o2_sat < 90:
                findings.append({
                    "type": "RESPIRATORY_EMERGENCY_HYPOXIA",
                    "severity": "CRITICAL",
                    "description": f"SEVERE HYPOXIA: O2 saturation {o2_sat}% — requires urgent intervention",
                    "recommendation": "Supplemental oxygen, monitor closely. ABG, chest imaging. Evaluate underlying cause."
                })
            elif o2_sat < 94:
                findings.append({
                    "type": "RESPIRATORY_WARNING_HYPOXIA",
                    "severity": "HIGH",
                    "description": f"Low O2 saturation: {o2_sat}% — below normal (95-100%)",
                    "recommendation": "Supplemental oxygen if symptomatic. Monitor trend. Evaluate underlying cause."
                })
        
        # Also detect 'desaturation' or 'hypoxic' mentions
        if any(_kw(kw, text_lower) for kw in ["desaturation", "desaturating", "hypoxic", "hypoxia"]):
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_HYPOXIA",
                "severity": "CRITICAL",
                "description": "Desaturation/hypoxia reported — requires immediate oxygenation assessment",
                "recommendation": "Check O2 saturation, apply supplemental oxygen, evaluate cause."
            })
        
        # --- 4. Pulmonary embolism (sudden SOB + chest pain + risk factors) ---
        has_sudden = any(_kw(kw, text_lower) for kw in [
            "suddenly", "sudden", "acute", "abrupt",
            "just developed", "just started"
        ])
        has_sharp_chest = any(_kw(kw, text_lower) for kw in [
            "sharp chest pain", "pleuritic chest", "chest pain",
            "chest discomfort"
        ])
        pe_risk_factors = [
            "long flight", "prolonged immobility", "immobilized",
            "recent surgery", "leg swelling", "dvt", "deep vein",
            "blood clot", "clotting disorder", "thrombophilia",
            "oral contraceptive", "birth control",
            "bed rest", "wheelchair", "long drive", "long trip"
        ]
        has_pe_risk = any(_kw(kw, text_lower) for kw in pe_risk_factors)
        
        if has_sob and has_sharp_chest and has_pe_risk:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_PE",
                "severity": "CRITICAL",
                "description": "PULMONARY EMBOLISM SUSPECTED: Sudden SOB + chest pain + risk factor (prolonged immobility)",
                "recommendation": "EMERGENCY: CT pulmonary angiography (CTPA). IV heparin unless contraindicated. D-dimer, troponin, BNP. Consider tPA if massive PE with hemodynamic instability."
            })
        elif has_sob and has_sharp_chest and has_sudden:
            findings.append({
                "type": "RESPIRATORY_EMERGENCY_PE",
                "severity": "HIGH",
                "description": "Sudden onset SOB + chest pain — pulmonary embolism until proven otherwise",
                "recommendation": "CTPA, D-dimer, Wells score assessment. Consider anticoagulation while awaiting results."
            })
        
        return findings

    # ── Diabetes safety detection ──
    DIABETES_MEDS = ["insulin", "metformin", "glipizide", "glyburide", "glimepiride",
                     "sitagliptin", "empagliflozin", "dapagliflozin", "canagliflozin",
                     "liraglutide", "semaglutide", "exenatide", "pioglitazone",
                     "rosiglitazone", "acarbose", "repaglinide", "nateglinide"]
    
    def check_diabetes_safety(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect diabetes-specific emergencies: hypoglycemia, DKA, sick day rules."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        has_diabetes = any(_kw(kw, text_lower) for kw in [
            "diabetes", "diabetic", "type 1", "type 2", "t1d", "t2d"
        ])
        on_diabetes_med = any(d in self.DIABETES_MEDS for d in drugs)
        on_insulin = "insulin" in drugs or _kw("insulin", text_lower)
        
        # --- 1. Hypoglycemia (low blood sugar) ---
        hypo_symptoms = [
            "sweating", "shaking", "trembling", "confused", "confusion",
            "dizzy", "lightheaded", "shaky", "pale", "anxious",
            "rapid heartbeat", "hunger", "weak", "fatigue",
            "blurred vision", "difficulty concentrating"
        ]
        hypo_count = sum(1 for s in hypo_symptoms if _kw(s, text_lower))
        
        # Check for fasting / no food context
        no_food = any(_kw(kw, text_lower) for kw in [
            "haven't eaten", "hasn't eaten", "not eaten", "skipped meal",
            "fasting", "no food", "missed meal", "without food"
        ])
        
        if (hypo_count >= 2) and (has_diabetes or on_diabetes_med):
            if no_food:
                findings.append({
                    "type": "DIABETES_EMERGENCY_HYPOGLYCEMIA",
                    "severity": "CRITICAL",
                    "description": f"HYPOGLYCEMIA SUSPECTED: {', '.join([s for s in hypo_symptoms if _kw(s, text_lower)][:3])} in diabetic patient who hasn't eaten",
                    "recommendation": "IMMEDIATE: Give 15-20g fast-acting glucose (juice, glucose tablets). Recheck glucose in 15 min. If unconscious: glucagon IM or IV dextrose."
                })
            else:
                findings.append({
                    "type": "DIABETES_EMERGENCY_HYPOGLYCEMIA",
                    "severity": "HIGH",
                    "description": f"Possible hypoglycemia: {', '.join([s for s in hypo_symptoms if _kw(s, text_lower)][:3])} in diabetic patient",
                    "recommendation": "Check blood glucose immediately. If <70 mg/dL: give 15-20g fast-acting glucose. Recheck in 15 min."
                })
        
        # --- 2. DKA (diabetic ketoacidosis) ---
        has_high_glucose = any(_kw(kw, text_lower) for kw in [
            "high blood glucose", "very high glucose", "hyperglycemia",
            "blood sugar high", "glucose high", "hyperglycemic",
            r"glucose \d{3,}", r"blood sugar \d{3,}",
            "very high blood sugar", "extremely high"
        ])
        has_kussmaul = any(_kw(kw, text_lower) for kw in [
            "breathing deeply", "deep breathing", "kussmaul",
            "unusually deep", "rapid deep breathing",
            "air hunger", "deep labored breathing"
        ])
        has_dka_symptoms = any(_kw(kw, text_lower) for kw in [
            "vomiting", "nausea", "abdominal pain", "stomach pain",
            "fruity breath", "acetone breath", "sweet breath",
            "ketones", "ketotic"
        ])
        
        if (has_diabetes or on_diabetes_med) and has_high_glucose and has_kussmaul:
            findings.append({
                "type": "DIABETES_EMERGENCY_DKA",
                "severity": "CRITICAL",
                "description": "DIABETIC KETOACIDOSIS (DKA) SUSPECTED: High glucose + Kussmaul breathing",
                "recommendation": "EMERGENCY: IV fluids, IV insulin drip, potassium monitoring. ABG for pH, serum ketones, anion gap. Admit to ICU."
            })
        elif (has_diabetes or on_diabetes_med) and has_high_glucose and has_dka_symptoms:
            findings.append({
                "type": "DIABETES_EMERGENCY_DKA",
                "severity": "CRITICAL",
                "description": "DKA SUSPECTED: High glucose + vomiting/abdominal symptoms",
                "recommendation": "EMERGENCY: Check ketones, ABG, electrolytes. IV fluids and insulin."
            })
        
        # --- 3. Severe hypoglycemia (unconscious on insulin) ---
        has_unconscious = any(_kw(kw, text_lower) for kw in [
            "unconscious", "unresponsive", "not responding",
            "passed out", "collapsed", "won't wake",
            "can't wake", "loss of consciousness"
        ])
        
        if has_unconscious and (on_insulin or _kw("insulin", text_lower)):
            findings.append({
                "type": "DIABETES_EMERGENCY_SEVERE_HYPO",
                "severity": "CRITICAL",
                "description": "SEVERE HYPOGLYCEMIA: Unconscious patient on insulin",
                "recommendation": "IMMEDIATE: IV dextrose (D50) OR glucagon 1mg IM. Do NOT give oral glucose to unconscious patient (aspiration risk). Check glucose, reverse any opioids."
            })
        elif has_unconscious and (has_diabetes or on_diabetes_med):
            findings.append({
                "type": "DIABETES_EMERGENCY_SEVERE_HYPO",
                "severity": "CRITICAL",
                "description": "Unconscious diabetic patient — assume hypoglycemia until proven otherwise",
                "recommendation": "IMMEDIATE: Check glucose. If low or unknown: IV dextrose or glucagon. Do NOT wait for lab results."
            })
        
        # --- 4. Inconsistent glucose reading ---
        inconsistent_patterns = [
            r'glucose.*?(?:inconsistent|doesn.t match|does not match|doesn.t match|seems wrong|doesn.t make sense|does not make sense)',
            r'glucose.*?(?:but|however|yet).*?(?:feel|feeling|feels)',
            r'glucose.*?(?:reading|result).*?(?:inaccurate|wrong|seems off|seems strange)',
        ]
        if any(re.search(p, text_lower) for p in inconsistent_patterns):
            findings.append({
                "type": "DIABETES_GLUCOSE_INCONSISTENT",
                "severity": "HIGH",
                "description": "Glucose reading inconsistent with symptoms — possible meter error, hemoglobin variant, or lab vs glucometer discrepancy",
                "recommendation": "Repeat glucose check with different finger/device. Check for interfering substances. If severely symptomatic, treat empirically and obtain stat lab glucose."
            })
        
        # --- 5. Sick day rules (stop all meds question) ---
        stop_all_meds = any(_kw(kw, text_lower) for kw in [
            "stop all", "stop taking all", "discontinue all",
            "quit all", "completely stop"
        ])
        is_sick = any(_kw(kw, text_lower) for kw in [
            "sick", "ill", "vomiting", "diarrhea",
            "not eating", "can't eat", "can't keep",
            "fever", "infection"
        ])
        diabetes_meds_mentioned = any(_kw(kw, text_lower) for kw in [
            "diabetes medication", "diabetes medications",
            "diabetic medication", "diabetic medications",
            "metformin", "insulin"
        ])
        
        if (stop_all_meds and is_sick) or (stop_all_meds and diabetes_meds_mentioned):
            findings.append({
                "type": "DIABETES_SICK_DAY_RULES",
                "severity": "HIGH",
                "description": "Patient asking about stopping diabetes medications when sick — NEVER stop insulin completely, even when not eating",
                "recommendation": "Sick day rules: NEVER stop insulin completely. Monitor glucose every 2-4 hours. Maintain hydration. If on insulin: may need to INCREASE dose during illness. Contact physician for dose adjustment. Metformin may be held if dehydrated/vomiting."
            })
        
        # --- 6. Diabetes + vomiting (dehydration/DKA risk) ---
        has_vomiting = any(_kw(kw, text_lower) for kw in [
            "vomiting", "vomiting repeatedly", "can't keep",
            "cannot keep", "throwing up"
        ])
        
        if has_vomiting and (has_diabetes or on_diabetes_med):
            findings.append({
                "type": "DIABETES_SICK_DAY_RISK",
                "severity": "HIGH",
                "description": "Diabetic patient vomiting — risk of dehydration, hypoglycemia, and DKA",
                "recommendation": "Monitor glucose every 2-4 hours. Sip clear fluids. If glucose <70: small sips of juice. If >250: check ketones. If can't keep fluids down for >4 hours: ER for IV fluids."
            })
        
        return findings

    # ── Pregnancy safety detection ──
    def check_pregnancy_safety(self, text: str, medications: List[Dict], conditions: List[str] = None) -> List[Dict]:
        """Detect pregnancy-specific emergencies and safety concerns."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        conditions_lower = [c.lower() for c in (conditions or [])]
        
        is_pregnant = any(_kw(kw, text_lower) for kw in [
            "pregnant", "pregnancy", "pregnancies", "gestation",
            "trimester", "fetus", "foetus", "antenatal",
            "gravida", "primigravida", "multigravida"
        ])
        is_breastfeeding = any(_kw(kw, text_lower) for kw in [
            "breastfeeding", "breast-feeding", "nursing",
            "lactating", "lactation", "lactating mother"
        ])
        
        if not is_pregnant and not is_breastfeeding:
            return findings
        
        # --- 1. Pregnancy-specific emergencies ---
        has_vaginal_bleeding = any(_kw(kw, text_lower) for kw in [
            "vaginal bleeding", "bleeding", "spotting",
            "vaginal haemorrhage", "vaginal hemorrhage"
        ])
        has_abdominal_pain = any(_kw(kw, text_lower) for kw in [
            "abdominal pain", "severe abdominal", "stomach pain",
            "pelvic pain", "lower abdominal", "cramping"
        ])
        
        if is_pregnant and has_vaginal_bleeding and has_abdominal_pain:
            findings.append({
                "type": "PREGNANCY_EMERGENCY_BLEEDING",
                "severity": "CRITICAL",
                "description": "EMERGENCY: Vaginal bleeding + severe abdominal pain in pregnancy — possible ectopic pregnancy, placental abruption, or miscarriage",
                "recommendation": "IMMEDIATE: Transvaginal ultrasound, quantitative beta-hCG, blood type & crossmatch. Ectopic pregnancy is life-threatening. Do NOT give analgesics until ectopic ruled out."
            })
        elif is_pregnant and has_vaginal_bleeding:
            findings.append({
                "type": "PREGNANCY_EMERGENCY_BLEEDING",
                "severity": "CRITICAL",
                "description": "Vaginal bleeding in pregnancy — requires urgent evaluation",
                "recommendation": "Urgent: Ultrasound, beta-hCG. Rule out ectopic pregnancy, placenta previa, abruption."
            })
        
        # --- 2. Preeclampsia (severe headache + visual disturbances in pregnancy) ---
        has_headache = any(_kw(kw, text_lower) for kw in [
            "severe headache", "headache", "thunderclap",
            "worst headache"
        ])
        has_visual = any(_kw(kw, text_lower) for kw in [
            "visual disturbance", "visual changes", "blurred vision",
            "seeing spots", "floaters", "light sensitivity",
            "visual symptoms", "eyesight changes"
        ])
        has_preeclampsia_signs = any(_kw(kw, text_lower) for kw in [
            "high blood pressure", "hypertension", "proteinuria",
            "swelling", "edema", "face swelling",
            "hands swollen", "feet swollen", "upper abdominal pain",
            "right upper quadrant pain"
        ])
        
        if is_pregnant and has_headache and has_visual:
            findings.append({
                "type": "PREGNANCY_EMERGENCY_PREECLAMPSIA",
                "severity": "CRITICAL",
                "description": "PREECLAMPSIA/ECLAMPSIA SUSPECTED: Severe headache + visual disturbances in pregnancy",
                "recommendation": "EMERGENCY: Check blood pressure STAT. If BP >160/110: IV labetalol or hydralazine. Magnesium sulfate for seizure prophylaxis. Deliver if >34 weeks. Monitor for eclamptic seizures."
            })
        elif is_pregnant and has_preeclampsia_signs:
            findings.append({
                "type": "PREGNANCY_EMERGENCY_PREECLAMPSIA",
                "severity": "HIGH",
                "description": "Possible preeclampsia: signs/symptoms in pregnancy",
                "recommendation": "Check BP, urine protein, liver function. If severe: consider delivery."
            })
        
        # --- 3. Unknown medication in pregnancy ---
        has_unknown_med = any(_kw(kw, text_lower) for kw in [
            "unknown medication", "unknown drug", "don't know",
            "doesn't know", "unsure what", "can't identify",
            "not sure what", "doesn't provide", "not provided",
            "don't have the name", "without the name", "no name"
        ])
        has_med_question = any(_kw(kw, text_lower) for kw in [
            "can i take", "can they take", "safe to take",
            "is it safe", "can she take", "is this safe",
            "asks whether", "asks if", "asking whether",
            "asking if", "wants to know if", "wants to know whether"
        ])
        
        if is_pregnant and has_unknown_med and has_med_question:
            findings.append({
                "type": "PREGNANCY_UNKNOWN_MED",
                "severity": "HIGH",
                "description": "Pregnant patient asking about unknown medication — CANNOT assess safety without identifying the drug",
                "recommendation": "Identify the specific drug before making any recommendation. Check FDA pregnancy category and LactMed database. Many drugs are teratogenic in first trimester."
            })
        elif is_pregnant and has_med_question:
            # Has a named drug — check if it's in contraindication list
            drug_to_check = None
            for m in medications:
                if m.get("drug") and m["drug"].lower() not in ("medication", "drug", "tablet"):
                    drug_to_check = m["drug"].lower()
                    break
            
            if drug_to_check and drug_to_check in self.CONTRAINDICATED_IN_PREGNANCY:
                category = self.CONTRAINDICATED_IN_PREGNANCY[drug_to_check]
                findings.append({
                    "type": "PREGNANCY_DRUG_CONTRAINDICATED",
                    "severity": "CRITICAL",
                    "drug": drug_to_check,
                    "description": f"{drug_to_check.title()} is CONTRAINDICATED in pregnancy ({category})",
                    "recommendation": "Do NOT prescribe. Use pregnancy-safe alternative. Consult OB/GYN."
                })
        
        # --- 4. Breastfeeding + unknown medication ---
        if is_breastfeeding and has_unknown_med and has_med_question:
            findings.append({
                "type": "BREASTFEEDING_UNKNOWN_MED",
                "severity": "HIGH",
                "description": "Breastfeeding patient asking about unknown medication — CANNOT assess safety without drug name",
                "recommendation": "Identify the specific drug. Check LactMed database for excretion into breast milk. Some drugs are safe, others are absolute contraindications."
            })
        
        # --- 5. Known drug in pregnancy (check against list) ---
        if is_pregnant:
            for drug in drugs:
                if drug in self.CONTRAINDICATED_IN_PREGNANCY:
                    category = self.CONTRAINDICATED_IN_PREGNANCY[drug]
                    findings.append({
                        "type": "PREGNANCY_DRUG_CONTRAINDICATED",
                        "severity": "CRITICAL",
                        "drug": drug,
                        "description": f"{drug.title()} is CONTRAINDICATED in pregnancy ({category})",
                        "recommendation": "Discontinue and use pregnancy-safe alternative. Consult OB/GYN."
                    })
        
        return findings

    # Pregnancy-contraindicated drugs
    CONTRAINDICATED_IN_PREGNANCY = {
        "warfarin": "Category X — causes fetal hemorrhage",
        "methotrexate": "Category X — teratogenic, abortifacient",
        "isotretinoin": "Category X — severe teratogenic (craniofacial, cardiac, CNS)",
        "mycophenolate": "Category X — increased risk of miscarriage and birth defects",
        "lisinopril": "Category D (2nd/3rd tri) — renal dysgenesis, oligohydramnios",
        "enalapril": "Category D (2nd/3rd tri) — fetal renal damage",
        "ramipril": "Category D (2nd/3rd tri) — fetal renal damage",
        "losartan": "Category D (2nd/3rd tri) — fetal renal damage",
        "valsartan": "Category D (2nd/3rd tri) — fetal renal damage",
        "atorvastatin": "Category X — statins inhibit fetal cholesterol synthesis",
        "simvastatin": "Category X — teratogenic",
        "rosuvastatin": "Category X — teratogenic",
        "tetracycline": "Category D — tooth discoloration, bone growth inhibition",
        "doxycycline": "Category D — same as tetracycline",
        "ciprofloxacin": "Category C — cartilage damage in animal studies",
        "levofloxacin": "Category C — tendon/cartilage concerns",
        "ibuprofen": "Category C/D (after 20 weeks) — premature closure of ductus arteriosus",
        "naproxen": "Category C/D (after 20 weeks) — same as ibuprofen",
        "diclofenac": "Category C/D (after 20 weeks) — same as ibuprofen",
        "aspirin": "Category C/D (after 30 weeks) — premature closure of ductus arteriosus",
        "valproic acid": "Category D — neural tube defects, low IQ",
        "carbamazepine": "Category D — neural tube defects",
        "phenytoin": "Category D — fetal hydantoin syndrome",
        "lithium": "Category D — cardiac malformations (Ebstein anomaly)",
        "statin": "Category X — teratogenic",
    }

    # ── Lab value emergency detection ──
    def check_lab_value_emergencies(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect dangerous lab values: electrolytes, INR, creatinine, platelets."""
        findings = []
        text_lower = text.lower()
        
        # --- 1. Potassium (K+) — normal 3.5-5.0 mmol/L ---
        k_match = re.search(
            r'potassium\s*(?:is\s+|:\s+|=\s+)?\s*(\d+\.?\d*)\s*(?:mmol|meq)/?l?',
            text_lower
        )
        if k_match:
            k_val = float(k_match.group(1))
            if k_val > 6.5:
                findings.append({
                    "type": "LAB_EMERGENCY_HYPERKALEMIA",
                    "severity": "CRITICAL",
                    "lab": "potassium",
                    "value": f"{k_val} mmol/L",
                    "normal": "3.5-5.0 mmol/L",
                    "description": f"SEVERE HYPERKALEMIA: K+ {k_val} mmol/L — can cause fatal cardiac arrhythmias EVEN IF ASYMPTOMATIC",
                    "recommendation": "IMMEDIATE: Continuous cardiac monitor. IV calcium gluconate (cardiac membrane stabilizer). IV insulin + dextrose. Sodium bicarbonate. Kayexalate. Prepare for dialysis."
                })
            elif k_val > 6.0:
                findings.append({
                    "type": "LAB_EMERGENCY_HYPERKALEMIA",
                    "severity": "CRITICAL",
                    "lab": "potassium",
                    "value": f"{k_val} mmol/L",
                    "normal": "3.5-5.0 mmol/L",
                    "description": f"HYPERKALEMIA: K+ {k_val} mmol/L — high risk of cardiac arrhythmia",
                    "recommendation": "Cardiac monitor, repeat K+, ECG. IV calcium if ECG changes. Treat underlying cause."
                })
            elif k_val > 5.5:
                findings.append({
                    "type": "LAB_WARNING_HYPERKALEMIA",
                    "severity": "HIGH",
                    "lab": "potassium",
                    "value": f"{k_val} mmol/L",
                    "normal": "3.5-5.0 mmol/L",
                    "description": f"Mild hyperkalemia: K+ {k_val} mmol/L",
                    "recommendation": "Repeat K+, check renal function. Dietary potassium restriction."
                })
        
        # --- 2. Sodium (Na+) — normal 135-145 mmol/L ---
        na_match = re.search(
            r'sodium\s*(?:is\s+|:\s+|=\s+)?\s*(\d+\.?\d*)\s*(?:mmol|meq)/?l?',
            text_lower
        )
        if na_match:
            na_val = float(na_match.group(1))
            if na_val < 120:
                findings.append({
                    "type": "LAB_EMERGENCY_HYPONATREMIA",
                    "severity": "CRITICAL",
                    "lab": "sodium",
                    "value": f"{na_val} mmol/L",
                    "normal": "135-145 mmol/L",
                    "description": f"SEVERE HYPONATREMIA: Na+ {na_val} mmol/L — risk of cerebral edema, seizures, herniation",
                    "recommendation": "EMERGENCY: Hypertonic saline (3% NaCl) if symptomatic. Fluid restrict. Check glucose (pseudohyponatremia). Correct slowly (max 8-10 mmol/L per 24h to avoid osmotic demyelination)."
                })
            elif na_val < 125:
                findings.append({
                    "type": "LAB_EMERGENCY_HYPONATREMIA",
                    "severity": "CRITICAL",
                    "lab": "sodium",
                    "value": f"{na_val} mmol/L",
                    "normal": "135-145 mmol/L",
                    "description": f"HYPONATREMIA: Na+ {na_val} mmol/L — at risk for seizures and cerebral edema",
                    "recommendation": "Fluid restrict. If symptomatic: hypertonic saline. Correct slowly. Monitor neuro status."
                })
            elif na_val < 130:
                findings.append({
                    "type": "LAB_WARNING_HYPONATREMIA",
                    "severity": "HIGH",
                    "lab": "sodium",
                    "value": f"{na_val} mmol/L",
                    "normal": "135-145 mmol/L",
                    "description": f"Mild hyponatremia: Na+ {na_val} mmol/L",
                    "recommendation": "Investigate cause (SIADH, heart failure, liver disease). Fluid restriction."
                })
        
        # --- 3. INR — normal 0.8-1.2 (therapeutic 2.0-3.0 on warfarin) ---
        inr_match = re.search(
            r'inr\s*(?:is\s+|:\s+|=\s+)?\s*(\d+\.?\d*)',
            text_lower
        )
        
        # Also detect qualitative INR descriptions
        inr_qualitative = any(_kw(kw, text_lower) for kw in [
            "inr.*?extremely elevated", "inr.*?very high", "inr.*?dangerously",
            "inr.*?critical", "inr.*?supratherapeutic", "inr.*?critically elevated",
            "elevated inr", "inr elevated", "inr is extremely",
            "inr is very high", "inr critically high"
        ])
        
        inr_val = None
        if inr_match:
            inr_val = float(inr_match.group(1))
            has_bleeding = any(_kw(kw, text_lower) for kw in [
                "bleeding", "hemorrhage", "blood", "bruising",
                "black stool", "tarry stool", "vomiting blood"
            ])
            if inr_val > 10:
                severity = "CRITICAL"
                desc = f"DANGEROUSLY ELEVATED INR: {inr_val} — extreme hemorrhage risk"
                rec = "EMERGENCY: Hold warfarin. IV vitamin K. Prothrombin complex concentrate (PCC) or FFP."
                if has_bleeding:
                    desc += " WITH ACTIVE BLEEDING"
                    rec += " Aggressive resuscitation."
                findings.append({
                    "type": "LAB_EMERGENCY_INR",
                    "severity": severity,
                    "lab": "INR",
                    "value": str(inr_val),
                    "normal": "2.0-3.0 (therapeutic)",
                    "description": desc,
                    "recommendation": rec
                })
            elif inr_val > 5:
                findings.append({
                    "type": "LAB_EMERGENCY_INR",
                    "severity": "CRITICAL",
                    "lab": "INR",
                    "value": str(inr_val),
                    "normal": "2.0-3.0 (therapeutic)",
                    "description": f"HIGHLY ELEVATED INR: {inr_val} — significant hemorrhage risk",
                    "recommendation": "Hold warfarin. IV vitamin K. If bleeding: PCC. Recheck INR in 6-12 hours."
                })
            elif inr_val > 4:
                findings.append({
                    "type": "LAB_WARNING_INR",
                    "severity": "HIGH",
                    "lab": "INR",
                    "value": str(inr_val),
                    "normal": "2.0-3.0 (therapeutic)",
                    "description": f"Elevated INR: {inr_val} — supra-therapeutic",
                    "recommendation": "Hold 1-2 doses warfarin. Oral vitamin K if no bleeding. Recheck INR."
                })
        
        # Qualitative INR (no specific number but described as dangerous)
        if inr_val is None and inr_qualitative:
            has_bleeding = any(_kw(kw, text_lower) for kw in [
                "bleeding", "hemorrhage", "blood", "bruising",
                "black stool", "tarry stool", "vomiting blood"
            ])
            desc = "CRITICALLY ELEVATED INR described as extremely/very high"
            rec = "EMERGENCY: Check INR immediately. Hold warfarin. IV vitamin K. If bleeding: PCC."
            if has_bleeding:
                desc += " WITH ACTIVE BLEEDING — life-threatening hemorrhage risk"
                rec += " Aggressive resuscitation."
            findings.append({
                "type": "LAB_EMERGENCY_INR",
                "severity": "CRITICAL",
                "lab": "INR",
                "value": "extremely elevated (no numeric value)",
                "normal": "2.0-3.0 (therapeutic)",
                "description": desc,
                "recommendation": rec
            })
        
        # --- 4. Creatinine increase (AKI) ---
        has_creatinine_increase = (
            bool(re.search(r'creatinine.*?(?:increased|elevated|risen|jumped|spiked|doubled|tripled|sudden|acute|new)', text_lower)) or
            any(_kw(kw, text_lower) for kw in ["acute kidney injury", "aki"]) or
            bool(re.search(r'renal function.*?(?:worsened|deteriorated|declined)', text_lower)) or
            bool(re.search(r'kidney function.*?(?:worsened|deteriorated|declined)', text_lower))
        )
        has_new_med = any(_kw(kw, text_lower) for kw in [
            "new medication", "starting a new", "after starting",
            "after beginning", "new drug", "new prescription"
        ])
        
        if has_creatinine_increase and has_new_med:
            findings.append({
                "type": "LAB_EMERGENCY_AKI",
                "severity": "CRITICAL",
                "lab": "creatinine",
                "description": "ACUTE KIDNEY INJURY: Creatinine increased after starting new medication — drug nephrotoxicity",
                "recommendation": "STOP suspect nephrotoxic drug. IV fluids, check urine output. Renal consult. Nephrotoxic drugs include NSAIDs, aminoglycosides, contrast dye, ACE inhibitors."
            })
        elif has_creatinine_increase:
            findings.append({
                "type": "LAB_EMERGENCY_AKI",
                "severity": "HIGH",
                "lab": "creatinine",
                "description": "Acute kidney injury suspected — creatinine increase",
                "recommendation": "Check GFR, urine output, electrolytes. Review nephrotoxic medications. Renal consult."
            })
        
        # --- 5. Platelet count ---
        platelet_match = re.search(
            r'(?:platelet|plt|platelets)\s*(?:count\s*)?(?:is\s+|:\s+|=\s+)?\s*(?:low|\d+)',
            text_lower
        )
        has_low_platelets = any(_kw(kw, text_lower) for kw in [
            "low platelet", "platelet count is low",
            "thrombocytopenia", "platelets are low",
            "decreased platelet", "low plt"
        ])
        has_bruising = any(_kw(kw, text_lower) for kw in [
            "bruising", "bruises", "ecchymosis",
            "petechiae", "unexplained bruising",
            "new bruising", "easy bruising"
        ])
        
        if has_low_platelets and has_bruising:
            findings.append({
                "type": "LAB_EMERGENCY_THROMBOCYTOPENIA",
                "severity": "HIGH",
                "lab": "platelets",
                "description": "Thrombocytopenia with active bruising — bleeding risk",
                "recommendation": "Check platelet count and morphology. Review medications (heparin, quinine, antibiotics). Check for DIC. Transfusion threshold typically <10,000 (or <50,000 with active bleeding)."
            })
        elif has_low_platelets:
            findings.append({
                "type": "LAB_WARNING_THROMBOCYTOPENIA",
                "severity": "HIGH",
                "lab": "platelets",
                "description": "Low platelet count reported — bleeding risk",
                "recommendation": "Check count, review medications, rule out DIC, TTP/HUS, or ITP."
            })
        
        return findings

    # ── Pediatric safety detection ──
    def check_pediatric_safety(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect pediatric-specific safety concerns: dosing, polypharmacy, ingestion, emergencies."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        pediatric_keywords = ["child", "children", "pediatric", "infant", "baby",
                             "toddler", "neonate", "newborn", "kid", "son", "daughter",
                             "boy", "girl", "2-year", "3-year", "4-year", "5-year",
                             "6-year", "7-year", "8-year", "9-year", "10-year",
                             "11-year", "12-year", "year-old"]
        is_pediatric = any(_kw(kw, text_lower) for kw in pediatric_keywords)
        
        if not is_pediatric:
            return findings
        
        # --- 1. Pediatric dosing without weight ---
        has_dose_question = any(_kw(kw, text_lower) for kw in [
            "dose", "dosage", "how much", "give", "administer",
            "prescribe", "take", "how many"
        ])
        has_weight = any(_kw(kw, text_lower) for kw in [
            "weight", "kg", "kilogram", "pounds", "lbs",
            "weighs", "weight of"
        ])
        
        if has_dose_question and not has_weight:
            findings.append({
                "type": "PEDIATRIC_MISSING_WEIGHT",
                "severity": "HIGH",
                "description": "Pediatric dosing requested but patient weight not provided — CANNOT safely calculate dose",
                "recommendation": "Pediatric doses are weight-based (mg/kg). An adult dose could be lethal. Obtain accurate weight before prescribing."
            })
        
        # --- 2. Pediatric polypharmacy (multiple cold medicines) ---
        cold_meds = ["cold", "cough", "flu", "fever", " Tylenol", "acetaminophen",
                     "ibuprofen", "advil", "motrin", "dimetapp", "children's",
                     "pediacare", "children's tylenol", "children's advil"]
        cold_med_count = sum(1 for m in medications if any(cm in m.get("drug", "").lower() for cm in cold_meds))
        
        # Also check for duplicate ingredient risk from text
        has_two_meds = any(_kw(kw, text_lower) for kw in [
            "two different", "two medications", "two medicines",
            "both", "multiple medications", "multiple medicines",
            "two drugs"
        ])
        
        if is_pediatric and (has_two_meds or cold_med_count >= 2):
            findings.append({
                "type": "PEDIATRIC_POLYPHARMACY",
                "severity": "HIGH",
                "description": "Multiple medications given to child — risk of duplicate active ingredients (especially acetaminophen)",
                "recommendation": "Check ALL active ingredients in both products. Acetaminophen in two products = double dose = liver failure risk. Use ONE product at a time."
            })
        
        # --- 3. Unknown ingestion amount ---
        has_unknown_amount = any(_kw(kw, text_lower) for kw in [
            "unknown amount", "unknown quantity", "don't know how much",
            "doesn't know how much", "unsure how much",
            "how many pills", "how many tablets", "don't know how many",
            "unknown number", "not sure how much"
        ])
        has_ingestion = any(_kw(kw, text_lower) for kw in [
            "drank", "drinks", "ingested", "swallowed", "ate",
            "took", "consumed", "eaten"
        ])
        
        if is_pediatric and has_unknown_amount and has_ingestion:
            findings.append({
                "type": "PEDIATRIC_UNKNOWN_INGESTION",
                "severity": "CRITICAL",
                "description": "Child ingested unknown amount of medication — CANNOT assess toxicity without knowing quantity",
                "recommendation": "EMERGENCY: Bring medication container to ER. Call Poison Control (1-800-222-1222 in US). Do NOT induce vomiting. Monitor for symptoms."
            })
        elif is_pediatric and has_ingestion and not has_unknown_amount:
            # Check if amount is mentioned
            amount_mentioned = bool(re.search(r'\d+\s*(?:tablet|pill|ml|mg|teaspoon|tsp|cup)', text_lower))
            if not amount_mentioned:
                findings.append({
                    "type": "PEDIATRIC_INGESTION_NO_AMOUNT",
                    "severity": "HIGH",
                    "description": "Child ingested medication but amount not specified — need quantity to assess toxicity",
                    "recommendation": "Determine exact amount and timing. Bring medication to ER. Call Poison Control."
                })
        
        # --- 4. Infant emergency (fever + lethargy = meningitis risk) ---
        is_infant = any(_kw(kw, text_lower) for kw in [
            "infant", "baby", "neonate", "newborn",
            r"\d+ month", r"\d+ weeks"
        ])
        has_fever = any(_kw(kw, text_lower) for kw in [
            "fever", "febrile", "temperature", "high temperature",
            "feels hot"
        ])
        has_lethargy = any(_kw(kw, text_lower) for kw in [
            "difficult to wake", "hard to wake", "won't wake",
            "lethargic", "unresponsive", "unusually sleepy",
            "unusually quiet", "not alert", "floppy",
            "not feeding", "not eating", "poor feeding"
        ])
        
        if is_infant and has_fever and has_lethargy:
            findings.append({
                "type": "PEDIATRIC_EMERGENCY_INFECTIOUS",
                "severity": "CRITICAL",
                "description": "INFANT EMERGENCY: Fever + lethargy/poor responsiveness — possible meningitis or serious systemic infection",
                "recommendation": "EMERGENCY: Immediate medical evaluation. Blood cultures, LP, IV antibiotics. Infants can deteriorate rapidly. Do NOT wait or give home medications."
            })
        
        # --- 5. Pediatric cyanosis (bluish lips) ---
        has_cyanosis = any(_kw(kw, text_lower) for kw in [
            "bluish lips", "blue lips", "cyanotic", "cyanosis",
            "blue around mouth", "blue fingernails", "blue nails",
            "bluish skin", "bluish color"
        ])
        
        if is_pediatric and has_cyanosis:
            findings.append({
                "type": "PEDIATRIC_EMERGENCY_CYANOSIS",
                "severity": "CRITICAL",
                "description": "Pediatric cyanosis (bluish lips) — respiratory or cardiac emergency",
                "recommendation": "IMMEDIATE: High-flow oxygen, assess airway/breathing/circulation. Pulse oximetry. Possible causes: respiratory distress, cardiac disease, poisoning. Call emergency services."
            })
        
        # --- 6. Splitting adult tablets for children ---
        has_split_question = any(_kw(kw, text_lower) for kw in [
            "split", "halve", "cut in half", "break in half",
            "crush", "tablet split"
        ])
        has_adult_med = any(_kw(kw, text_lower) for kw in [
            "adult tablet", "adult dose", "adult strength",
            "my medication", "my tablet"
        ])
        
        if is_pediatric and has_split_question and has_adult_med:
            findings.append({
                "type": "PEDIATRIC_TABLET_SPLITTING",
                "severity": "HIGH",
                "description": "Splitting adult tablet for child — unsafe: uneven halves, incorrect dose, not designed for pediatric use",
                "recommendation": "Do NOT split adult tablets for children. Use pediatric formulations (liquid/suspension). Doses must be weight-based (mg/kg). Consult physician for correct pediatric formulation."
            })
        elif is_pediatric and has_split_question:
            findings.append({
                "type": "PEDIATRIC_TABLET_SPLITTING",
                "severity": "MODERATE",
                "description": "Splitting tablets for pediatric patient — verify this is appropriate for this medication",
                "recommendation": "Some tablets can be split, others cannot (extended-release, enteric-coated). Consult pharmacist."
            })
        
        return findings

    # ── Poisoning / Overdose detection ──
    SEDATIVE_DRUGS = ["zolpidem", "ambien", "eszopiclone", "lunesta", "suvorexant",
                      "lorazepam", "diazepam", "alprazolam", "clonazepam",
                      "temazepam", "midazolam", "chlordiazepoxide",
                      "phenobarbital", "pentobarbital", "secobarbital",
                      "diphenhydramine", "benadryl", "doxepin"]
    
    def check_poisoning_overdose(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect poisoning, overdose, and suicide risk scenarios."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]
        
        # --- 1. Overdose patterns ---
        has_overdose = any(_kw(kw, text_lower) for kw in [
            "too many", "overdose", "overdosed", "large amount",
            "took too much", "took more", "excessive amount",
            "took an extra", "double dose", "triple dose"
        ])
        has_sleeping_pill = any(_kw(kw, text_lower) for kw in [
            "sleeping pill", "sleeping tablets", "sleeping medication",
            "sleeping tablets", "sleeping medicine", "sedative",
            "sleeping drug"
        ])
        
        if has_overdose:
            # Check if it's a sedative/sleeping pill (especially dangerous)
            if has_sleeping_pill or any(d in self.SEDATIVE_DRUGS for d in drugs):
                findings.append({
                    "type": "POISONING_SEDATIVE_OVERDOSE",
                    "severity": "CRITICAL",
                    "description": "SEDATIVE OVERDOSE: Sleeping pills/sedatives cause respiratory depression and can be fatal even when patient feels awake initially",
                    "recommendation": "EMERGENCY: Call Poison Control (1-800-222-1222). Do NOT leave patient alone. Monitor breathing. Flumazenil if benzodiazepine (but NOT if mixed with other drugs). Induce vomiting only if within 1 hour and NOT a sedative."
                })
            else:
                findings.append({
                    "type": "POISONING_GENERAL_OVERDOSE",
                    "severity": "CRITICAL",
                    "description": "OVERDOSE SUSPECTED: Medication taken in excessive amount",
                    "recommendation": "EMERGENCY: Call Poison Control (1-800-222-1222). Bring medication container to ER. Monitor vital signs. Do NOT induce vomiting without medical guidance."
                })
        
        # --- 2. Asymptomatic overdose (feels fine but took too much) ---
        asymptomatic_overdose = any(_kw(kw, text_lower) for kw in [
            "feel okay", "feel fine", "feels fine", "feels okay",
            "awake and feel", "no symptoms", "asymptomatic"
        ])
        if has_overdose and asymptomatic_overdose:
            findings.append({
                "type": "POISONING_DELAYED_TOXICITY",
                "severity": "CRITICAL",
                "description": "OVERDOSE but patient feels fine — DELAYED TOXICITY WARNING: many drug toxicities take hours to manifest",
                "recommendation": "IMMEDIATE: Call Poison Control. Do NOT reassure. Acetaminophen liver damage peaks at 72h. Opioid respiratory depression can be delayed. Do NOT leave patient alone."
            })
        
        # --- 3. Unknown drug + unknown quantity ---
        has_unknown_drug = any(_kw(kw, text_lower) for kw in [
            "unknown", "doesn't know the name", "don't know the name",
            "not sure what", "no name", "without a name",
            "unidentified"
        ])
        has_unknown_quantity = any(_kw(kw, text_lower) for kw in [
            "unknown quantity", "unknown amount", "don't know how much",
            "doesn't know how much", "unsure how much",
            "unknown dose", "don't know how many"
        ])
        
        if has_unknown_drug and has_unknown_quantity:
            findings.append({
                "type": "POISONING_UNKNOWN_INGESTION",
                "severity": "CRITICAL",
                "description": "UNKNOWN DRUG + UNKNOWN QUANTITY ingested — CANNOT assess toxicity or guide treatment",
                "recommendation": "EMERGENCY: Bring ALL medication containers to ER. Call Poison Control. Do NOT induce vomiting (may worsen if caustic). Monitor ABCs. Drug screen, lactate, electrolytes."
            })
        
        # --- 4. Intentional overdose (suicidal) ---
        has_intentional = any(_kw(kw, text_lower) for kw in [
            "intentionally", "deliberately", "on purpose",
            "suicide", "suicidal", "self-harm", "self harm",
            "attempted suicide", "overdose intent"
        ])
        
        if has_intentional and has_overdose:
            findings.append({
                "type": "POISONING_INTENTIONAL",
                "severity": "CRITICAL",
                "description": "INTENTIONAL OVERDOSE — suicide risk. This is a psychiatric emergency in addition to medical emergency",
                "recommendation": "IMMEDIATE: Medical stabilization + psychiatric evaluation. 1:1 observation. Call crisis team. Notify mental health. All medications removed. Document for safeguarding."
            })
        elif has_intentional:
            findings.append({
                "type": "SUICIDE_RISK",
                "severity": "CRITICAL",
                "description": "Suicide risk mentioned — immediate psychiatric evaluation needed",
                "recommendation": "Crisis intervention. 1:1 observation. Remove medications. Mental health referral."
            })
        
        # --- 5. Fatal dose question (DO NOT PROVIDE) ---
        fatal_dose_patterns = [
            r'how\s+much.*?fatal',
            r'fatal\s+dose',
            r'lethal\s+dose',
            r'kill.*?(?:me|someone)',
            r'die\s+from',
            r'death\s+dose',
            r'overdose.*?kill',
            r'how\s+much.*?kill',
            r'how\s+much.*?die',
            r'enough\s+to\s+(?:kill|die)',
        ]
        
        for pattern in fatal_dose_patterns:
            if re.search(pattern, text_lower):
                findings.append({
                    "type": "POISONING_FATAL_DOSE_REFUSAL",
                    "severity": "CRITICAL",
                    "description": "CRITICAL SAFETY ALERT: Patient asking about fatal/lethal dose — system must NOT provide lethal dose information",
                    "recommendation": "DO NOT PROVIDE lethal dose calculations. Instead: If patient is in distress, call emergency services. If patient is suicidal, provide crisis hotline (988 Suicide & Crisis Lifeline). Provide Poison Control number (1-800-222-1222)."
                })
                break
        
        # --- 6. Child + unattended medication (poisoning risk) ---
        is_pediatric = any(_kw(kw, text_lower) for kw in [
            "child", "children", "toddler", "baby", "infant",
            "son", "daughter", "kid"
        ])
        has_unattended = any(_kw(kw, text_lower) for kw in [
            "unattended", "left out", "left behind",
            "accessible", "reached", "found", "got hold of",
            "got into"
        ])
        
        if is_pediatric and (has_unattended or (has_overdose and is_pediatric)):
            findings.append({
                "type": "POISONING_PEDIATRIC_ACCESS",
                "severity": "CRITICAL",
                "description": "Pediatric poisoning risk: child had access to unattended medication",
                "recommendation": "EMERGENCY: Call Poison Control. Bring medication to ER. Child-proof all medications immediately. Symptoms may be delayed."
            })

        return findings

    # ── "Don't Trust the Premise" detection ──
    def check_premise_challenges(self, text: str, medications: List[Dict], allergies: List[str] = None) -> List[Dict]:
        """Detect cases where the user's premise should be challenged."""
        findings = []
        text_lower = text.lower()

        # --- 1. Unconfirmed diagnosis being treated as fact ---
        false_certainty = any(_kw(kw, text_lower) for kw in [
            "definitely has", "certainly has", "obviously has",
            "clearly has", "must have", "has to be",
            "without a doubt", "confirmed"
        ])
        asking_for_treatment = any(_kw(kw, text_lower) for kw in [
            "which antibiotic", "what should i prescribe", "what drug",
            "which medication", "what should they take", "give them",
            "prescribe", "treat with"
        ])

        if false_certainty and asking_for_treatment:
            findings.append({
                "type": "PREMISE_UNCONFIRMED_DIAGNOSIS",
                "severity": "HIGH",
                "description": "Diagnosis stated as definite but may not be confirmed — should not prescribe based on assumption",
                "recommendation": "Confirm diagnosis with appropriate tests (imaging, labs, cultures) BEFORE prescribing. Do not treat presumed diagnoses without diagnostic evidence."
            })

        # --- 2. Age used to dismiss serious symptoms ---
        age_dismissal = any(_kw(kw, text_lower) for kw in [
            "young", "young patient", "young person", "young adult",
            "teenager", "not old", "too young"
        ])
        dismissal_words = any(_kw(kw, text_lower) for kw in [
            "can't be dangerous", "can't be serious", "shouldn't be",
            "unlikely to be", "nothing to worry", "not a concern",
            "benign", "not concerning", "isn't serious"
        ])
        serious_symptom = any(_kw(kw, text_lower) for kw in [
            "headache", "chest pain", "abdominal pain", "seizure",
            "shortness of breath", "bleeding", "weakness", "numbness",
            "vision changes", "fever", "vomiting"
        ])

        if age_dismissal and dismissal_words and serious_symptom:
            findings.append({
                "type": "PREMISE_AGE_DISMISSAL",
                "severity": "HIGH",
                "description": "Age used to dismiss serious symptoms — age does NOT eliminate emergency causes",
                "recommendation": "Young patients can have serious conditions (SAH, meningitis, PE, DKA). Do NOT dismiss symptoms based on age. Investigate appropriately."
            })

        # --- 3. Premature treatment discontinuation ---
        feels_better = any(_kw(kw, text_lower) for kw in [
            "feels better", "feeling better", "feels fine",
            "feeling fine", "symptoms resolved", "symptoms gone",
            "feels well", "feeling well"
        ])
        stopping_medication = any(_kw(kw, text_lower) for kw in [
            "stop", "discontinue", "cease", "quit", "end",
            "not take", "no longer", "don't need"
        ])
        antibiotics_mentioned = any(_kw(kw, text_lower) for kw in [
            "antibiotic", "antibiotics", "course", "full course",
            "treatment", "therapy"
        ])

        if feels_better and stopping_medication and antibiotics_mentioned:
            findings.append({
                "type": "PREMISE_PREMATURE_STOP",
                "severity": "HIGH",
                "description": "Patient wants to stop antibiotics because they feel better — premature discontinuation risks resistance and relapse",
                "recommendation": "Complete the FULL prescribed course of antibiotics. Stopping early promotes antibiotic resistance and may lead to relapse. Only stop if directed by prescriber."
            })
        elif feels_better and stopping_medication:
            findings.append({
                "type": "PREMISE_PREMATURE_STOP",
                "severity": "MODERATE",
                "description": "Patient wants to stop medication because they feel better — consult prescriber before stopping",
                "recommendation": "Do not stop medications without consulting the prescribing physician. Feeling better does not always mean the condition is fully treated."
            })

        # --- 4. Authority bias: doctor prescribed = safe despite allergy ---
        authority_bias = any(_kw(kw, text_lower) for kw in [
            "doctor prescribed", "physician prescribed", "prescribed by doctor",
            "prescribed by a doctor", "the doctor", "a doctor",
            "from a doctor", "came from a doctor"
        ])
        allergy_mention = any(_kw(kw, text_lower) for kw in [
            "allergy", "allergic", "allergies", "hypersensitivity",
            "intolerance", "anaphylaxis", "reaction"
        ])

        if authority_bias and allergy_mention:
            findings.append({
                "type": "PREMISE_AUTHORITY_BIAS",
                "severity": "CRITICAL",
                "description": "Authority bias detected: assuming prescription is safe BECAUSE it came from a doctor — allergies exist regardless of who prescribes",
                "recommendation": "ALWAYS verify medication against patient allergies regardless of prescriber. Allergy checking is an independent safety check. A prescription does not override allergy risk."
            })

        # --- 5. Anxiety used to dismiss chest pain ---
        anxiety_dismissal = any(_kw(kw, text_lower) for kw in [
            "anxiety", "anxious", "panic", "panic attack",
            "stress", "stressed"
        ])
        chest_pain = any(_kw(kw, text_lower) for kw in [
            "chest pain", "chest pressure", "chest tightness",
            "chest discomfort"
        ])
        dismissal_of_chest = any(_kw(kw, text_lower) for kw in [
            "isn't serious", "not serious", "not cardiac",
            "not a heart", "can't be", "so the chest"
        ])

        if anxiety_dismissal and chest_pain and dismissal_of_chest:
            findings.append({
                "type": "PREMISE_SYMPTOM_DISMISSAL",
                "severity": "CRITICAL",
                "description": "Chest pain being dismissed as 'just anxiety' — anxiety does NOT exclude cardiac causes",
                "recommendation": "Chest pain requires cardiac workup even in patients with anxiety history. Many MI patients have prior anxiety diagnoses. ECG, troponin, chest X-ray needed."
            })

        return findings

    # ── Hallucination Prevention ──
    def check_hallucination_prevention(self, text: str, medications: List[Dict], allergies: List[str] = None) -> List[Dict]:
        """Detect requests that would require hallucinated/unsupported clinical claims."""
        findings = []
        text_lower = text.lower()

        # --- 1. Unspecified medication: 'medication X' / 'this drug' without a real name ---
        unspecified_med_patterns = [
            r"medication [x-z]", r"drug [x-z]", r"medicine [x-z]",
            r"this drug", r"the drug", r"that drug",
            r"this medication", r"the medication", r"that medication",
            r"a medication", r"some medication",
        ]
        has_unspecified_med = any(re.search(p, text_lower) for p in unspecified_med_patterns)
        asking_for_alternative = any(_kw(kw, text_lower) for kw in [
            "alternative", "substitute", "instead", "replace",
            "what should i prescribe", "what dose", "how much",
            "what drug", "which drug", "which medication"
        ])
        has_real_drug = any(m.get("drug", "") for m in medications)

        if has_unspecified_med and asking_for_alternative and not has_real_drug:
            findings.append({
                "type": "HALLUCINATION_UNSPECIFIED_MED",
                "severity": "HIGH",
                "description": "Cannot provide alternative dose without identifying the specific medication — 'medication X' is not a real drug name",
                "recommendation": "Identify the specific drug before recommending alternatives. Allergy alternatives depend on drug class, patient factors (age, weight, renal/hepatic function), and clinical context."
            })

        # --- 2. Absolute safety claim: 'safe with all medications' is never true ---
        absolute_safety = any(_kw(phrase, text_lower) for phrase in [
            "safe with all", "safe with every", "safe to take with all",
            "no interactions", "no drug interactions",
            "compatible with all", "works with all",
            "won't interact", "won't cause interactions",
            "completely safe", "totally safe", "100% safe",
            "always safe", "universally safe"
        ])
        asking_for_confirmation = any(_kw(kw, text_lower) for kw in [
            "correct?", "right?", "isn't it?", "true?",
            "is that", "can i", "confirm", "verify"
        ])

        if absolute_safety or (asking_for_confirmation and any(_kw(kw, text_lower) for kw in [
            "safe", "interact", "conflict"
        ])):
            findings.append({
                "type": "HALLUCINATION_ABSOLUTE_SAFETY",
                "severity": "CRITICAL",
                "description": "Absolute safety claims are never valid — every drug has potential interactions and individual patient factors affect safety",
                "recommendation": "Drug safety depends on: specific drug combination, patient genetics (CYP450 polymorphisms), age, renal/hepatic function, other medications, and clinical context. Always check against a comprehensive interaction database."
            })

        # --- 3. Exact treatment without sufficient clinical information ---
        asking_for_exact = any(_kw(kw, text_lower) for kw in [
            "give the exact", "what exact", "tell me the exact",
            "provide the exact", "what is the exact",
            "which antibiotic", "what antibiotic",
            "what dose and duration", "how long should",
            "prescribe", "prescription"
        ])
        has_diagnosis = any(_kw(kw, text_lower) for kw in [
            "pneumonia", "infection", "disease", "condition",
            "syndrome", "disorder", "diagnosis"
        ])

        if asking_for_exact and has_diagnosis:
            findings.append({
                "type": "HALLUCINATION_INSUFFICIENT_INFO",
                "severity": "HIGH",
                "description": "Treatment depends on multiple clinical factors not provided — cannot give 'exact' regimen without patient-specific information",
                "recommendation": "Prescribing requires: patient age/weight, renal/hepatic function, local antibiogram/resistance patterns, allergy history, severity of illness, comorbidities, prior treatment response, and clinical guidelines. One universal regimen does not exist."
            })

        # --- 4. Dosing question with missing critical clinical parameters ---
        asking_for_dose = any(_kw(kw, text_lower) for kw in [
            "what dose", "how much should", "what is the dose",
            "dosing", "dosage", "how many mg", "what strength",
            "what should the dose be"
        ])
        has_organ_impairment = any(_kw(kw, text_lower) for kw in [
            "kidney", "renal", "liver", "hepatic",
            "kidney function", "renal function", "hepatic function",
            "creatinine", "egfr", "gfr", "impaired"
        ])
        missing_drug = not has_real_drug or not any(m.get("dosage", "") for m in medications)

        if asking_for_dose and has_organ_impairment:
            findings.append({
                "type": "HALLUCINATION_MISSING_PARAMS",
                "severity": "HIGH",
                "description": "Dosing with organ impairment requires specific clinical data not provided — cannot safely calculate dose",
                "recommendation": "Renal/hepatic dosing requires: specific drug name, exact GFR or CrCl (or Child-Pugh score), patient weight, indication, and drug-specific dosing guidelines. Impaired organ function can dramatically alter drug clearance."
            })

        return findings

    # ── Privacy & Identity Safety ──
    def check_privacy_identity(self, text: str) -> List[Dict]:
        """Detect privacy, PHI/PII exposure, and record-access safety issues."""
        findings = []
        text_lower = text.lower()

        # --- 1. PHI/PII Exposure: patient-identifying info being exposed or summarized ---
        pii_keywords = [
            "full name", "patient name", "patient's name",
            "address", "phone number", "phone",
            "medical record", "medical record number", "mrn",
            "social security", "ssn", "date of birth", "dob",
            "email address", "insurance number",
            "aadhaar", "pan card"
        ]
        pii_count = sum(1 for kw in pii_keywords if _kw(kw, text_lower))

        exposure_action = any(_kw(kw, text_lower) for kw in [
            "summarize everything", "show me", "display",
            "tell me all", "give me all", "what is",
            "read out", "print", "extract all"
        ])

        if pii_count >= 2 and exposure_action:
            findings.append({
                "type": "PRIVACY_PHI_EXPOSURE",
                "severity": "CRITICAL",
                "description": f"Patient-identifying information (PII/PHI) detected in text ({pii_count} identifiers) — summarizing or exposing this information violates patient privacy",
                "recommendation": "NEVER expose, summarize, or store patient-identifying information (name, address, phone, MRN, SSN, DOB) without proper authorization and HIPAA-compliant safeguards. Minimize PHI exposure to only what is clinically necessary."
            })
        elif pii_count >= 1 and exposure_action:
            findings.append({
                "type": "PRIVACY_PHI_EXPOSURE",
                "severity": "HIGH",
                "description": "Possible patient-identifying information detected — handle with care to avoid unnecessary exposure",
                "recommendation": "Ensure PHI is only shared with authorized personnel on a need-to-know basis. De-identify data when possible."
            })

        # --- 2. Unauthorized record access: 'find history' / 'look up' by name/DOB ---
        record_access = any(_kw(kw, text_lower) for kw in [
            "find this patient", "look up", "search for",
            "pull up", "retrieve", "access",
            "find history", "search history",
            "find medical", "look up medical"
        ])
        identification_method = any(_kw(kw, text_lower) for kw in [
            "using their name", "by name", "name and",
            "date of birth", "dob", "birth date",
            "by their dob", "using dob",
            "patient id", "mrn", "record number"
        ])

        if record_access and identification_method:
            findings.append({
                "type": "PRIVACY_UNAUTHORIZED_ACCESS",
                "severity": "CRITICAL",
                "description": "System does NOT have access to patient medical records — cannot look up patient history by name or identifiers",
                "recommendation": "This system analyzes uploaded prescription text only — it has no access to patient databases, EHR systems, or medical records. Patient lookups require authorized access to hospital EMR systems with proper audit trails."
            })
        elif record_access:
            findings.append({
                "type": "PRIVACY_UNAUTHORIZED_ACCESS",
                "severity": "HIGH",
                "description": "Request assumes access to patient records that this system does not have",
                "recommendation": "This system analyzes uploaded prescription documents only. It cannot access, retrieve, or search patient medical records from any database or EHR system."
            })

        # --- 3. Fabricate prescriptions for other patients ---
        fabricate_request = any(_kw(kw, text_lower) for kw in [
            "another patient", "other patient", "different patient",
            "a patient's", "someone else",
            "give me", "generate", "create",
            "write a prescription", "make a prescription"
        ])
        based_on = any(_kw(kw, text_lower) for kw in [
            "based on their", "based on the",
            "based on diagnosis", "based on condition",
            "based on symptoms",
            "copy their", "same as"
        ])
        has_prescription_intent = any(_kw(kw, text_lower) for kw in [
            "prescription", "prescribe", "medication",
            "drug order", "dose", "regimen"
        ])

        if fabricate_request and (based_on or has_prescription_intent):
            findings.append({
                "type": "PRIVACY_NO_FABRICATION",
                "severity": "CRITICAL",
                "description": "Cannot generate prescriptions for other patients based on shared diagnosis — every patient requires individualized assessment",
                "recommendation": "Prescriptions must be individualized based on: the actual patient's history, allergies, weight, organ function, current medications, and clinical examination. Copying prescriptions between patients is dangerous and unethical. Each patient deserves their own clinical assessment."
            })

        return findings

    # ── Mental Health Safety ──
    def check_mental_health_safety(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Detect dangerous psychiatric medication patterns."""
        findings = []
        text_lower = text.lower()
        drugs = [m.get("drug", "").lower() for m in medications]

        # --- 1. Abrupt benzodiazepine discontinuation ---
        benzo_drugs = ["diazepam", "lorazepam", "alprazolam", "clonazepam",
                       "chlordiazepoxide", "oxazepam", "temazepam", "midazolam",
                       "triazolam"]
        has_benzo = any(d in benzo_drugs for d in drugs)
        has_abrupt_stop = any(_kw(kw, text_lower) for kw in [
            "abruptly stopped", "suddenly stopped", "cold turkey",
            "stopped taking", "discontinued suddenly", "abrupt discontinuation",
            "stopped it", "quit taking", "abruptly discontinued"
        ])
        has_withdrawal = any(_kw(kw, text_lower) for kw in [
            "tremor", "tremors", "seizure", "seizures",
            "anxiety", "insomnia", "hallucination", "delirium",
            "sweating", " agitation"
        ])

        if has_benzo and has_abrupt_stop:
            findings.append({
                "type": "MENTAL_HEALTH_BENZO_WITHDRAWAL",
                "severity": "CRITICAL",
                "description": "ABRUPT BENZODIAZEPINE DISCONTINUATION: Can cause life-threatening seizures, delirium, and psychosis after years of use",
                "recommendation": "NEVER stop benzodiazepines abruptly after prolonged use. Seizures can occur 24-72h after last dose. Gradual taper (10-25% reduction every 1-2 weeks) is mandatory. Emergency medical supervision required."
            })
            if has_withdrawal:
                findings.append({
                    "type": "MENTAL_HEALTH_BENZO_WITHDRAWAL_ACTIVE",
                    "severity": "CRITICAL",
                    "description": "ACTIVE BENZODIAZEPINE WITHDRAWAL SYMPTOMS detected — seizure risk is IMMINENT",
                    "recommendation": "EMERGENCY: Resume benzodiazepine at last effective dose immediately. Stabilize, then begin supervised taper. ICU admission if seizure activity. Do NOT wait for scheduled appointment."
                })

        # --- 2. Abrupt SSRI/SNRI discontinuation ---
        ssri_snri_drugs = ["fluoxetine", "sertraline", "escitalopram", "citalopram",
                           "paroxetine", "fluvoxamine", "venlafaxine", "duloxetine",
                           "desvenlafaxine", "milnacipran", "vilazodone", "vortioxetine"]
        has_ssri_snri = any(d in ssri_snri_drugs for d in drugs)

        if has_ssri_snri and has_abrupt_stop:
            findings.append({
                "type": "MENTAL_HEALTH_SSD_WITHDRAWAL",
                "severity": "HIGH",
                "description": "SSRI/SNRI DISCONTINUATION SYNDROME: Abrupt stop causes dizziness, nausea, brain zaps, irritability, insomnia",
                "recommendation": "SSRI/SNRI should be tapered over 2-4 weeks. Symptoms usually resolve within 1 week but can last months (especially paroxetine, venlafaxine). Resume medication and begin supervised taper."
            })

        # --- 3. Serotonin syndrome risk (serotonergic drug combinations) ---
        serotonergic_drugs = {
            "ssri": ["fluoxetine", "sertraline", "escitalopram", "citalopram",
                      "paroxetine", "fluvoxamine", "vilazodone", "vortioxetine"],
            "snri": ["venlafaxine", "duloxetine", "desvenlafaxine", "milnacipran"],
            "maoi": ["phenelzine", "tranylcypromine", "selegiline", "isocarboxazid"],
            "triptan": ["sumatriptan", "rizatriptan", "zolmitriptan", "naratriptan",
                         "eletriptan", "frovatriptan", "almotriptan"],
            "tramadol": ["tramadol"],
            "linezolid": ["linezolid"],
            "methylthioninium": ["methylene blue"],
            "tca": ["amitriptyline", "nortriptyline", "imipramine", "clomipramine",
                     "doxepin", "desipramine", "trimipramine"],
        }

        serotonergic_categories_found = []
        for cat, cat_drugs in serotonergic_drugs.items():
            if any(d in cat_drugs for d in drugs):
                serotonergic_categories_found.append(cat)

        # MAOI + SSRI/SNRI = always dangerous
        if "maoi" in serotonergic_categories_found and (
            "ssri" in serotonergic_categories_found or
            "snri" in serotonergic_categories_found
        ):
            findings.append({
                "type": "MENTAL_HEALTH_SEROTONIN_SYNDROME",
                "severity": "CRITICAL",
                "description": "MAOI + SSRI/SNRI COMBINATION: Serotonin syndrome is LIKELY and potentially FATAL. Hyperthermia, rigidity, myoclonus, autonomic instability",
                "recommendation": "EMERGENCY: STOP both medications immediately. ICU admission. Cyproheptadine (serotonin antagonist). Cooling measures for hyperthermia. This combination requires a 14-day washout period between medications."
            })
        elif len(serotonergic_categories_found) >= 2 and "maoi" not in serotonergic_categories_found:
            # Multiple serotonergic drugs (without MAOI) = moderate risk
            findings.append({
                "type": "MENTAL_HEALTH_SEROTONIN_RISK",
                "severity": "HIGH",
                "description": f"Multiple serotonergic drugs detected ({', '.join(serotonergic_categories_found)}) — risk of serotonin syndrome",
                "recommendation": "Monitor for serotonin syndrome: agitation, hyperreflexia, myoclonus, diarrhea, hyperthermia. Consider dose adjustment or alternative. Serotonin syndrome can range from mild to life-threatening."
            })

        # --- 4. Clozapine monitoring requirements ---
        has_clozapine = any(d in ["clozapine"] for d in drugs)
        has_monitoring = any(_kw(kw, text_lower) for kw in [
            "blood count", "anc", "absolute neutrophil",
            "monitoring schedule", "regular blood",
            "cloba", "clozapine registry"
        ])
        no_monitoring = any(_kw(kw, text_lower) for kw in [
            "no recent blood", "no blood count", "no monitoring",
            "not monitored", "without monitoring",
            "no anc", "no registry"
        ])

        if has_clozapine and (no_monitoring or not has_monitoring):
            findings.append({
                "type": "MENTAL_HEALTH_CLOZAPINE_MONITORING",
                "severity": "CRITICAL",
                "description": "CLOZAPINE requires mandatory blood count monitoring (ANC) — risk of fatal agranulocytosis (1-2% of patients)",
                "recommendation": "MANDATORY: Weekly ANC for first 6 months, then biweekly for 6 months, then monthly. REMS program enrollment required. If ANC < 1500 — HOLD clozapine. If ANC < 1000 — EMERGENT neutropenia protocol."
            })

        # --- 5. Lithium monitoring requirements ---
        has_lithium = any(d in ["lithium", "lithobid", "lithane"] for d in drugs)
        lithium_toxicity_signs = any(_kw(kw, text_lower) for kw in [
            "increased thirst", "frequent urination", "tremor",
            "vomiting", "confusion", "drowsiness", "slurred speech",
            "muscle weakness", "ataxia", "blurred vision"
        ])
        no_levels = any(_kw(kw, text_lower) for kw in [
            "no lithium level", "levels not checked", "not checked",
            "no recent level", "no level", "without monitoring"
        ])

        if has_lithium and (no_levels or lithium_toxicity_signs):
            severity = "CRITICAL" if lithium_toxicity_signs else "HIGH"
            desc = "LITHIUM TOXICITY SIGNS detected" if lithium_toxicity_signs else "Lithium prescribed without documented level monitoring"
            findings.append({
                "type": "MENTAL_HEALTH_LITHIUM_MONITORING",
                "severity": severity,
                "description": f"{desc} — lithium has a narrow therapeutic index (0.6-1.2 mEq/L)",
                "recommendation": "Lithium levels must be checked every 3-6 months (or more frequently if dose changes, dehydration, new medications, or toxicity signs). Target level: 0.6-0.8 mEq/L for maintenance. Signs of toxicity: tremor, vomiting, confusion, renal impairment."
            })

        # --- 6. Suicidal ideation in medication context ---
        has_suicidal_ideation = any(_kw(kw, text_lower) for kw in [
            "suicidal thoughts", "suicidal ideation", "wants to die",
            "wants to kill", "self-harm", "self harm",
            "thoughts of death", "end my life", "end their life",
            "no reason to live", "better off dead",
            "increased suicidal", "suicide plan"
        ])
        has_ssri = any(d in ssri_snri_drugs for d in drugs)
        has_young_patient = any(_kw(kw, text_lower) for kw in [
            "adolescent", "teenager", "young adult", "14", "15", "16",
            "17", "18", "19", "20", "21", "22", "24"
        ])

        if has_suicidal_ideation:
            if has_ssri:
                findings.append({
                    "type": "MENTAL_HEALTH_SSRI_SUICIDE_RISK",
                    "severity": "CRITICAL",
                    "description": "SSRI + SUICIDAL IDEATION: FDA black box warning — increased suicidality risk in first weeks of SSRI treatment, especially in young adults",
                    "recommendation": "Crisis intervention. Close monitoring (weekly visits for first 4 weeks). Consider dose adjustment. Safety plan. Remove access to lethal means. 1:1 observation if acute risk. Consult psychiatry."
                })
            else:
                findings.append({
                    "type": "MENTAL_HEALTH_SUICIDAL_IDEATION",
                    "severity": "CRITICAL",
                    "description": "Suicidal ideation detected — immediate psychiatric evaluation and safety assessment required",
                    "recommendation": "Crisis intervention. Safety plan. Remove access to lethal means. Consider hospitalization if active plan/intent. 988 Suicide & Crisis Lifeline. Document risk level."
                })

        return findings

    # ── Complex Multi-Drug Interactions ──
    def check_complex_interactions(self, medications: List[Dict]) -> List[Dict]:
        """Detect dangerous multi-drug combinations beyond pairwise interactions."""
        findings = []
        drugs = [m.get("drug", "").lower() for m in medications]

        # --- 1. QT Prolongation: 2+ QT-prolonging drugs = risk of Torsades de Pointes ---
        qt_prolonging_drugs = {
            "class_ia": ["quinidine", "procainamide", "disopyramide"],
            "class_iii": ["amiodarone", "sotalol", "dofetilide", "ibutilide", "dronedarone"],
            "antibiotics": ["azithromycin", "clarithromycin", "erythromycin", "levofloxacin",
                           "moxifloxacin", "ciprofloxacin"],
            "antipsychotics": ["haloperidol", "ziprasidone", "thioridazine", "pimozide",
                              "risperidone", "quetiapine"],
            "antidepressants": ["citalopram", "escitalopram", "amitriptyline", "nortriptyline",
                               "imipramine", "venlafaxine"],
            "antiemetics": ["ondansetron", "granisetron", "droperidol"],
            "antifungals": ["fluconazole", "ketoconazole", "voriconazole"],
            "antimalarials": ["hydroxychloroquine", "chloroquine", "mefloquine"],
        }

        qt_drugs_found = []
        for cat, cat_drugs in qt_prolonging_drugs.items():
            for d in drugs:
                if d in cat_drugs:
                    qt_drugs_found.append((d, cat))

        if len(qt_drugs_found) >= 3:
            drug_names = [f"{d} ({c})" for d, c in qt_drugs_found]
            findings.append({
                "type": "INTERACTION_QT_PROLONGATION_TRIPLE",
                "severity": "CRITICAL",
                "description": f"TRIPLE QT PROLONGATION RISK: {len(qt_drugs_found)} QT-prolonging drugs detected: {', '.join(drug_names)}. Risk of Torsades de Pointes and fatal arrhythmia",
                "recommendation": "EMERGENCY: Obtain STAT ECG with QTc measurement. If QTc >500ms — HOLD highest-risk drug. Electrolyte check (K+ >4.0, Mg2+ >2.0). Consider alternatives. Continuous cardiac monitoring."
            })
        elif len(qt_drugs_found) >= 2:
            drug_names = [f"{d} ({c})" for d, c in qt_drugs_found]
            findings.append({
                "type": "INTERACTION_QT_PROLONGATION",
                "severity": "HIGH",
                "description": f"QT PROLONGATION RISK: {len(qt_drugs_found)} QT-prolonging drugs: {', '.join(drug_names)}. Increased risk of Torsades de Pointes",
                "recommendation": "Obtain ECG with QTc. Check electrolytes. If QTc >470ms (men) or >480ms (women) — consider drug substitution. Avoid additional QT-prolonging drugs."
            })

        # --- 2. Triple Nephrotoxicity: ACE inhibitor + NSAID + Diuretic = "Triple Whammy" ---
        has_ace_or_arb = any(d in [
            "lisinopril", "enalapril", "ramipril", "benazepril", "captopril",
            "quinapril", "perindopril", "losartan", "valsartan", "irbesartan",
            "candesartan", "olmesartan", "telmisartan"
        ] for d in drugs)
        has_nsaid = any(d in [
            "ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam",
            "indomethacin", "ketorolac", "piroxicam", "aspirin"
        ] for d in drugs)
        has_diuretic = any(d in [
            "furosemide", "hydrochlorothiazide", "hctz", "metolazone",
            "spironolactone", "torsemide", "bumetanide", "amiloride",
            "triamterene", "chlorthalidone", "indapamide"
        ] for d in drugs)

        if has_ace_or_arb and has_nsaid and has_diuretic:
            findings.append({
                "type": "INTERACTION_TRIPLE_NEPHROTOXICITY",
                "severity": "CRITICAL",
                "description": "TRIPLE WHAMMY: ACE inhibitor + NSAID + Diuretic combination causes acute kidney injury in up to 31% of patients",
                "recommendation": "AVOID this combination. NSAID reduces renal perfusion, ACE inhibitor reduces efferent arteriolar tone, diuretic reduces volume — all 3 hit the kidney simultaneously. Use acetaminophen instead of NSAID. Monitor creatinine and potassium closely."
            })
        elif has_ace_or_arb and has_nsaid:
            findings.append({
                "type": "INTERACTION_ACE_NSAID",
                "severity": "HIGH",
                "description": "ACE inhibitor + NSAID: Reduced antihypertensive effect + increased renal risk",
                "recommendation": "Monitor blood pressure and renal function. Use lowest NSAID dose for shortest duration. Consider acetaminophen."
            })

        # --- 3. CYP3A4 Inhibitor + High-Risk Statin (rhabdomyolysis) ---
        cyp3a4_inhibitors = [
            "clarithromycin", "erythromycin", "azole antifungals",
            "fluconazole", "ketoconazole", "voriconazole", "posaconazole",
            "ritonavir", "lopinavir", "atazanavir", "darunavir",
            "diltiazem", "verapamil", "grapefruit",
            "ciprofloxacin"
        ]
        high_risk_statins = ["simvastatin", "lovastatin", "atorvastatin"]
        has_cyp3a4_inhibitor = any(d in cyp3a4_inhibitors for d in drugs)
        has_high_risk_statin = any(d in high_risk_statins for d in drugs)

        if has_cyp3a4_inhibitor and has_high_risk_statin:
            statin_name = next(d for d in drugs if d in high_risk_statins)
            inhibitor_name = next(d for d in drugs if d in cyp3a4_inhibitors)
            findings.append({
                "type": "INTERACTION_CYP3A4_STATIN",
                "severity": "CRITICAL",
                "description": f"CYP3A4 INHIBITOR ({inhibitor_name}) + STATIN ({statin_name}): Increased statin levels can cause rhabdomyolysis (muscle breakdown → kidney failure)",
                "recommendation": "HOLD statin during antibiotic/antifungal course, OR use a non-CYP3A4-metabolized statin (rosuvastatin, pravastatin). Monitor for muscle pain, tenderness, dark urine (CK levels). Rhabdomyolysis is a medical emergency."
            })

        # --- 4. Hyperkalemia Triple Threat: ACE + K-sparing diuretic + K supplement ---
        has_k_sparing = any(d in [
            "spironolactone", "eplerenone", "amiloride", "triamterene"
        ] for d in drugs)
        has_k_supplement = any(d in [
            "potassium chloride", "potassium citrate", "potassium bicarbonate",
            "k-dur", "klor-con", "k-Tab"
        ] for d in drugs)

        if has_ace_or_arb and has_k_sparing and has_k_supplement:
            findings.append({
                "type": "INTERACTION_HYPERKALEMIA_TRIPLE",
                "severity": "CRITICAL",
                "description": "HYPERKALEMIA TRIPLE THREAT: ACE inhibitor + potassium-sparing diuretic + potassium supplement — fatal hyperkalemia risk",
                "recommendation": "REMOVE potassium supplement immediately. Check K+ level STAT. If K+ >6.0 — emergency treatment (calcium gluconate, insulin+glucose, kayexalate). Monitor ECG for peaked T-waves."
            })
        elif has_ace_or_arb and has_k_sparing:
            findings.append({
                "type": "INTERACTION_HYPERKALEMIA",
                "severity": "HIGH",
                "description": "ACE inhibitor + potassium-sparing diuretic: Increased hyperkalemia risk",
                "recommendation": "Monitor potassium levels regularly. Do NOT add potassium supplements. Target K+ <5.0 mEq/L."
            })

        # --- 5. Serotonin syndrome: enhance with tramadol/linezolid ---
        serotonergic_triples = [
            ("ssri", "maoi", "tramadol"),
            ("ssri", "maoi", "linezolid"),
            ("snri", "maoi", "tramadol"),
            ("ssri", "tramadol", "linezolid"),
            ("ssri", "tramadol", "triptan"),
            ("snri", "tramadol", "linezolid"),
        ]
        ssri_drugs = ["fluoxetine", "sertraline", "escitalopram", "citalopram",
                      "paroxetine", "fluvoxamine", "vilazodone", "vortioxetine"]
        snri_drugs = ["venlafaxine", "duloxetine", "desvenlafaxine", "milnacipran"]
        maoi_drugs = ["phenelzine", "tranylcypromine", "selegiline", "isocarboxazid"]
        triptan_drugs = ["sumatriptan", "rizatriptan", "zolmitriptan", "naratriptan",
                         "eletriptan", "frovatriptan", "almotriptan"]

        drug_sets = {
            "ssri": ssri_drugs, "snri": snri_drugs, "maoi": maoi_drugs,
            "tramadol": ["tramadol"], "linezolid": ["linezolid"],
            "triptan": triptan_drugs
        }

        found_cats = set()
        for cat, cat_drugs in drug_sets.items():
            if any(d in cat_drugs for d in drugs):
                found_cats.add(cat)

        for combo in serotonergic_triples:
            if all(cat in found_cats for cat in combo):
                combo_names = " + ".join(combo)
                # Don't duplicate if already flagged by mental health safety
                already_flagged = any(
                    f["type"] == "MENTAL_HEALTH_SEROTONIN_SYNDROME"
                    for f in findings
                )
                if not already_flagged:
                    findings.append({
                        "type": "INTERACTION_SEROTONIN_TRIPLE",
                        "severity": "CRITICAL",
                        "description": f"SEROTONIN SYNDROME TRIPLE THREAT ({combo_names}): All 3 serotonergic categories present — life-threatening serotonin syndrome",
                        "recommendation": "EMERGENCY: Stop all serotonergic agents. ICU admission. Cyproheptadine. Cooling for hyperthermia. Monitor for rigidity, myoclonus, autonomic instability."
                    })
                break

        # --- 6. Warfarin + NSAID + Antibiotic: triple bleeding risk ---
        has_warfarin = any(d in ["warfarin", "coumadin"] for d in drugs)
        has_antibiotic = any(d in [
            "ciprofloxacin", "levofloxacin", "metronidazole", "fluconazole",
            "trimethoprim-sulfamethoxazole", "bactrim"
        ] for d in drugs)

        if has_warfarin and has_nsaid and has_antibiotic:
            abx_name = next(d for d in drugs if d in [
                "ciprofloxacin", "levofloxacin", "metronidazole", "fluconazole",
                "trimethoprim-sulfamethoxazole", "bactrim"
            ])
            findings.append({
                "type": "INTERACTION_WARFARIN_TRIPLE_BLEEDING",
                "severity": "CRITICAL",
                "description": f"WARFARIN + NSAID + ANTIBIOTIC ({abx_name}): Triple bleeding risk — NSAID causes GI erosion, antibiotic alters INR, warfarin anticoagulates",
                "recommendation": "Avoid NSAID with warfarin (use acetaminophen). Monitor INR within 3-5 days of starting antibiotic. Watch for bleeding signs: dark stools, bruising, blood in urine."
            })

        return findings

    TERATOGENS = {
        "isotretinoin", "acitretin", "methotrexate", "thalidomide", "valproic acid", "valproate", "mycophenolate",
        "leflunomide", "misoprostol", "ribavirin", "bosentan", "ambrisentan", "warfarin", "finasteride", "dutasteride",
    }

    def check_reproductive_age_teratogens(self, text: str, medications: List[Dict]) -> List[Dict]:
        """Flag known teratogens for female patients aged 12-50 (e.g. 'Age/Sex: 24 / F')."""
        if not text:
            return []
        m = re.search(r"(\d{1,2})\s*(?:years?|yrs?|y)?\s*/\s*(f|female)\b", text.lower())
        if not m:
            return []
        age = int(m.group(1))
        if not 12 <= age <= 50:
            return []
        findings = []
        for med in medications:
            d = (med.get("drug") or "").lower()
            if d in self.TERATOGENS:
                findings.append({
                    "type": "TERATOGEN_CHILDBEARING_AGE",
                    "severity": "HIGH",
                    "drug": d,
                    "description": f"{d.title()} can cause birth defects and the patient is a {age}-year-old woman",
                    "recommendation": "Confirm a negative pregnancy test and reliable contraception before dispensing; counsel on teratogenic risk.",
                })
        return findings

    def check_duplicate_medications(self, medications: List[Dict]) -> List[Dict]:
        """
        Check for duplicate medications.
        
        Args:
            medications: List of dicts with 'drug' key
            
        Returns:
            List of duplicate findings
        """
        findings = []
        
        seen_drugs = {}
        for med in medications:
            drug = med.get("drug", "").lower().strip()
            
            if drug in seen_drugs:
                findings.append({
                    "type": "DUPLICATE_MEDICATION",
                    "severity": "MODERATE",
                    "drug": drug,
                    "description": f"Duplicate medication detected: {drug}",
                    "recommendation": "Verify if intentional or remove duplicate"
                })
            else:
                seen_drugs[drug] = med

        return findings
    
    def check_condition_contraindications(self, drugs: List[str], conditions: List[str]) -> List[Dict]:
        """
        Check for drug-condition contraindications.
        """
        findings = []
        if not drugs or not conditions:
            return findings
        
        drugs_lower = [d.lower().strip() for d in drugs if d]
        conditions_lower = [c.lower().strip() for c in conditions if c]
        
        for condition in conditions_lower:
            for cond_key, contraindication in CONDITION_CONTRAINDICATIONS.items():
                # Handle both dict and list formats
                if isinstance(contraindication, list):
                    contraindicated_drugs = contraindication
                    reason = "Potential contraindication"
                    recommendation = "Consult physician"
                elif isinstance(contraindication, dict):
                    contraindicated_drugs = contraindication.get("drugs", [])
                    reason = contraindication.get("reason", "Potential contraindication")
                    recommendation = contraindication.get("recommendation", "Consult physician")
                else:
                    continue
                
                # Normalize underscores/spaces for matching
                cond_key_norm = cond_key.replace("_", " ").lower()
                condition_norm = condition.replace("_", " ").lower()
                
                # Also handle abbreviations and synonyms
                condition_synonyms = {
                    "ckd": "chronic kidney disease",
                    "chf": "heart failure",
                    "copd": "chronic obstructive pulmonary disease",
                    "dvt": "deep vein thrombosis",
                    "pe": "pulmonary embolism",
                    "gerd": "gastroesophageal reflux disease",
                    "ra": "rheumatoid arthritis",
                    "sle": "systemic lupus erythematosus",
                    "uti": "urinary tract infection",
                }
                condition_expanded = condition_synonyms.get(condition_norm, condition_norm)
                
                if (condition_norm in cond_key_norm or cond_key_norm in condition_norm or
                    condition_expanded in cond_key_norm or cond_key_norm in condition_expanded):
                    for drug in drugs_lower:
                        for contra_drug in contraindicated_drugs:
                            contra_norm = contra_drug.lower().replace("_", "")
                            drug_norm = drug.replace(" ", "")
                            if (drug_norm in contra_norm or contra_norm in drug_norm or
                                drug in contra_drug or contra_drug in drug):
                                findings.append({
                                    "type": "CONDITION_CONTRAINDICATION",
                                    "severity": "HIGH",
                                    "drug": drug,
                                    "condition": condition,
                                    "description": reason,
                                    "recommendation": recommendation,
                                })
        
        return findings
    
    def _parse_dosage(self, dosage_str: str) -> float:
        """
        Parse dosage string to number in mg.
        
        Examples:
          '500mg' -> 500.0
          '1.5g'  -> 1500.0
          '1kg'   -> 1000000.0  (1 kg = 1000 g = 1,000,000 mg)
          '500mcg' -> 0.5
        """
        if not dosage_str:
            return 0.0
        
        # Remove spaces
        dosage_str = dosage_str.replace(" ", "")
        
        # Try to extract number + unit (handle kg before g)
        match = re.match(r'(\d+(?:\.\d+)?)\s*(kg|g|mg|mcg|ml|units?|iu|meq|%)?', dosage_str, re.IGNORECASE)
        
        if match:
            value = float(match.group(1))
            unit = (match.group(2) or "mg").lower()
            
            # Convert to mg
            if unit == "kg":
                value *= 1000000  # 1 kg = 1000 g = 1,000,000 mg
            elif unit == "g":
                value *= 1000
            elif unit == "mcg":
                value /= 1000
            # mg, ml, units, iu, meq, % — keep as-is (treat as mg equivalent)
            
            return value
        
        return 0.0
    
    def analyze_prescription(self, medications: List[Dict], allergies: List[str] = None,
                             conditions: List[str] = None, text: str = "") -> Dict:
        """
        Complete analysis of a prescription for anomalies.
        """
        all_findings = []
        
        if not medications and not text:
            return {
                "findings": [], "total_anomalies": 0,
                "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0},
                "has_critical": False, "has_high": False, "safe_to_proceed": True,
            }
        
        drugs = [m.get("drug", "") for m in medications if m.get("drug")]
        
        interaction_findings = self.check_drug_interactions(drugs)
        all_findings.extend(interaction_findings)
        
        if allergies:
            allergy_findings = self.check_drug_allergies(drugs, allergies)
            all_findings.extend(allergy_findings)
        
        dosage_findings = self.check_dosage_anomalies(medications, text)
        all_findings.extend(dosage_findings)
        
        duplicate_findings = self.check_duplicate_medications(medications)
        all_findings.extend(duplicate_findings)
        
        # Therapeutic class duplication
        class_dup_findings = self.check_therapeutic_class_duplication(medications)
        all_findings.extend(class_dup_findings)
        
        if conditions:
            contraindication_findings = self.check_condition_contraindications(drugs, conditions)
            all_findings.extend(contraindication_findings)
        
        if text:
            all_findings.extend(self.check_reproductive_age_teratogens(text, medications))

        # Allergy context analysis (tolerance, uncertainty, severity)
        if allergies and text:
            allergy_context_findings = self.check_allergy_context(text, allergies)
            all_findings.extend(allergy_context_findings)
        
        # Adverse reaction detection (requires text)
        if text:
            reaction_findings = self.check_adverse_reactions(text)
            all_findings.extend(reaction_findings)
        
        # Vital sign analysis (requires text + medications)
        if text:
            vital_findings = self.check_vital_signs(text, medications)
            all_findings.extend(vital_findings)
        
        # Overdose patterns (requires text)
        if text:
            overdose_findings = self.check_overdose_patterns(text, medications)
            all_findings.extend(overdose_findings)
        
        # Pediatric concerns (requires text)
        if text:
            pediatric_findings = self.check_pediatric_concerns(text, medications)
            all_findings.extend(pediatric_findings)
        
        # Anticoagulant-specific safety (requires text)
        if text:
            anticoag_findings = self.check_anticoagulant_safety(text, medications)
            all_findings.extend(anticoag_findings)
        
        # Cardiac emergency detection (requires text)
        if text:
            cardiac_findings = self.check_cardiac_emergencies(text, medications)
            all_findings.extend(cardiac_findings)
        
        # Neurological emergency detection (requires text)
        if text:
            neuro_findings = self.check_neuro_emergencies(text, medications)
            all_findings.extend(neuro_findings)
        
        # Respiratory emergency detection (requires text)
        if text:
            resp_findings = self.check_respiratory_emergencies(text, medications)
            all_findings.extend(resp_findings)
        
        # Diabetes safety detection (requires text)
        if text:
            diabetes_findings = self.check_diabetes_safety(text, medications)
            all_findings.extend(diabetes_findings)
        
        # Pregnancy safety detection (requires text)
        if text:
            pregnancy_findings = self.check_pregnancy_safety(text, medications, conditions)
            all_findings.extend(pregnancy_findings)
        
        # Lab value emergency detection (requires text)
        if text:
            lab_findings = self.check_lab_value_emergencies(text, medications)
            all_findings.extend(lab_findings)
        
        # Pediatric safety detection (requires text)
        if text:
            peds_findings = self.check_pediatric_safety(text, medications)
            all_findings.extend(peds_findings)
        
        # Poisoning / overdose detection (requires text)
        if text:
            poison_findings = self.check_poisoning_overdose(text, medications)
            all_findings.extend(poison_findings)

        # Mental health safety (requires text)
        if text:
            mental_findings = self.check_mental_health_safety(text, medications)
            all_findings.extend(mental_findings)

        # Complex multi-drug interactions (no text needed, just medication list)
        complex_findings = self.check_complex_interactions(medications)
        all_findings.extend(complex_findings)

        # Don't Trust the Premise detection (requires text)
        if text:
            premise_findings = self.check_premise_challenges(text, medications, allergies)
            all_findings.extend(premise_findings)

        # Hallucination prevention (requires text)
        if text:
            halluc_findings = self.check_hallucination_prevention(text, medications, allergies)
            all_findings.extend(halluc_findings)

        # Privacy & identity safety (requires text)
        if text:
            privacy_findings = self.check_privacy_identity(text)
            all_findings.extend(privacy_findings)

        # Incomplete prescription detection
        incomplete_findings = self.check_incomplete_prescription(text, medications)
        all_findings.extend(incomplete_findings)
        
        # Deduplicate findings
        seen = set()
        unique_findings = []
        for f in all_findings:
            key = (f.get("type"), f.get("drug", ""), f.get("symptom", ""))
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
        
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        for finding in unique_findings:
            severity = finding.get("severity", "LOW")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "findings": unique_findings,
            "total_anomalies": len(unique_findings),
            "severity_counts": severity_counts,
            "has_critical": severity_counts["CRITICAL"] > 0,
            "has_high": severity_counts["HIGH"] > 0,
            "safe_to_proceed": severity_counts["CRITICAL"] == 0 and severity_counts["HIGH"] == 0,
        }
    
    def format_findings(self, analysis: Dict) -> str:
        """
        Format analysis results for display.
        """
        if analysis["total_anomalies"] == 0:
            return "No anomalies detected. Prescription appears safe."
        
        lines = []
        lines.append(f"Found {analysis['total_anomalies']} potential issue(s):\n")
        
        # Group by severity
        for severity in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            severity_findings = [f for f in analysis["findings"] if f["severity"] == severity]
            
            if severity_findings:
                lines.append(f"\n{'='*50}")
                lines.append(f"{severity} SEVERITY ({len(severity_findings)} issues)")
                lines.append(f"{'='*50}")
                
                for finding in severity_findings:
                    lines.append(f"\nType: {finding['type']}")
                    lines.append(f"Description: {finding['description']}")
                    lines.append(f"Recommendation: {finding['recommendation']}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    detector = AnomalyDetector()
    
    # Test medications
    medications = [
        {"drug": "warfarin", "dosage": "5mg"},
        {"drug": "aspirin", "dosage": "81mg"},
        {"drug": "ibuprofen", "dosage": "400mg"}
    ]
    
    allergies = ["penicillin"]
    conditions = ["hypertension", "diabetes"]
    
    analysis = detector.analyze_prescription(medications, allergies, conditions)
    print(detector.format_findings(analysis))

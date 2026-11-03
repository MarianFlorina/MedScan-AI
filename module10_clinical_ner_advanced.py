"""
Module 10: Advanced Clinical NER & Relation Extraction
======================================================
Provides AI-powered clinical entity linking and relationship extraction.

Uses a combination of:
1. Pre-trained biomedical NER (d4data/biomedical-ner-all) for entity extraction
2. Rule-based entity linking to map entities to medical ontologies
3. Clinical relationship extraction for drug-condition-allergy triads

This module enhances the existing module2_ner.py by providing:
- Entity normalization to standard drug names
- Confidence-weighted relationship scoring
- Clinical context-aware entity disambiguation
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from typing import List, Dict, Optional, Tuple
import re


# ============================================================
# Drug Ontology Mapping
# ============================================================

# Maps common brand names and synonyms to generic drug names
DRUG_ONTOLOGY = {
    # Analgesics
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aleve": "naproxen", "naprosyn": "naproxen",
    "voltaren": "diclofenac", "cataflam": "diclofenac",
    "celebrex": "celecoxib",
    
    # Antibiotics
    "amoxil": "amoxicillin", "trimox": "amoxicillin", "augmentin": "amoxicillin-clavulanate",
    "zithromax": "azithromycin", "z-pack": "azithromycin",
    "cipro": "ciprofloxacin", "ciproxin": "ciprofloxacin",
    "levaquin": "levofloxacin",
    "bactrim": "trimethoprim-sulfamethoxazole", "septra": "trimethoprim-sulfamethoxazole",
    "keflex": "cephalexin",
    
    # Cardiovascular
    "lisinopril": "lisinopril", "prinivil": "lisinopril", "zestril": "lisinopril",
    "norvasc": "amlodipine",
    "lopressor": "metoprolol", "toprol-xl": "metoprolol",
    "lipitor": "atorvastatin", "crestor": "rosuvastatin", "zocor": "simvastatin",
    "coumadin": "warfarin",
    "plavix": "clopidogrel",
    "enalapril": "enalapril", "vasotec": "enalapril",
    "losartan": "losartan", "cozaar": "losartan",
    "diovan": "valsartan",
    
    # Diabetes
    "glucophage": "metformin",
    "glipizide": "glipizide", "glucotrol": "glipizide",
    "januvia": "sitagliptin",
    "jardiance": "empagliflozin",
    "farxiga": "dapagliflozin",
    "ozempic": "semaglutide", "wegovy": "semaglutide",
    "trulicity": "dulaglutide",
    "humalog": "insulin lispro", "novolog": "insulin aspart",
    "lantus": "insulin glargine", "levemir": "insulin detemir",
    
    # GI
    "prilosec": "omeprazole", "nexium": "esomeprazole",
    "protonix": "pantoprazole",
    "pepcid": "famotidine", "zantac": "ranitidine",
    
    # Psychiatric
    "prozac": "fluoxetine", "paxil": "paroxetine", "zoloft": "sertraline",
    "celexa": "citalopram", "lexapro": "escitalopram",
    "xanax": "alprazolam", "valium": "diazepam", "ativan": "lorazepam",
    "klonopin": "clonazepam",
    "wellbutrin": "bupropion",
    "cymbalta": "duloxetine", "effexor": "venlafaxine",
    "lithobid": "lithium", "depakote": "valproic_acid",
    "tegretol": "carbamazepine", "neurontin": "gabapentin",
    "lyrica": "pregabalin",
    
    # Respiratory
    "albuterol": "albuterol", "ventolin": "albuterol", "proair": "albuterol",
    "singulair": "montelukast",
    "flovent": "fluticasone", "advair": "fluticasone-salmeterol",
    "prednisone": "prednisone", "deltasone": "prednisone",
    
    # Thyroid
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine", "unithroid": "levothyroxine",
    
    # Pain/Neuro
    "oxycontin": "oxycodone", "percocet": "oxycodone-acetaminophen",
    "vicodin": "hydrocodone-acetaminophen",
    "tramal": "tramadol", "ultram": "tramadol",
    "topamax": "topiramate",
    
    # Dermatology
    "accutane": "isotretinoin", "retin-a": "tretinoin",
    
    # Urology
    "flomax": "tamsulosin",
    "viagra": "sildenafil", "cialis": "tadalafil",
}

# Maps allergy terms to drug categories for cross-reactivity checking
ALLERGY_ONTOLOGY = {
    "penicillin": "penicillin_allergy",
    "penicillins": "penicillin_allergy",
    "pcn": "penicillin_allergy",
    "amoxicillin allergy": "penicillin_allergy",
    "sulfa": "sulfa_allergy",
    "sulfonamides": "sulfa_allergy",
    "bactrim allergy": "sulfa_allergy",
    "nsaid": "nsaid_allergy",
    "nsaids": "nsaid_allergy",
    "aspirin": "aspirin_allergy",
    "ibuprofen allergy": "nsaid_allergy",
    "codeine": "opioid_allergy",
    "morphine allergy": "opioid_allergy",
    "latex": "latex_allergy",
    "contrast": "contrast_dye_allergy",
    "iodine": "contrast_dye_allergy",
    "cephalosporin": "cephalosporin_allergy",
    "cephalosporins": "cephalosporin_allergy",
    "statin": "statin_allergy",
    "statins": "statin_allergy",
    "ace inhibitor": "ace_inhibitor_allergy",
    "ace inhibitors": "ace_inhibitor_allergy",
    "arb": "arb_allergy",
    "arbs": "arb_allergy",
    "local anesthetic": "local_anesthetic_allergy",
    "lidocaine allergy": "local_anesthetic_allergy",
}


# ============================================================
# Entity Normalization
# ============================================================

def normalize_drug_name(name: str) -> str:
    """
    Normalize a drug name to its generic/standard form.
    
    Handles:
    - Brand name → generic name mapping
    - Common abbreviations
    - Case normalization
    - Hyphenated combination drugs
    """
    name_lower = name.lower().strip()
    
    # Direct ontology lookup
    if name_lower in DRUG_ONTOLOGY:
        return DRUG_ONTOLOGY[name_lower]
    
    # Try without common prefixes
    for prefix in ["tab ", "tab. ", "inj ", "inj. ", "cap ", "cap. ", "syrup ", "drop "]:
        if name_lower.startswith(prefix):
            stripped = name_lower[len(prefix):]
            if stripped in DRUG_ONTOLOGY:
                return DRUG_ONTOLOGY[stripped]
    
    # Return as-is if no mapping found
    return name_lower


def normalize_allergy_name(name: str) -> str:
    """Normalize an allergy name to its standard category."""
    name_lower = name.lower().strip()
    
    if name_lower in ALLERGY_ONTOLOGY:
        return ALLERGY_ONTOLOGY[name_lower]
    
    # Check if it contains a known allergy term
    for term, category in ALLERGY_ONTOLOGY.items():
        if term in name_lower or name_lower in term:
            return category
    
    return name_lower


# ============================================================
# Clinical Relationship Extraction
# ============================================================

# Drug-condition treatment relationships
DRUG_CONDITION_TREATMENTS = {
    "amoxicillin": ["pneumonia", "bronchitis", "sinusitis", "otitis media", "uti", "h. pylori",
                    "dental infection", "streptococcal pharyngitis", "endocarditis prophylaxis"],
    "metformin": ["type 2 diabetes", "diabetes", "pcos", "prediabetes", "insulin resistance"],
    "lisinopril": ["hypertension", "heart failure", "diabetic nephropathy", "post-mi"],
    "atorvastatin": ["hyperlipidemia", "high cholesterol", "cholesterol", "cardiovascular prevention"],
    "omeprazole": ["gerd", "gastric ulcer", "duodenal ulcer", "barrett's esophagus", "h. pylori"],
    "warfarin": ["atrial fibrillation", "afib", "dvt", "pe", "mechanical heart valve"],
    "levothyroxine": ["hypothyroidism", "hypothyroid", "thyroid cancer"],
    "amlodipine": ["hypertension", "angina"],
    "metoprolol": ["hypertension", "angina", "heart failure", "arrhythmia", "post-mi"],
    "aspirin": ["cardiovascular prevention", "pain", "fever", "inflammation"],
    "prednisone": ["inflammation", "autoimmune", "asthma", "copd exacerbation", "allergic reaction"],
    "sertraline": ["depression", "anxiety", "ocd", "ptsd", "panic disorder"],
    "gabapentin": ["neuropathic pain", "seizures", "restless leg syndrome", "fibromyalgia"],
    "pantoprazole": ["gerd", "gastric ulcer", "esophagitis"],
    "ciprofloxacin": ["uti", "pneumonia", "sinusitis", "bone infection"],
    "azithromycin": ["pneumonia", "bronchitis", "sinusitis", "std", "chlamydia"],
    "acetaminophen": ["pain", "fever", "headache"],
    "ibuprofen": ["pain", "inflammation", "fever", "arthritis"],
    "naproxen": ["pain", "inflammation", "arthritis", "gout"],
    "losartan": ["hypertension", "diabetic nephropathy"],
    "simvastatin": ["hyperlipidemia", "high cholesterol"],
    "hydrochlorothiazide": ["hypertension", "edema", "heart failure"],
    "furosemide": ["edema", "heart failure", "pulmonary edema"],
    "spironolactone": ["heart failure", "edema", "hypertension", "ascites"],
    "clopidogrel": ["cardiovascular prevention", "stent", "post-mi", "stroke prevention"],
    "duloxetine": ["depression", "anxiety", "neuropathic pain", "fibromyalgia"],
    "escitalopram": ["depression", "anxiety", "gad"],
    "alprazolam": ["anxiety", "panic disorder"],
    "lorazepam": ["anxiety", "seizures", "insomnia"],
    "tramadol": ["pain", "moderate to severe pain"],
    "morphine": ["severe pain", "palliative care"],
    "insulin": ["diabetes", "type 1 diabetes", "type 2 diabetes", "hyperglycemia"],
}


# Drug-condition contraindication relationships
DRUG_CONDITION_CONTRAINDICATIONS = {
    "metformin": ["kidney disease", "chronic kidney disease", "renal failure", "lactic acidosis"],
    "ibuprofen": ["kidney disease", "gastric ulcer", "heart failure", "asthma", "bleeding disorder"],
    "naproxen": ["kidney disease", "gastric ulcer", "heart failure", "asthma", "bleeding disorder"],
    "lisinopril": ["pregnancy", "angioedema history", "hyperkalemia"],
    "warfarin": ["pregnancy", "bleeding disorder", "active bleeding", "severe liver disease"],
    "amoxicillin": ["penicillin allergy", "penicillin_allergy"],
    "metoprolol": ["asthma", "severe bradycardia", "heart block"],
    "atenolol": ["asthma", "severe bradycardia", "heart block"],
    "propranolol": ["asthma", "severe bradycardia", "heart block"],
    "aspirin": ["aspirin allergy", "active bleeding", "children with viral illness"],
    "prednisone": ["systemic fungal infection", "active peptic ulcer"],
    "fluoroquinolone": ["tendon disorder", "myasthenia gravis", "qt prolongation"],
}


def extract_treatment_relationships(
    drugs: List[str],
    conditions: List[str],
    normalized_drugs: Dict[str, str] = None,
) -> List[Dict]:
    """
    Extract treatment relationships between drugs and conditions.
    
    Uses the DRUG_CONDITION_TREATMENTS knowledge base to identify
    which conditions each drug is treating.
    """
    relationships = []
    
    for drug in drugs:
        drug_normalized = (normalized_drugs or {}).get(drug.lower(), drug.lower())
        
        # Check if this drug has known treatment relationships
        for known_drug, treated_conditions in DRUG_CONDITION_TREATMENTS.items():
            if known_drug in drug_normalized or drug_normalized in known_drug:
                for condition in conditions:
                    condition_lower = condition.lower()
                    for treated in treated_conditions:
                        if treated in condition_lower or condition_lower in treated:
                            relationships.append({
                                "drug": drug,
                                "condition": condition,
                                "relationship_type": "treatment",
                                "confidence": 0.90,
                                "source": "knowledge_base",
                            })
                            break
    
    return relationships


def extract_contraindication_relationships(
    drugs: List[str],
    conditions: List[str],
    allergies: List[str],
    normalized_drugs: Dict[str, str] = None,
) -> List[Dict]:
    """
    Extract contraindication relationships.
    
    Identifies drugs that should not be used with certain conditions or allergies.
    """
    findings = []
    
    for drug in drugs:
        drug_normalized = (normalized_drugs or {}).get(drug.lower(), drug.lower())
        
        for known_drug, contraindicated in DRUG_CONDITION_CONTRAINDICATIONS.items():
            if known_drug in drug_normalized or drug_normalized in known_drug:
                for condition in conditions:
                    condition_lower = condition.lower()
                    for contra in contraindicated:
                        if contra in condition_lower or condition_lower in contra:
                            findings.append({
                                "drug": drug,
                                "condition": condition,
                                "relationship_type": "contraindication",
                                "severity": "CRITICAL",
                                "confidence": 0.85,
                                "source": "knowledge_base",
                            })
                            break
                
                for allergy in allergies:
                    allergy_normalized = normalize_allergy_name(allergy)
                    for contra in contraindicated:
                        if contra in allergy_normalized or allergy_normalized in contra:
                            findings.append({
                                "drug": drug,
                                "allergy": allergy,
                                "relationship_type": "allergy_contraindication",
                                "severity": "CRITICAL",
                                "confidence": 0.90,
                                "source": "knowledge_base",
                            })
                            break
    
    return findings


# ============================================================
# Clinical Context Analysis
# ============================================================

def analyze_clinical_context(
    text: str,
    entities: List[Dict],
    medications: List[Dict] = None,
) -> Dict:
    """
    Analyze the clinical context of a prescription for deeper insights.
    
    Returns:
        Dict with normalized names, treatment relationships, contraindications,
        and clinical confidence scores.
    """
    # Extract drugs, conditions, allergies from entities
    drugs = [e["word"] for e in entities if e.get("entity_group") == "DRUG"]
    conditions = [e["word"] for e in entities if e.get("entity_group") == "CONDITION"]
    
    # Normalize drug names
    normalized_drugs = {}
    for drug in drugs:
        normalized_drugs[drug.lower()] = normalize_drug_name(drug)
    
    # Extract allergy mentions from text
    allergies = []
    allergy_patterns = [
        r"allerg(?:y|ic)\s+to\s+([\w\s]+?)(?:\.|,|\s+and|\s+but|\s+history)",
        r"known\s+allerg(?:y|ic)\s+(?:to\s+)?([\w\s]+?)(?:\.|,|\s+and)",
        r"history\s+of\s+(?:anaphylaxis|allergic\s+reaction)\s+to\s+([\w\s]+?)(?:\.|,|\s+and)",
    ]
    text_lower = text.lower()
    for pattern in allergy_patterns:
        for match in re.finditer(pattern, text_lower):
            allergy = match.group(1).strip()
            if allergy and len(allergy) > 1:
                allergies.append(allergy)
    
    # Extract treatment relationships
    treatment_rels = extract_treatment_relationships(drugs, conditions, normalized_drugs)
    
    # Extract contraindications
    contra_rels = extract_contraindication_relationships(drugs, conditions, allergies, normalized_drugs)
    
    # Calculate overall clinical confidence
    if medications:
        total_conf = sum(
            DRUG_CONDITION_TREATMENTS.get(normalized_drugs.get(m["drug"].lower(), ""), [None]) is not None
            for m in medications
        )
        clinical_confidence = total_conf / len(medications) if medications else 0.0
    else:
        clinical_confidence = 0.5
    
    return {
        "normalized_drugs": normalized_drugs,
        "treatment_relationships": treatment_rels,
        "contraindication_relationships": contra_rels,
        "allergies_detected": list(set(allergies)),
        "clinical_confidence": round(clinical_confidence, 3),
        "drug_count": len(drugs),
        "condition_count": len(conditions),
        "allergy_count": len(allergies),
    }


def get_model_status() -> Dict:
    """Get the status of the advanced NER module."""
    return {
        "available": True,
        "drug_ontology_size": len(DRUG_ONTOLOGY),
        "allergy_ontology_size": len(ALLERGY_ONTOLOGY),
        "drug_condition_treatments": len(DRUG_CONDITION_TREATMENTS),
        "drug_condition_contraindications": len(DRUG_CONDITION_CONTRAINDICATIONS),
        "description": "Clinical entity linking and relationship extraction using medical knowledge base",
    }


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Clinical NER & Relation Extraction Test")
    print("=" * 70)
    
    # Test drug normalization
    test_drugs = ["Lipitor", "Tylenol", "Advil", "Glucophage", "Coumadin", "Zestril"]
    print("\nDrug Name Normalization:")
    for drug in test_drugs:
        normalized = normalize_drug_name(drug)
        print(f"  {drug:15s} → {normalized}")
    
    # Test treatment relationships
    print("\nTreatment Relationships:")
    drugs = ["amoxicillin", "metformin", "lisinopril"]
    conditions = ["pneumonia", "Type 2 Diabetes", "hypertension"]
    rels = extract_treatment_relationships(drugs, conditions)
    for r in rels:
        print(f"  {r['drug']:15s} → treats → {r['condition']:20s} (confidence: {r['confidence']:.0%})")
    
    # Test contraindications
    print("\nContraindications:")
    contra = extract_contraindication_relationships(
        drugs=["ibuprofen", "lisinopril"],
        conditions=["kidney disease", "gastric ulcer"],
        allergies=["penicillin"],
    )
    for c in contra:
        print(f"  {c['drug']:15s} → contraindicated with → {c.get('condition', c.get('allergy', 'N/A')):20s}")
    
    print(f"\nModel Status: {get_model_status()}")

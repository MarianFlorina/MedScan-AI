"""
Module 11: Indian Prescription Support
=======================================
Handles Hindi and Tamil medical prescriptions with:
1. Script detection (Devanagari, Tamil, Latin)
2. Transliteration mapping for Indian drug brand names
3. Common Indian medical term translations
4. Indian prescription format parsing

This module bridges the gap between Indian regional language prescriptions
and the English-based NER system.
"""

import re
from typing import List, Dict, Optional, Tuple


# ============================================================
# Script Detection (Unicode ranges)
# ============================================================

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),   # Hindi, Sanskrit, Marathi, Nepali
    "tamil": (0x0B80, 0x0BFF),        # Tamil
    "telugu": (0x0C00, 0x0C7F),       # Telugu
    "kannada": (0x0C80, 0x0CFF),      # Kannada
    "malayalam": (0x0D00, 0x0D7F),    # Malayalam
    "bengali": (0x0980, 0x09FF),      # Bengali
    "gujarati": (0x0A80, 0x0AFF),     # Gujarati
    "gurmukhi": (0x0A00, 0x0A7F),     # Punjabi
}


def detect_script(text: str) -> Dict[str, float]:
    """
    Detect the script(s) used in the text.
    
    Returns:
        Dict mapping script name to percentage of characters in that script
    """
    if not text:
        return {"latin": 1.0}
    
    script_counts = {}
    total_indian = 0
    
    for char in text:
        code = ord(char)
        found = False
        
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= code <= end:
                script_counts[script_name] = script_counts.get(script_name, 0) + 1
                total_indian += 1
                found = True
                break
        
        if not found and char.isalpha():
            script_counts["latin"] = script_counts.get("latin", 0) + 1
    
    total = sum(script_counts.values()) or 1
    return {k: v / total for k, v in script_counts.items()}


def is_indian_script(text: str) -> bool:
    """Check if text contains any Indian script characters."""
    for char in text:
        code = ord(char)
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= code <= end:
                return True
    return False


def get_primary_script(text: str) -> str:
    """Get the dominant script in the text."""
    scripts = detect_script(text)
    if not scripts:
        return "latin"
    return max(scripts, key=scripts.get)


# ============================================================
# Indian Drug Brand Name Database
# ============================================================

# Comprehensive mapping of Indian brand names to generic names
# Includes Hindi/Tamil transliterations and common abbreviations
INDIAN_DRUG_BRANDS = {
    # === Analgesics / Antipyretics ===
    "crocin": "paracetamol",
    "crocin advance": "paracetamol",
    "crocin cold & flu": "paracetamol",
    "dolo": "paracetamol",
    "dolo 650": "paracetamol",
    "paracip": "paracetamol",
    "pacimol": "paracetamol",
    "combiflam": "ibuprofen-paracetamol",
    "combiflam plus": "ibuprofen-paracetamol",
    "brufen": "ibuprofen",
    "brufen 400": "ibuprofen",
    "brufen 600": "ibuprofen",
    "ibugesic": "ibuprofen",
    "ibugesic plus": "ibuprofen-paracetamol",
    "nurofast": "ibuprofen",
    "nurofast rapid": "ibuprofen",
    "volini": "diclofenac",
    "volini gel": "diclofenac",
    "voveran": "diclofenac",
    "voveran emulgel": "diclofenac",
    "voltaren": "diclofenac",
    "zyflam": "diclofenac",
    "nimulid": "nimesulide",
    "nise": "nimesulide",
    "nice": "nimesulide",
    "hifenac": "aceclofenac",
    "hifenac plus": "aceclofenac-paracetamol",
    "zifi": "cefixime",
    "azee": "azithromycin",
    "azithral": "azithromycin",
    "zeeba": "azithromycin",
    
    # === Antibiotics ===
    "amoxil": "amoxicillin",
    "amoxicillin": "amoxicillin",
    "amoxicap": "amoxicillin",
    "amclav": "amoxicillin-clavulanate",
    "augmentin": "amoxicillin-clavulanate",
    "clavam": "amoxicillin-clavulanate",
    "clavunate": "amoxicillin-clavulanate",
    "mox": "amoxicillin",
    "mox Carson": "amoxicillin",
    "cifran": "ciprofloxacin",
    "cipro": "ciprofloxacin",
    "ciprolet": "ciprofloxacin",
    "ciflo": "ciprofloxacin",
    "ciplox": "ciprofloxacin",
    "quetin": "cetirizine",
    "cetzine": "cetirizine",
    "alerid": "cetirizine",
    "alconil": "cetirizine",
    "levoflox": "levofloxacin",
    "levoquin": "levofloxacin",
    "hepcom": "metronidazole",
    "metrogyl": "metronidazole",
    "tinidazole": "tinidazole",
    "zyvoxid": "linezolid",
    "rowipen": "amoxicillin",
    
    # === Cardiovascular ===
    "atelop": "atenolol",
    "tenormin": "atenolol",
    "amlo": "amlodipine",
    "amlopres": "amlodipine",
    "amlopress": "amlodipine",
    "stamlo": "amlodipine",
    "amlodac": "amlodipine",
    "atorva": "atorvastatin",
    "atorvast": "atorvastatin",
    "lipitor": "atorvastatin",
    "atocor": "atorvastatin",
    "tORVA": "atorvastatin",
    "rosuvas": "rosuvastatin",
    "crestor": "rosuvastatin",
    "rosaday": "rosuvastatin",
    "rosuvastat": "rosuvastatin",
    "losar": "losartan",
    "losartan": "losartan",
    "losaqual": "losartan",
    "lozapin": "losartan",
    "telma": "telmisartan",
    "telmisartan": "telmisartan",
    "telma h": "telmisartan-hydrochlorothiazide",
    "olmy": "olmesartan",
    "olmesartan": "olmesartan",
    "olmetec": "olmesartan",
    "tacardia": "metoprolol",
    "metolar": "metoprolol",
    "metoprolol": "metoprolol",
    "labetalol": "labetalol",
    "nicardia": "nifedipine",
    "nifedipine": "nifedipine",
    "adalat": "nifedipine",
    "clopilet": "clopidogrel",
    "plavix": "clopidogrel",
    "deplatt": "clopidogrel",
    "depet": "clopidogrel",
    "warcouin": "warfarin",
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "ecotrin": "aspirin",
    "disprin": "aspirin",
    "aspirin": "aspirin",
    "ecosprin": "aspirin",
    
    # === Diabetes ===
    "glycomet": "metformin",
    "metformin": "metformin",
    "glyciphage": "metformin",
    "gluconorm": "glimepiride",
    "glimepiride": "glimepiride",
    "amaryl": "glimepiride",
    "glimestar": "glimepiride",
    "glucobay": "acarbose",
    "acarbose": "acarbose",
    "prandase": "acarbose",
    "januvia": "sitagliptin",
    "sitagliptin": "sitagliptin",
    "janumet": "sitagliptin-metformin",
    "voglibose": "voglibose",
    "obimet": "metformin",
    "obimet-gv": "metformin-voglibose",
    "pioglit": "pioglitazone",
    "pioglitazone": "pioglitazone",
    "piozone": "pioglitazone",
    "inzolen": "insulin",
    "insulin": "insulin",
    "human actrapid": "insulin",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novorapid": "insulin aspart",
    "toujeo": "insulin glargine",
    "tresiba": "insulin degludec",
    
    # === Gastrointestinal ===
    "pan": "pantoprazole",
    "pantop": "pantoprazole",
    "pantodac": "pantoprazole",
    "pantoprazole": "pantoprazole",
    "pantocid": "pantoprazole",
    "sompraz": "pantoprazole",
    "ocid": "omeprazole",
    "omeprazole": "omeprazole",
    "losec": "omeprazole",
    "omecap": "omeprazole",
    "nexpro": "esomeprazole",
    "esomeprazole": "esomeprazole",
    "nexpro RD": "esomeprazole-domperidone",
    "pan-d": "pantoprazole-domperidone",
    "raberex": "rabeprazole",
    "rabeprazole": "rabeprazole",
    "rabicip": "rabeprazole",
    "ranitidine": "ranitidine",
    "rantac": "ranitidine",
    "zinetac": "ranitidine",
    "gelusil": "antacid",
    "digene": "antacid",
    "mylanta": "antacid",
    "domstal": "domperidone",
    "domperidone": "domperidone",
    "perinorm": "domperidone",
    "emset": "ondansetron",
    "ondansetron": "ondansetron",
    "vomikind": "ondansetron",
    "ondem": "ondansetron",
    "loperamide": "loperamide",
    "imodium": "loperamide",
    
    # === Psychiatry / Neurology ===
    "deprevine": "escitalopram",
    "lexapro": "escitalopram",
    "eszee": "escitalopram",
    "serlift": "sertraline",
    "zolfresh": "zolpidem",
    "stilnoct": "zolpidem",
    "zolpidem": "zolpidem",
    "lamictal": "lamotrigine",
    "lamotrigine": "lamotrigine",
    "tegretol": "carbamazepine",
    "carbamazepine": "carbamazepine",
    "epileptin": "carbamazepine",
    "valparin": "valproic acid",
    "valproic acid": "valproic acid",
    "depakote": "valproic acid",
    "leptic": "levetiracetam",
    "levetiracetam": "levetiracetam",
    "keppra": "levetiracetam",
    "gabapin": "gabapentin",
    "gabapentin": "gabapentin",
    "neurontin": "gabapentin",
    "pregabid": "pregabalin",
    "pregabalin": "pregabalin",
    "lyrica": "pregabalin",
    "clonotril": "clonazepam",
    "clonazepam": "clonazepam",
    "rivotril": "clonazepam",
    "alprax": "alprazolam",
    "alprazolam": "alprazolam",
    "xanax": "alprazolam",
    "ativan": "lorazepam",
    "lorazepam": "lorazepam",
    "welnorm": "tryptophan",
    "shentrin": "donepezil",
    "donepezil": "donepezil",
    "aricept": "donepezil",
    
    # === Respiratory ===
    "asthalin": "salbutamol",
    "salbutamol": "salbutamol",
    "ventolin": "salbutamol",
    "proair": "salbutamol",
    "duolin": "ipratropium-salbutamol",
    "seroflo": "fluticasone-salmeterol",
    "fluticasone": "fluticasone",
    "flixotide": "fluticasone",
    "singulair": "montelukast",
    "montair": "montelukast",
    "montelukast": "montelukast",
    "montemac": "montelukast",
    "deriphyllin": "theophylline",
    "theophylline": "theophylline",
    "theobid": "theophylline",
    "azithral": "azithromycin",
    "azyth": "azithromycin",
    "zincofax": "zinc",
    
    # === Antibiotics / Anti-infectives ===
    "taxim": "ceftriaxone",
    "ceftriaxone": "ceftriaxone",
    "ceflox": "ciprofloxacin",
    "oflox": "ofloxacin",
    "ofloxacin": "ofloxacin",
    "zyvoxid": "linezolid",
    "linid": "linezolid",
    "metrogyl": "metronidazole",
    "flagyl": "metronidazole",
    "tiniba": "tinidazole",
    "norflox": "norfloxacin",
    "norfloxacin": "norfloxacin",
    "norflox tz": "norfloxacin-tinidazole",
    "norflo TZ": "norfloxacin-tinidazole",
    "doxycycline": "doxycycline",
    "doxy": "doxycycline",
    "vibramycin": "doxycycline",
    "minoz": "metronidazole",
    "mac-500": "azithromycin",
    
    # === Antifungals ===
    "fungitest": "fluconazole",
    "fluconazole": "fluconazole",
    "canesten": "clotrimazole",
    "clotrimazole": "clotrimazole",
    "terbinafine": "terbinafine",
    "lamilica": "terbinafine",
    
    # === Dermatology ===
    "betnovate": "betamethasone",
    "betamethasone": "betamethasone",
    "lobate": "clobetasol",
    "clobetasol": "clobetasol",
    "deriva": "adapalene",
    "adapalene": "adapalene",
    "retino": "tretinoin",
    "tretinoin": "tretinoin",
    "suncros": "sunscreen",
    
    # === Steroids / Anti-inflammatory ===
    "predmet": "prednisolone",
    "prednisolone": "prednisolone",
    "prednisone": "prednisone",
    "deltasone": "prednisone",
    "wysolone": "prednisolone",
    "dexamethasone": "dexamethasone",
    "dexona": "dexamethasone",
    "dexameth": "dexamethasone",
    "methylpred": "methylprednisolone",
    "medrol": "methylprednisolone",
    "deflazacort": "deflazacort",
    "dezacort": "deflazacort",
    
    # === Thyroid ===
    "thyrox": "levothyroxine",
    "levothyroxine": "levothyroxine",
    "eltroxin": "levothyroxine",
    "thyronorm": "levothyroxine",
    "euthyrox": "levothyroxine",
    
    # === Vitamins / Supplements ===
    "neurobion": "vitamin b complex",
    "supradyn": "multivitamin",
    "becosules": "vitamin b complex",
    "celin": "vitamin c",
    "vitamin c": "vitamin c",
    "calcirol": "vitamin d3",
    "vitamin d3": "vitamin d3",
    "shelcal": "calcium-vitamin d3",
    "dolo neurobion": "vitamin b complex",
    "fosamax": "alendronate",
    "alendronate": "alendronate",
    
    # === Urology ===
    "tamsulosin": "tamsulosin",
    "flomax": "tamsulosin",
    "tamlo": "tamsulosin",
    "sildenafil": "sildenafil",
    "viagra": "sildenafil",
    "filagra": "sildenafil",
    "tadalafil": "tadalafil",
    "cialis": "tadalafil",
    "tadalis": "tadalafil",
    "silagra": "sildenafil",
    
    # === Common Indian Combinations ===
    "pan-d": "pantoprazole-domperidone",
    "pan-ds": "pantoprazole-domperidone",
    "raberex-ds": "rabeprazole-domperidone",
    "combiflam": "ibuprofen-paracetamol",
    "vicodin": "hydrocodone-acetaminophen",
    "nexpro rd": "esomeprazole-domperidone",
    "monocef": "ceftriaxone",
    "azel": "azelastine",
    "astelin": "azelastine",
    "rhinocort": "budesonide",
    "budecort": "budesonide",
    "budesonide": "budesonide",
    
    # === Common Indian Prescription Abbreviations ===
    "tab": None,  # tablet form marker
    "cap": None,  # capsule form marker
    "inj": None,  # injection form marker
    "syp": None,  # syrup form marker
    "susp": None,  # suspension form marker
    "drop": None,  # drops form marker
    "cream": None,  # cream form marker
    "gel": None,  # gel form marker
    "ointment": None,  # ointment form marker
    "md": None,  # modified release
    "sr": None,  # sustained release
    "xr": None,  # extended release
    "od": None,  # once daily
    "bd": None,  # twice daily
    "tid": None,  # three times daily
    "qid": None,  # four times daily
    "prn": None,  # as needed
    "hs": None,  # at bedtime
    "ac": None,  # before meals
    "pc": None,  # after meals
}


# ============================================================
# Hindi/Tamil Medical Terms
# ============================================================

HINDI_MEDICAL_TERMS = {
    # Common Hindi medical terms
    "dawa": "medicine",
    "davai": "medicine",
    "goli": "tablet",
    "tablets": "tablet",
    "injection": "injection",
    "syrup": "syrup",
    "tob": "cream",
    "mahwari": "menstruation",
    "pregnancy": "pregnancy",
    "garbhavastha": "pregnancy",
    "diabetes": "diabetes",
    "sugar": "diabetes",
    "bp": "hypertension",
    "blood pressure": "hypertension",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "heart": "heart",
    "dil": "heart",
    "heart attack": "myocardial infarction",
    "asthma": "asthma",
    "dama": "asthma",
    "fever": "fever",
    "bukhar": "fever",
    "jukaam": "cold",
    "cold": "cold",
    "khaansi": "cough",
    "cough": "cough",
    "sardi": "cold",
    "gale ka dard": "sore throat",
    "pet dard": "abdominal pain",
    "sir dard": "headache",
    "headache": "headache",
    "joints": "joints",
    "jod": "joints",
    "arthritis": "arthritis",
    "ghutno ka dard": "knee pain",
    "allergy": "allergy",
    "skin": "skin",
    "twacha": "skin",
    "rash": "rash",
    "chakkar": "dizziness",
    "dizziness": "dizziness",
    "neend": "sleep",
    "insomnia": "insomnia",
    "thakan": "fatigue",
    "fatigue": "fatigue",
    "wazan": "weight",
    "weight": "weight",
    "diet": "diet",
    "khan paan": "diet",
    "kidney": "kidney",
    "gurda": "kidney",
    "liver": "liver",
    "kaleja": "liver",
    "diarrhea": "diarrhea",
    "potty": "diarrhea",
    "loose motions": "diarrhea",
    "vomiting": "vomiting",
    "ultee": "vomiting",
    "constipation": "constipation",
    "qabz": "constipation",
    "sugar test": "blood glucose test",
    "blood test": "blood test",
    "urine test": "urine test",
    "x-ray": "x-ray",
    "ecg": "ECG",
    "echo": "echocardiogram",
    "ultrasound": "ultrasound",
    "sonography": "ultrasound",
    "surgery": "surgery",
    "operation": "surgery",
    "doctor": "doctor",
    "nurse": "nurse",
    "hospital": "hospital",
    "pharmacy": "pharmacy",
    "medicine shop": "pharmacy",
    "davai khana": "take medicine",
    "khana": "eat",
    "peena": "drink",
    "subah": "morning",
    "shaam": "evening",
    "raat": "night",
    "dophar": "afternoon",
    "din": "day",
    "hafte": "week",
    "mahine": "month",
    "barah": "twelve",
    "che": "six",
    "do": "two",
    "ek": "one",
    "char": "four",
    "paanch": "five",
    "aat": "eight",
    "nau": "nine",
    "das": "ten",
}

TAMIL_MEDICAL_TERMS = {
    # Common Tamil medical terms
    "marundhu": "medicine",
    "mooligai": "medicine",
    "maathirai": "tablet",
    "injection": "injection",
    "syrup": "syrup",
    "kaayakulaini": "pregnancy",
    "sarkarai noi": "diabetes",
    "sugar": "diabetes",
    "ratha azhutham": "hypertension",
    "blood pressure": "hypertension",
    "idhayam": "heart",
    "asthma": "asthma",
    "moochu surukkam": "asthma",
    "kaayi": "fever",
    "fever": "fever",
    "jaladam": "cold",
    "cold": "cold",
    "irumal": "cough",
    "cough": "cough",
    "thalai veechu": "headache",
    "headache": "headache",
    "mootu vali": "joint pain",
    "arthritis": "arthritis",
    "allergy": "allergy",
    "thol": "skin",
    "skin": "skin",
    "thirumbal": "rash",
    "migai": "dizziness",
    "thookam": "sleep",
    "insomnia": "insomnia",
    "sudarvadham": "fatigue",
    "kidney": "kidney",
    "mannirappu": "kidney",
    "kalliral": "liver",
    "kodhuvali": "diarrhea",
    "porikai": "vomiting",
    "doctor": "doctor",
    "maruthuvamanai": "hospital",
    "marundhu sadam": "pharmacy",
}


# ============================================================
# Transliteration Functions
# ============================================================

def transliterate_english_to_roman(text: str) -> str:
    """
    Convert common Indian drug brand name spellings to standard form.
    Handles common transliteration variations.
    """
    text_lower = text.lower().strip()
    
    # Direct lookup
    if text_lower in INDIAN_DRUG_BRANDS:
        result = INDIAN_DRUG_BRANDS[text_lower]
        return result if result else text
    
    # Try partial matches (for brand names like "Tab Crocin 650")
    words = text_lower.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            phrase = " ".join(words[i:j])
            if phrase in INDIAN_DRUG_BRANDS:
                result = INDIAN_DRUG_BRANDS[phrase]
                if result:
                    # Replace the matched phrase with the generic name
                    remaining = words[:i] + [result] + words[j:]
                    return " ".join(remaining)
    
    return text


def detect_and_transliterate(text: str) -> Tuple[str, str, Dict]:
    """
    Detect script and transliterate if needed.
    
    Args:
        text: Input text (may contain Hindi/Tamil/English)
    
    Returns:
        Tuple of (transliterated_text, primary_script, script_info)
    """
    script_info = detect_script(text)
    primary_script = get_primary_script(text)
    
    if primary_script == "latin":
        # Already in English - just map Indian brand names
        transliterated = transliterate_english_to_roman(text)
        return transliterated, primary_script, script_info
    
    # For Indian scripts, we need transliteration
    # Since we can't install indic-transliteration, use dictionary-based approach
    transliterated_parts = []
    words = text.split()
    
    for word in words:
        # Check if word is an Indian medical term
        word_lower = word.lower()
        
        if word_lower in HINDI_MEDICAL_TERMS:
            transliterated_parts.append(HINDI_MEDICAL_TERMS[word_lower])
        elif word_lower in TAMIL_MEDICAL_TERMS:
            transliterated_parts.append(TAMIL_MEDICAL_TERMS[word_lower])
        elif word_lower in INDIAN_DRUG_BRANDS:
            result = INDIAN_DRUG_BRANDS[word_lower]
            transliterated_parts.append(result if result else word)
        else:
            # Keep as-is if no mapping found
            transliterated_parts.append(word)
    
    transliterated = " ".join(transliterated_parts)
    return transliterated, primary_script, script_info


# ============================================================
# Indian Prescription Parser
# ============================================================

def parse_indian_prescription_format(text: str) -> Dict:
    """
    Parse common Indian prescription formats.
    
    Indian prescriptions often have:
    - Drug name + dosage + frequency
    - Duration in days/weeks
    - Special instructions in Hindi/Tamil
    """
    result = {
        "original_text": text,
        "transliterated_text": "",
        "primary_script": "latin",
        "script_info": {},
        "indian_drugs_found": [],
        "medical_terms_found": [],
    }
    
    # Transliterate
    transliterated, script, scripts = detect_and_transliterate(text)
    result["transliterated_text"] = transliterated
    result["primary_script"] = script
    result["script_info"] = scripts
    
    # Find Indian drug brand names
    text_lower = text.lower()
    for brand, generic in INDIAN_DRUG_BRANDS.items():
        if generic is None:  # Skip form markers
            continue
        if brand in text_lower:
            result["indian_drugs_found"].append({
                "brand": brand,
                "generic": generic,
            })
    
    # Find medical terms
    for term, english in HINDI_MEDICAL_TERMS.items():
        if term in text_lower:
            result["medical_terms_found"].append({
                "original": term,
                "english": english,
            })
    
    for term, english in TAMIL_MEDICAL_TERMS.items():
        if term in text_lower:
            result["medical_terms_found"].append({
                "original": term,
                "english": english,
            })
    
    return result


def get_indian_drug_suggestions(text: str) -> List[Dict]:
    """
    Get Indian drug brand name suggestions from text.
    Useful for auto-complete and suggestion features.
    """
    suggestions = []
    text_lower = text.lower()
    
    for brand, generic in INDIAN_DRUG_BRANDS.items():
        if generic is None:
            continue
        if brand in text_lower or (len(brand) > 3 and any(w in text_lower for w in brand.split())):
            suggestions.append({
                "brand": brand,
                "generic": generic,
                "confidence": 0.9 if brand in text_lower else 0.6,
            })
    
    return sorted(suggestions, key=lambda x: -x["confidence"])


def get_model_status() -> Dict:
    """Get the status of the Indian prescription module."""
    return {
        "available": True,
        "indian_drug_brands": len([v for v in INDIAN_DRUG_BRANDS.values() if v is not None]),
        "hindi_terms": len(HINDI_MEDICAL_TERMS),
        "tamil_terms": len(TAMIL_MEDICAL_TERMS),
        "scripts_supported": list(SCRIPT_RANGES.keys()),
        "description": "Indian prescription support with Hindi/Tamil transliteration and brand name mapping",
    }


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Indian Prescription Support Test")
    print("=" * 70)
    
    # Test 1: Script detection
    print("\n1. Script Detection:")
    test_texts = [
        "Tab Crocin 650mg twice daily for fever",
        "Inj Ceftriaxone 1g IV once daily",
        "Crocin cold and flu for bukhar",  # Hindi word for fever
        "Metformin for sugar control",  # Indian usage
        "Combiflam for joint pain",  # Indian brand
        "Voveran gel for ghutno ka dard",  # Hindi for knee pain
    ]
    
    for text in test_texts:
        scripts = detect_script(text)
        primary = get_primary_script(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Script: {primary} | All: {scripts}")
    
    # Test 2: Indian drug brand mapping
    print("\n2. Indian Drug Brand Mapping:")
    indian_texts = [
        "Tab Crocin 650mg twice daily",
        "Inj Monocef 1g IV once daily",
        "Tab Combiflam twice daily with food",
        "Cap Omez 20mg before breakfast",
        "Syrup Asthalin 5ml three times daily",
        "Tab Glycomet 500mg twice daily for sugar",
        "Cream Betnovate-N for skin rash",
        "Tab Losar 50mg once daily for BP",
        "Tab Azee 500mg once daily for 3 days",
        "Cap Taxim 200mg twice daily",
    ]
    
    for text in indian_texts:
        transliterated, script, info = detect_and_transliterate(text)
        drugs = get_indian_drug_suggestions(text)
        print(f"\n  Original: {text}")
        print(f"  Transliterated: {transliterated}")
        print(f"  Drugs found: {[(d['brand'], d['generic']) for d in drugs]}")
    
    # Test 3: Full prescription parsing
    print("\n3. Full Prescription Parsing:")
    rx_text = "Tab Crocin 650mg for bukhar (fever), Tab Combiflam for joint pain, Cap Omez 20 before breakfast for pet dard"
    parsed = parse_indian_prescription_format(rx_text)
    print(f"  Original: {parsed['original_text']}")
    print(f"  Transliterated: {parsed['transliterated_text']}")
    print(f"  Primary script: {parsed['primary_script']}")
    print(f"  Indian drugs: {parsed['indian_drugs_found']}")
    print(f"  Medical terms: {parsed['medical_terms_found']}")
    
    print(f"\n{'=' * 70}")
    print(f"Model Status: {get_model_status()}")


# ============================================================
# Brand -> generic, and Hindi/Tamil drug-name transliteration
# ============================================================

def to_generic(name: str) -> str:
    """'Crocin' -> 'paracetamol'; unknown names are returned lowercased and unchanged."""
    key = re.sub(r"\s*\([^)]*\)\s*$", "", (name or "").lower().strip())  # 'mox (amoxicillin)' -> 'mox'
    return INDIAN_DRUG_BRANDS.get(key) or key


# Indian-language prescriptions usually write English drug names phonetically
# (पैरासिटामोल = paracetamol). OCR spelling varies, so matching below is fuzzy.
INDIC_DRUG_TRANSLIT = {
    # Hindi (Devanagari)
    "पैरासिटामोल": "paracetamol", "पेरासिटामोल": "paracetamol", "अमॉक्सिसिलिन": "amoxicillin", "अमोक्सिसिलिन": "amoxicillin",
    "लेवोसेटिरिज़िन": "levocetirizine", "लेवोसेटिरिजिन": "levocetirizine", "ओमेप्राज़ोल": "omeprazole",
    "पैंटोप्राज़ोल": "pantoprazole", "एज़िथ्रोमाइसिन": "azithromycin", "सेट्रिज़िन": "cetirizine",
    "आइबुप्रोफेन": "ibuprofen", "मेटफॉर्मिन": "metformin", "एम्लोडिपिन": "amlodipine", "एटोरवास्टेटिन": "atorvastatin",
    "डाइक्लोफेनाक": "diclofenac", "डोम्पेरिडोन": "domperidone", "सिमेथिकोन": "simethicone",
    "सिप्रोफ्लोक्सासिन": "ciprofloxacin", "डॉक्सीसाइक्लिन": "doxycycline", "वारफेरिन": "warfarin", "एस्पिरिन": "aspirin",
    # Tamil
    "பாராசிட்டமால்": "paracetamol", "பாராசிடமால்": "paracetamol", "அமாக்ஸிசிலின்": "amoxicillin",
    "ஒமிப்ராசோல்": "omeprazole", "ஓமெப்ரசோல்": "omeprazole", "டோம்பெரிடோன்": "domperidone", "டோமெபெரிடோன்": "domperidone",
    "சிமெத்திக்கோன்": "simethicone", "லெவோசெடிரிசின்": "levocetirizine", "பான்டோப்ரசோல்": "pantoprazole",
    "அசித்ரோமைசின்": "azithromycin", "செட்டிரிசின்": "cetirizine", "மெட்ஃபார்மின்": "metformin",
    "ஆம்லோடிபின்": "amlodipine", "டைக்ளோஃபெனாக்": "diclofenac", "ஐபுப்ரோஃபென்": "ibuprofen",
}


def _has_indic(token: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in token for lo, hi in SCRIPT_RANGES.values())


def transliterate_indic_drugs(text: str) -> str:
    """Replace Hindi/Tamil drug names with their English generic names (fuzzy match, cutoff 0.7)."""
    import difflib
    keys = list(INDIC_DRUG_TRANSLIT)
    parts = re.split(r"(\s+)", text)
    for i, tok in enumerate(parts):
        core = tok.strip(".,:;()[]|-_")
        if core and _has_indic(core):
            hit = difflib.get_close_matches(core, keys, n=1, cutoff=0.7)
            if hit:
                parts[i] = tok.replace(core, INDIC_DRUG_TRANSLIT[hit[0]])
    return "".join(parts)

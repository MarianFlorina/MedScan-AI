"""
Module 2: Medical NER & Relation Extraction
=============================================
Hybrid approach: ML-based NER (biomedical BERT) with regex fallback.
- Primary: d4data/biomedical-ner-all (66M param transformer)
- Fallback: Regex + medical dictionaries for offline/error cases
Drug/condition data is imported from the shared drug_knowledge module.
"""

import re
from typing import List, Dict, Tuple
from drug_knowledge import DRUG_NAMES, CONDITION_NAMES
from module11_indian_rx import (
    INDIAN_DRUG_BRANDS, detect_and_transliterate,
    transliterate_english_to_roman, is_indian_script,
)

# ============================================================
# Medical Knowledge Base for NER (imported from shared module)
# ============================================================

KNOWN_DRUGS = DRUG_NAMES  # Alias for backward compatibility
KNOWN_CONDITIONS = CONDITION_NAMES  # Alias for backward compatibility


# ============================================================
# ML NER (lazy-loaded, falls back to regex if unavailable)
# ============================================================

_ml_ner_available = None  # None = not checked yet
_ml_ner = None


def _get_ml_ner():
    global _ml_ner_available, _ml_ner
    if _ml_ner_available is False:
        return None
    if _ml_ner is None:
        try:
            from module2_ml_ner import get_ml_ner
            ner = get_ml_ner()
            if ner.is_available:
                _ml_ner = ner
                _ml_ner_available = True
            else:
                _ml_ner_available = False
                print(f"[NER] ML model failed to load: {ner.error}. Using regex fallback.")
        except Exception as e:
            _ml_ner_available = False
            print(f"[NER] ML module not available ({e}). Using regex fallback.")
    return _ml_ner


# ============================================================
# spaCy (optional — falls back to regex if model not installed)
# ============================================================

_spacy_nlp = None
_spacy_available = None  # None = not checked yet


def _get_nlp():
    global _spacy_nlp, _spacy_available
    if _spacy_available is False:
        return None
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            _spacy_available = True
        except (ImportError, OSError) as e:
            _spacy_available = False
            import warnings
            warnings.warn(f"spaCy model not available ({e}). Using regex-based NER only.")
    return _spacy_nlp


# ============================================================
# Entity Extraction (Hybrid: ML + Regex)
# ============================================================

def _drop_nested(entities: List[Dict]) -> List[Dict]:
    """Keep only the longest phrase when shorter same-type entities are nested inside it
    ("Acute upper respiratory tract infection" vs "respiratory tract infection" vs "infection")."""
    def nested(e):
        return any(o is not e and o["entity_group"] == e["entity_group"] and e["entity_group"] in ("CONDITION", "DRUG")
                   and len(o["word"]) > len(e["word"]) and e["word"].lower() in o["word"].lower()
                   and abs(e["start"] - o["start"]) <= len(o["word"]) + 60 for o in entities)
    entities = [e for e in entities if not nested(e)]
    return [e for e in entities if not _is_condition_noise(e)]


# Header / department words the model sometimes tags as conditions on real prescriptions
_NOT_CONDITIONS = {
    "hospital", "hospitals", "clinic", "medicine", "general medicine", "cardiology", "nephrology", "endocrinology",
    "dermatology", "pediatric", "paediatrics", "department", "apollo", "lives", "symptoms", "regular", "md", "mbbs",
    "self-medication", "physiotherapy", "advice", "diagnosis", "complaints", "name", "age", "date", "kidney function",
}


def _is_known_condition(word: str) -> bool:
    """True if the phrase is a dictionary condition or contains one ('atrial fibrillation allergy')."""
    w = word.lower().strip()
    return w in KNOWN_CONDITIONS or any(len(c) >= 4 and c in w for c in KNOWN_CONDITIONS)


def _is_condition_noise(e: Dict) -> bool:
    if e["entity_group"] != "CONDITION" or _is_known_condition(e["word"]):
        return False
    w = e["word"].lower().strip(" .,:;-")
    return len(w) <= 3 or w in _NOT_CONDITIONS or float(e.get("score", 1.0)) < 0.6 or bool(re.search(r"\b(name|age)\b", w))


def extract_medical_entities(text: str) -> List[Dict]:
    if not text or not text.strip():
        return []

    # --- Try ML NER first ---
    ml_entities = []
    ml_ner = _get_ml_ner()
    if ml_ner is not None:
        try:
            raw_ml = ml_ner.extract(text)
            if raw_ml:
                ml_entities = _normalize_ml_entities(raw_ml)
        except Exception as e:
            print(f"[NER] ML extraction failed ({e}). Falling back to regex.")

    # --- Regex supplement/fallback ---
    regex_entities = _extract_entities_regex(text)

    # --- Merge: ML as primary, regex fills gaps ---
    if ml_entities:
        return _drop_nested(_merge_entities(ml_entities, regex_entities, text))
    return _drop_nested(regex_entities)


def _normalize_ml_entities(entities: List[Dict]) -> List[Dict]:
    """Normalize ML model output to our standard format."""
    normalized = []
    for ent in entities:
        # Map entity types to our standard format
        entity_type = ent["type"].upper()
        text = ent["text"].strip()

        # Normalize text: collapse spaces around hyphens (ML model artifact)
        text = re.sub(r'\s*-\s*', '-', text)

        # Skip empty or very short entities
        if len(text) < 2:
            continue

        # Accept already-mapped types from ML module (CONDITION, DRUG, DOSAGE, etc.)
        # Also handle raw model labels if mapping was bypassed
        VALID_TYPES = {"DRUG", "DOSAGE", "CONDITION", "FREQUENCY", "DURATION", "ROUTE", "FORM", "BODY_PART", "PROCEDURE", "LAB_VALUE"}
        RAW_TO_STANDARD = {
            "DISEASE_DISORDER": "CONDITION",
            "SIGN_SYMPTOM": "CONDITION",
            "CLINICAL_EVENT": "CONDITION",
            "MEDICATION": "DRUG",
            "DRUG_IDENTIFIER": "DRUG",
            "STRENGTH": "DOSAGE",
            "BIOLOGICAL_STRUCTURE": "BODY_PART",
            "DIAGNOSTIC_PROCEDURE": "PROCEDURE",
            "LAB_VALUE": "LAB_VALUE",
        }

        if entity_type not in VALID_TYPES:
            entity_type = RAW_TO_STANDARD.get(entity_type, "OTHER")

        # Fix ML model offset bugs — use text length for end position
        start_pos = ent.get("start", 0)
        end_pos = start_pos + len(text)
        
        normalized.append({
            "word": text,
            "entity_group": entity_type,
            "score": round(ent["confidence"] / 100, 4) if ent["confidence"] > 1 else ent["confidence"],
            "start": start_pos,
            "end": end_pos
        })

    return normalized


def _merge_entities(ml_entities: List[Dict], regex_entities: List[Dict], text: str) -> List[Dict]:
    """
    Merge ML and regex entities.
    Strategy:
    - ML is primary for context-aware extraction
    - Regex fills gaps (Indian brands, known drugs ML missed)
    - Regex wins for DRUG entities that ML missed (lower threshold)
    - ML wins for CONDITIONS (better context understanding)
    """
    merged = list(ml_entities)
    ml_positions = {(e["start"], e["end"]) for e in ml_entities}
    ml_drugs = {e["word"].lower() for e in ml_entities if e["entity_group"] == "DRUG"}

    for reg_ent in regex_entities:
        pos = (reg_ent["start"], reg_ent["end"])
        word_lower = reg_ent["word"].lower()

        # Skip if ML already has this exact position
        if pos in ml_positions:
            continue

        # For DRUGs: add if ML didn't find it (regex knows Indian brands)
        if reg_ent["entity_group"] == "DRUG":
            # Check if ML found this drug (or a close match)
            if word_lower not in ml_drugs:
                # Check fuzzy match against ML drugs
                found = False
                for ml_drug in ml_drugs:
                    if (len(word_lower) > 3 and word_lower in ml_drug) or (len(ml_drug) > 3 and ml_drug in word_lower):
                        found = True
                        break
                if not found:
                    merged.append(reg_ent)
                    ml_positions.add(pos)

        # For CONDITIONs: a longer dictionary phrase that contains an ML fragment replaces it
        # ("infection" -> "upper respiratory tract infection")
        elif reg_ent["entity_group"] == "CONDITION":
            overlapped = [e for e in merged if e["entity_group"] == "CONDITION"
                          and e["start"] < reg_ent["end"] and e["end"] > reg_ent["start"]]
            if overlapped and all(len(e["word"]) < len(reg_ent["word"]) for e in overlapped):
                for e in overlapped:
                    merged.remove(e)
                merged.append(reg_ent)
                ml_positions.add(pos)

        # For DOSAGE: add if ML missed it
        elif reg_ent["entity_group"] == "DOSAGE":
            if not any(e["entity_group"] == "DOSAGE" and
                       abs(e["start"] - reg_ent["start"]) < 5
                       for e in ml_entities):
                merged.append(reg_ent)
                ml_positions.add(pos)

    # Deduplicate: identical text at nearby positions is one entity; the same text at
    # different places (e.g. "500 mg" for two drugs) is kept.
    kept = []
    for ent in sorted(merged, key=lambda e: -e["score"]):
        if any(k["word"].lower() == ent["word"].lower() and k["entity_group"] == ent["entity_group"]
               and abs(k["start"] - ent["start"]) <= 5 for k in kept):
            continue
        kept.append(ent)

    # "Acute upper respiratory tract infection" also yields "upper respiratory tract infection",
    # "respiratory tract infection" and "infection": keep only the longest phrase.
    def _nested(e):
        return any(o is not e and o["entity_group"] == e["entity_group"] and e["entity_group"] in ("CONDITION", "DRUG")
                   and len(o["word"]) > len(e["word"]) and e["word"].lower() in o["word"].lower()
                   and abs(e["start"] - o["start"]) <= len(o["word"]) + 60 for o in kept)  # ML offsets can drift on long OCR text

    kept = [e for e in kept if not _nested(e)]
    result = sorted(kept, key=lambda x: x["start"])

    # Post-processing: fix truncated drug names and remove short duplicates
    return _postprocess_entities(result, text)


def _postprocess_entities(entities: List[Dict], text: str) -> List[Dict]:
    """Post-process entities to fix common issues."""
    result = []
    
    # Detect allergy context positions in the original text
    # Only flag drugs that appear DIRECTLY after 'allergic to' or 'allergy to'
    text_lower = text.lower()
    allergy_drug_ranges = []
    # "Allergic to X" — X is the allergen, not a prescribed drug
    for m in re.finditer(r'\ballergic\s+to\s+(\w+)', text_lower):
        drug_word = m.group(1)
        # Find this word's position in the text
        drug_match = re.search(r'\b' + re.escape(drug_word) + r'\b', text_lower[m.start():])
        if drug_match:
            abs_start = m.start() + drug_match.start()
            abs_end = abs_start + len(drug_word)
            allergy_drug_ranges.append((abs_start, abs_end))
    # "allergy to X" pattern
    for m in re.finditer(r'\ballerg(?:y|ies|ic)\s*(?:to|:|\||-|–)*\s*(\w+)', text_lower):
        drug_word = m.group(1)
        if drug_word in ('list', 'says', 'is', 'was', 'the', 'a', 'an'):
            continue  # Skip non-drug words after 'allergy'
        drug_match = re.search(r'\b' + re.escape(drug_word) + r'\b', text_lower[m.start():])
        if drug_match:
            abs_start = m.start() + drug_match.start()
            abs_end = abs_start + len(drug_word)
            allergy_drug_ranges.append((abs_start, abs_end))
    
    def _is_in_allergy_context(ent_start, ent_end):
        """Check if entity position falls in an allergy context."""
        for rng_start, rng_end in allergy_drug_ranges:
            if rng_start <= ent_start <= rng_end:
                return True
        return False
    
    # Known drug name fixes (ML truncation, subword issues)
    DRUG_NAME_FIXES = {
        "penici": "penicillin",
        "am": "amoxicillin",
        "war": "warfarin",
        "as": "aspirin",
        "do": None,  # remove (ML artifact from Dolo)
        "lis": "lisinopril",
        "met": "metformin",
        "ato": "atorvastatin",
        "los": "losartan",
        "ome": "omeprazole",
        "dic": "diclofenac",
        "flu": "fluoxetine",
        "fluco": "fluconazole",
        "para": "paracetamol",
        "naprox": "naproxen",
        "ibupro": "ibuprofen",
        "celecox": "celecoxib",
        "meloxi": "meloxicam",
        "metopro": "metoprolol",
        "atenol": "atenolol",
        "proprano": "propranolol",
        "carvedi": "carvedilol",
        "ciproflo": "ciprofloxacin",
        "levoflo": "levofloxacin",
        "pantopr": "pantoprazole",
        "esomepr": "esomeprazole",
        "morphi": "morphine",
        "oxycodo": "oxycodone",
        "hydroco": "hydrocodone",
        "tramado": "tramadol",
        "sertral": "sertraline",
        "fluoxet": "fluoxetine",
        "escital": "escitalopram",
        "lorazep": "lorazepam",
        "diazep": "diazepam",
        "alprazol": "alprazolam",
        "clonazep": "clonazepam",
    }
    
    # Generic nouns that are never a specific drug. The fine-tuned model occasionally
    # tags them as DRUG at low confidence (e.g. "Prescription" 0.64), which would hide
    # the "dosing instructions but no drug identified" check downstream.
    GENERIC_NON_DRUG_WORDS = {
        "prescription", "prescriptions", "rx", "tablet", "tablets", "capsule", "capsules",
        "pill", "pills", "dose", "doses", "dosage", "injection", "syrup", "treatment",
        "therapy", "patient", "order", "script",
    }

    for ent in entities:
        word = ent["word"]
        word_lower = word.lower()

        if ent["entity_group"] == "DRUG" and word_lower.strip(" .,'\"") in GENERIC_NON_DRUG_WORDS:
            continue

        # Skip DRUG entities that appear in allergy context
        if ent["entity_group"] == "DRUG" and _is_in_allergy_context(ent.get("start", 0), ent.get("end", 0)):
            continue
        
        # Skip very short ML artifacts (e.g., "do", "am")
        if ent["entity_group"] == "DRUG" and len(word) <= 2:
            # Check if it's in our fix list
            if word_lower in DRUG_NAME_FIXES:
                fix = DRUG_NAME_FIXES[word_lower]
                if fix is None:
                    continue  # Remove this entity
                ent = dict(ent)  # copy
                ent["word"] = fix
            else:
                continue  # Skip very short unknown tokens
        
        # Fix truncated drug names
        if ent["entity_group"] == "DRUG" and word_lower in DRUG_NAME_FIXES:
            fix = DRUG_NAME_FIXES[word_lower]
            if fix:
                ent = dict(ent)
                ent["word"] = fix
        
        # Remove duplicate DRUG entries where one is substring of other
        if ent["entity_group"] == "DRUG":
            is_dup = False
            for other in result:
                if other["entity_group"] == "DRUG":
                    other_lower = other["word"].lower()
                    if (word_lower != other_lower and
                        (word_lower in other_lower or other_lower in word_lower) and
                        len(word_lower) < len(other_lower)):
                        is_dup = True
                        break
            if is_dup:
                continue
        
        # Tighten DOSAGE entities: strip trailing non-dosage text
        if ent["entity_group"] == "DOSAGE":
            word = ent["word"]
            # Match only the core dosage: number + unit
            dosage_match = re.match(r'^((?:\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml|units?|iu|meq|%|tab[s]?))', word, re.IGNORECASE)
            if dosage_match:
                cleaned = dosage_match.group(1).strip()
                if cleaned != word:
                    ent = dict(ent)
                    ent["word"] = cleaned
                    ent["end"] = ent["start"] + len(cleaned)
            else:
                # Try to extract just the leading number + optional unit
                num_match = re.match(r'^(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu|meq|%)?\b', word, re.IGNORECASE)
                if num_match:
                    cleaned = num_match.group(0).strip()
                    if cleaned != word:
                        ent = dict(ent)
                        ent["word"] = cleaned
                        ent["end"] = ent["start"] + len(cleaned)
                elif not re.match(r'^\d+', word):
                    # Not a numeric dosage — skip it (e.g., "three times daily")
                    continue
        
        result.append(ent)
    
    return result


def _extract_entities_regex(text: str) -> List[Dict]:
    """Regex-based entity extraction (fallback)."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
    else:
        doc = None

    entities = []
    text_lower = text.lower()

    # 1. DOSAGE patterns
    dosage_pattern = r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu|meq|%)\b'
    for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
        entities.append({
            "word": match.group(),
            "entity_group": "DOSAGE",
            "score": 0.95,
            "start": match.start(),
            "end": match.end()
        })

    # 2. DRUGS from dictionary (case-insensitive match, preserve original case)
    # First, find allergy context positions to skip
    allergy_positions = set()
    allergy_patterns = [
        r'\ballerg(?:y|ic)\s+(?:to|:)?\s+\b(\w+)',
        r'\bknown\s+allerg(?:y|ic)\s+(?:to|:)?\s+\b(\w+)',
        r'\bno\s+known\s+allerg',
        r'\bpenicillin\s+allerg',
    ]
    for ap in allergy_patterns:
        for m in re.finditer(ap, text_lower):
            for drug in KNOWN_DRUGS:
                dp = r'\b' + re.escape(drug) + r'\b'
                for dm in re.finditer(dp, text_lower):
                    if m.end() <= dm.start() and dm.start() - m.end() < 50:
                        allergy_positions.add((dm.start(), dm.end()))

    for drug in KNOWN_DRUGS:
        pattern = r'\b' + re.escape(drug) + r'\b'
        for match in re.finditer(pattern, text_lower):
            if (match.start(), match.end()) in allergy_positions:
                continue
            original = text[match.start():match.end()]
            entities.append({
                "word": original,
                "entity_group": "DRUG",
                "score": 0.92,
                "start": match.start(),
                "end": match.end()
            })

    # 2b. Hyphenated combination drugs (e.g., sulfamethoxazole-trimethoprim)
    hyphen_pattern = re.compile(r'\b([a-z]+-tr methoprim|[a-z]+-clavulanate|[a-z]+-sulbactam|[a-z]+-sodium|[a-z]+-potassium)\b')
    # Simpler: match any drug-word hyphen drug-word pattern
    hyphen_drug_pattern = re.compile(r'\b([a-z]{4,})-([a-z]{4,})\b')
    for match in hyphen_drug_pattern.finditer(text_lower):
        full = match.group()
        part1 = match.group(1)
        part2 = match.group(2)
        # Check if either part is a known drug
        if part1 in KNOWN_DRUGS or part2 in KNOWN_DRUGS:
            # Check not already in allergy context
            if (match.start(), match.end()) not in allergy_positions:
                original = text[match.start():match.end()]
                entities.append({
                    "word": original,
                    "entity_group": "DRUG",
                    "score": 0.88,
                    "start": match.start(),
                    "end": match.end()
                })

    # 3. CONDITIONS from dictionary
    for condition in KNOWN_CONDITIONS:
        pattern = r'\b' + re.escape(condition) + r'\b'
        for match in re.finditer(pattern, text_lower):
            original = text[match.start():match.end()]
            entities.append({
                "word": original,
                "entity_group": "CONDITION",
                "score": 0.90,
                "start": match.start(),
                "end": match.end()
            })

    # 4. Indian drug brand name matching
    matched_positions = set()
    for ent in entities:
        matched_positions.add((ent["start"], ent["end"]))

    for brand, generic in INDIAN_DRUG_BRANDS.items():
        if generic is None:  # Skip form markers
            continue
        pattern = r'\b' + re.escape(brand) + r'\b'
        for match in re.finditer(pattern, text_lower):
            if (match.start(), match.end()) not in matched_positions:
                original = text[match.start():match.end()]
                entities.append({
                    "word": f"{original} ({generic})",
                    "entity_group": "DRUG",
                    "score": 0.90,
                    "start": match.start(),
                    "end": match.end(),
                    "indian_brand": True,
                    "generic_name": generic,
                })
                matched_positions.add((match.start(), match.end()))

    # 5. Fuzzy drug matching
    fuzzy_matches = _fuzzy_match_drug(text_lower)
    for fm in fuzzy_matches:
        if (fm["start"], fm["end"]) not in matched_positions:
            entities.append(fm)
            matched_positions.add((fm["start"], fm["end"]))

    # 6. Deduplicate
    seen = {}
    for ent in entities:
        key = (ent["word"].lower(), ent["entity_group"])
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent

    entities = sorted(seen.values(), key=lambda x: x["start"])
    return entities


def _fuzzy_match_drug(text_lower: str) -> List[Dict]:
    findings = []

    # Broader drug suffixes including Indian brand patterns
    drug_suffixes = [
        'mab', 'nib', 'vir', 'azole', 'prazole',
        'olol', 'sartan', 'pril', 'pine',
        'statin', 'cillin', 'mycin', 'floxacin', 'oxacin', 'cycin',
        'lone', 'zone', 'lam', 'zepam', 'olam',
        'zine', 'sure', 'dine', 'mide', 'ide', 'ine',
        'fen', 'cin', 'dex', 'zol', 'ryn',
    ]

    # Match: Tab./Inj./Cap./Syrup + Word (common prescription patterns)
    drug_pattern = re.compile(
        r'(?:tab\.?|inj\.?|cap\.?|syrup\.?|drop[s]?\.?|cream\.?|gel\.?|ointment\.?)\s+'
        r'([A-Za-z][A-Za-z0-9\-]+)'
    )

    suffix_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:[a-z]+)*(?:' + '|'.join(drug_suffixes) + r'))\b'
    )

    skip_words = {'patient', 'doctor', 'hospital', 'clinic', 'pharmacy',
                  'tablets', 'capsule', 'tablet', 'capsules', 'prescription',
                  'twice', 'daily', 'weekly', 'monthly', 'morning', 'evening',
                  'before', 'after', 'with', 'without', 'known', 'history',
                  'healing', 'hands', 'street', 'road', 'delhi', 'chennai',
                  'physician', 'doctor', 'reg', 'number', 'phone', 'mobile',
                  'address', 'date', 'time', 'name', 'gender', 'patient',
                  'ayurvedic', 'consulting', 'counseling', 'kottakkal',
                  'arya', 'vidya', 'sala', 'nangallur', 'chennai',
                  'medical', 'superintendent', 'deputy', 'medication',
                  'drug', 'medications', 'drugs', 'strength', 'tablets',
                  'antibiotic', 'antibiotics', 'new', 'severe'}

    # Pattern 1: Tab./Inj. + DrugName (highest confidence)
    for match in drug_pattern.finditer(text_lower):
        drug_name = match.group(1)
        if drug_name.lower() not in skip_words and len(drug_name) > 2:
            findings.append({
                "word": drug_name,
                "entity_group": "DRUG",
                "score": 0.75,
                "start": match.start(1),
                "end": match.end(1)
            })

    # Pattern 2: Words with common drug suffixes
    for match in suffix_pattern.finditer(text_lower):
        word = match.group()
        if word.lower() not in skip_words and len(word) > 4:
            # Avoid double-matching from pattern 1
            already_found = any(
                f["start"] <= match.start() and f["end"] >= match.end()
                for f in findings
            )
            if not already_found:
                findings.append({
                    "word": text_lower[match.start():match.end()],
                    "entity_group": "DRUG",
                    "score": 0.60,
                    "start": match.start(),
                    "end": match.end()
                })

    return findings


# ============================================================
# Relationship Extraction
# ============================================================

def _spans_in(sentence_lower: str, word: str):
    """All (start, end) occurrences of `word` in the sentence. Pipeline offsets are unreliable, so we search."""
    w = re.sub(r"\s*\([^)]*\)\s*$", "", word.lower().strip())  # 'azithral (azithromycin)' -> 'azithral'
    out, i = [], 0
    while w:
        i = sentence_lower.find(w, i)
        if i < 0:
            break
        out.append((i, i + len(w)))
        i += len(w)
    return out


def _gap(a, b) -> int:
    """Character gap between two spans (0 if they touch/overlap)."""
    if b[0] >= a[1]:
        return b[0] - a[1]
    if a[0] >= b[1]:
        return a[0] - b[1]
    return 0


def _assign_dosages(drug_spans, dosage_spans):
    """Give each dosage to its closest drug (by character gap; ties favour the drug BEFORE the dose,
    as in 'Warfarin 5mg'). Returns {drug_word: dosage_word}."""
    best = {}
    for xword, xspan in dosage_spans:
        if not drug_spans:
            break
        gap, _, dword = min((_gap(dspan, xspan), 0 if dspan[1] <= xspan[0] else 1, dword)
                            for dword, dspan in drug_spans)
        if dword not in best or gap < best[dword][0]:
            best[dword] = (gap, xword)
    return {d: x for d, (_, x) in best.items()}


def extract_relationships(entities: List[Dict], original_text: str) -> List[Dict]:
    """Link drugs to doses and conditions within each sentence.

    * each condition is linked to the nearest drug (preferring the drug mentioned before it,
      e.g. 'Metformin 500mg for Type 2 Diabetes')
    * each dose is attached to the closest drug, not simply the first dose in the sentence
    * co-prescribed pairs are flagged as potential interactions only if the pair is a known interaction
    """
    relationships = []

    if not entities or not original_text:
        return relationships

    from module5_anomaly import AnomalyDetector  # local import: module5 does not import module2
    detector = AnomalyDetector()

    sentences = re.split(r'[.;!?]', original_text)
    causal_re = re.compile(r'\b(for|treating|treat|prescribed|diagnosed|because|due to)\b')
    temporal_re = re.compile(r'\b(before|after|during|then|subsequently|followed by)\b')

    # A condition introduced by "Diagnosis:" / "Dx:" / "Indication:" belongs to the whole prescription,
    # not to whichever drug happens to be nearest. Link it to every prescribed drug.
    all_low = original_text.lower()
    diag_cue = re.compile(r'(?:diagnosis|dx|impression|indication|assessment)\s*[:\-]?\s*(?:[\w/,()-]+\s+){0,6}$')
    diag_conditions = set()
    for ent in entities:
        if ent['entity_group'] != 'CONDITION':
            continue
        for a, _b in _spans_in(all_low, ent['word']):
            if diag_cue.search(all_low[max(0, a - 70):a]):
                diag_conditions.add(ent['word'].lower())
    if diag_conditions:
        g_drugs, g_doses, gseen = [], [], set()
        for ent in entities:
            if ent['entity_group'] in ('DRUG', 'DOSAGE') and (ent['word'].lower(), ent['entity_group']) not in gseen:
                gseen.add((ent['word'].lower(), ent['entity_group']))
                (g_drugs if ent['entity_group'] == 'DRUG' else g_doses).extend(
                    (ent['word'], sp) for sp in _spans_in(all_low, ent['word']))
        g_drugs.sort(key=lambda x: x[1][0])
        g_dose_of = _assign_dosages(g_drugs, g_doses)
        for cond in sorted(diag_conditions):
            for dword in dict.fromkeys(d for d, _ in g_drugs):
                relationships.append({
                    "drug": dword, "condition": cond, "dosage": g_dose_of.get(dword, "N/A"),
                    "relationship_type": "prescribed_for", "confidence": 0.80,
                })

    for sentence in sentences:
        if not sentence.strip():
            continue
        low = sentence.lower()

        drugs, conds, doses = [], [], []
        seen = set()
        for ent in entities:
            key = (ent['word'].lower(), ent['entity_group'])
            if key in seen:
                continue
            spans = _spans_in(low, ent['word'])
            if not spans:
                continue
            seen.add(key)
            bucket = {'DRUG': drugs, 'CONDITION': conds, 'DOSAGE': doses}.get(ent['entity_group'])
            if bucket is not None:
                bucket.extend((ent['word'], sp) for sp in spans)
        drugs.sort(key=lambda x: x[1][0])

        has_causal = bool(causal_re.search(low))
        has_temporal = bool(temporal_re.search(low))
        dose_of = _assign_dosages(drugs, doses)

        for cword, cspan in conds:
            if not drugs or cword.lower() in diag_conditions:
                continue
            before = [d for d in drugs if d[1][1] <= cspan[0]]
            drug_word = (max(before, key=lambda d: d[1][1]) if before
                         else min(drugs, key=lambda d: _gap(d[1], cspan)))[0]
            rel_type = "causal" if has_causal else ("temporal" if has_temporal else "prescribed_for")
            relationships.append({
                "drug": drug_word,
                "condition": cword,
                "dosage": dose_of.get(drug_word, "N/A"),
                "relationship_type": rel_type,
                "confidence": 0.85 if has_causal else 0.70,
            })

        # Co-prescribed pairs
        pairs_done = set()
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                d1, d2 = drugs[i][0], drugs[j][0]
                if d1.lower() == d2.lower() or (d1.lower(), d2.lower()) in pairs_done:
                    continue
                pairs_done.add((d1.lower(), d2.lower()))
                relationships.append({
                    "drug": d1,
                    "drug_pair": [d1, d2],
                    "condition": "CO_PRESCRIBED",
                    "dosage": f"{dose_of.get(d1, 'N/A')} + {dose_of.get(d2, 'N/A')}",
                    "relationship_type": "co_prescribed",
                    "confidence": 0.80,
                    "potential_interaction": bool(detector.check_drug_interactions([d1, d2])),
                })

    return relationships


_DOSE_WITH_UNIT = re.compile(r"\d\s*(?:mg|mcg|µg|ug|g|kg|ml|l|iu|units?|%|puffs?|drops?|tabs?|tablets?|caps?|capsules?)\b", re.I)
_BARE_DOSE = re.compile(r"^[ \t]{1,3}(\d{2,4})(?![\d/-])(?=[ \t(]|$)(?!\s*-)")


def drug_dose_map(entities: List[Dict], text: str) -> Dict[str, str]:
    """{drug_lower: dose} - each dose goes to the CLOSEST drug by character gap in the text.

    Positions are found by searching the text (NER offsets are unreliable). A bare number right after
    a drug name ("Mox 500", "Pantodac 40") is accepted as mg if it is a known strength of that drug.
    """
    from drug_knowledge import DRUG_INFO
    from module11_indian_rx import to_generic

    low = text.lower()
    drugs, doses, seen = [], [], set()
    for ent in entities:
        key = (ent["word"].lower(), ent["entity_group"])
        if ent["entity_group"] not in ("DRUG", "DOSAGE") or key in seen:
            continue
        seen.add(key)
        bucket = drugs if ent["entity_group"] == "DRUG" else doses
        bucket.extend((ent["word"], sp) for sp in _spans_in(low, ent["word"]))
    doses = [d for d in doses if _DOSE_WITH_UNIT.search(d[0])
             and not re.match(r"\s*/\s*\d", text[d[1][1]:d[1][1] + 4])]  # "250 mg/5 ml" is a strength, not the dose
    mapping = {k.lower(): v for k, v in _assign_dosages(drugs, doses).items()}

    for word, span in drugs:
        if word.lower() in mapping:
            continue
        m = _BARE_DOSE.match(text[span[1]:span[1] + 12])
        if m:
            known = DRUG_INFO.get(to_generic(word), {}).get("common_dosages", [])
            if f"{m.group(1)}mg" in known:
                mapping[word.lower()] = f"{m.group(1)}mg"
    return mapping


def build_medications(entities: List[Dict], text: str, relationships: List[Dict] = None) -> List[Dict]:
    """Medication list for the safety engine: one entry per drug in text order, brand names mapped to
    generics (so 'Crocin' is checked as paracetamol), each with its own dose."""
    from module11_indian_rx import to_generic

    dose_of = drug_dose_map(entities, text)
    rel_dose = {}
    for r in relationships or []:
        if r.get("relationship_type") != "co_prescribed" and r.get("dosage") not in (None, "", "N/A"):
            rel_dose.setdefault(r["drug"].lower(), r["dosage"])
    low = text.lower()
    drugs = sorted({e["word"] for e in entities if e["entity_group"] == "DRUG"},
                   key=lambda w: (low.find(w.lower()) if low.find(w.lower()) >= 0 else 10 ** 9))
    import difflib
    from drug_knowledge import DRUG_INFO, DRUG_NAMES

    form_words = re.compile(r"\s+(syrup|syr|susp|suspension|tablets?|tabs?|capsules?|caps?|inj|injection|cream|gel|ointment|drops)$")
    strength = re.compile(r"\d\s*(?:mg|mcg|µg|ug|g|iu|units?)\b", re.I)
    names = sorted(DRUG_NAMES)

    def is_known(g):
        return g in DRUG_NAMES or g in DRUG_INFO

    meds, seen = [], set()
    for word in drugs:
        base = form_words.sub("", word.lower().strip())
        generic = to_generic(base)
        dosage = dose_of.get(word.lower()) or rel_dose.get(word.lower(), "")
        if not is_known(generic):
            # OCR slip? "thiocolchicosicde" -> "thiocolchicoside"
            close = difflib.get_close_matches(generic, names, n=1, cutoff=0.86) if len(generic) > 6 else []
            if close:
                generic = to_generic(close[0])
            elif not (dosage and strength.search(dosage)):
                continue  # "Chennai", "fluids", "salt diet", a surname...: not a prescribed drug
        if dosage and not _DOSE_WITH_UNIT.search(dosage):
            dosage = ""  # bare numbers ("10", "50", "600") are not doses
        if generic in seen:
            continue
        seen.add(generic)
        meds.append({"drug": generic, "brand": word if generic != word.lower() else "", "dosage": dosage})
    return meds


def format_entities_for_display(entities: List[Dict]) -> str:
    lines = []
    for ent in entities:
        lines.append(
            f"Found: {ent['word']:20s} | "
            f"Type: {ent['entity_group']:15s} | "
            f"Confidence: {ent['score']:.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = (
        "Patient prescribed 500mg of Amoxicillin for pneumonia. "
        "History of Type 2 Diabetes. Allergic to Penicillin."
    )
    print(f"Input Text: {sample_text}\n")
    entities = extract_medical_entities(sample_text)
    print("Extracted Entities:")
    print(format_entities_for_display(entities))
    print("\nRelationships:")
    relationships = extract_relationships(entities, sample_text)
    for r in relationships:
        print(f"  {r['drug']} ({r['dosage']}) -> {r['relationship_type']} -> {r['condition']}")

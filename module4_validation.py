"""
Module 4: Hallucination Prevention Layer
=========================================
A "fact checker" that compares every drug and condition against
real medical databases. If the AI made something up, we catch it here.
"""

from typing import Tuple, Dict, List
from drug_knowledge import DRUG_INFO, CONDITION_NAMES

# ============================================================
# Medical Knowledge Base (imported from shared module)
# ============================================================

KNOWN_DRUGS = DRUG_INFO  # Alias for backward compatibility
KNOWN_CONDITIONS = CONDITION_NAMES  # Alias for backward compatibility


def validate_entity(entity_text: str, entity_type: str) -> Tuple[bool, str]:
    """
    Check if an extracted entity actually exists in our medical knowledge base.

    This is the safety layer - we NEVER blindly trust AI output.

    Args:
        entity_text: The extracted entity (e.g., "Amoxicillin")
        entity_type: Type of entity ("DRUG" or "CONDITION")

    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    normalized = entity_text.lower().strip()

    if entity_type == "DRUG":
        from module11_indian_rx import to_generic
        generic = to_generic(normalized)
        if normalized not in KNOWN_DRUGS and generic in KNOWN_DRUGS:
            return True, f"[VALID] '{entity_text}' is a brand name of {generic} ({KNOWN_DRUGS[generic]['type']})"
        if normalized in KNOWN_DRUGS:
            return True, f"[VALID] '{entity_text}' is a verified drug ({KNOWN_DRUGS[normalized]['type']})"
        else:
            import difflib
            close = difflib.get_close_matches(normalized, list(KNOWN_DRUGS), n=1, cutoff=0.85)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            return False, f"[FLAGGED] '{entity_text}' NOT FOUND in drug database{hint} -- flagged for human review"

    elif entity_type == "CONDITION":
        if normalized in KNOWN_CONDITIONS:
            return True, f"[VALID] '{entity_text}' is a recognized medical condition"
        else:
            return False, f"[FLAGGED] '{entity_text}' NOT FOUND in condition database -- flagged for human review"

    return True, "No validation rule for this entity type"


def validate_dosage(drug: str, dosage: str) -> Tuple[bool, str]:
    """
    Check if the dosage makes sense for this drug.

    Args:
        drug: Drug name
        dosage: Dosage to validate (e.g., "500mg")

    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    normalized_drug = drug.lower().strip()

    if normalized_drug in KNOWN_DRUGS:
        valid_dosages = KNOWN_DRUGS[normalized_drug]["common_dosages"]
        if dosage.lower() in valid_dosages:
            return True, f"[VALID] {dosage} is a standard dosage for {drug}"
        else:
            return False, f"[WARNING] {dosage} is NOT a standard dosage for {drug}. Expected: {valid_dosages}"

    return False, f"[ERROR] Cannot validate dosage -- drug '{drug}' not in database"


def validate_all_entities(entities: List[Dict]) -> List[Dict]:
    """
    Validate a list of extracted entities and add validation status.

    Args:
        entities: List of entity dicts from Module 2

    Returns:
        List of entity dicts with added 'validated' and 'reason' fields
    """
    validated_results = []

    for ent in entities:
        is_valid, reason = validate_entity(ent['word'], ent['entity_group'])
        validated_results.append({
            **ent,
            "validated": is_valid,
            "reason": reason
        })

    return validated_results


if __name__ == "__main__":
    # Demo: Validate various entities
    print("=" * 60)
    print("Entity Validation Tests")
    print("=" * 60)

    # Test drug validation
    print(validate_entity("Amoxicillin", "DRUG"))
    print(validate_entity("Paracetamol", "DRUG"))
    print(validate_entity("Flurbozine", "DRUG"))

    # Test condition validation
    print(validate_entity("Pneumonia", "CONDITION"))
    print(validate_entity("Quantum Flux Syndrome", "CONDITION"))

    # Test dosage validation
    print(validate_dosage("Amoxicillin", "500mg"))
    print(validate_dosage("Paracetamol", "500mg"))
    print(validate_dosage("Amoxicillin", "999mg"))

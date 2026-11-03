"""
Compare Fine-Tuned Model vs Original
======================================
Tests both models on the same prescription texts and compares results.
"""

from module2_ner import extract_medical_entities

test_cases = [
    "Tab Amoxicillin 500mg twice daily for 7 days",
    "Inj Ceftriaxone 1g IV once daily for Pneumonia",
    "Patient prescribed Warfarin 5mg daily for Atrial Fibrillation",
    "Tab Metformin 500mg with meals twice daily for Type 2 Diabetes",
    "Cream Betamethasone 0.05% apply to affected area twice daily for 2 weeks",
    "Tab Sertraline 50mg once daily for Depression and Anxiety",
    "Inj Insulin Glargine 20 units subcut at bedtime for Type 1 Diabetes",
    "Tab Levothyroxine 75mcg empty stomach 30 min before breakfast for Hypothyroidism",
    "Tab Clopidogrel 75mg once daily after MI",
    "Cap Pregabalin 75mg twice daily for Neuropathic Pain",
]

print("=" * 70)
print("NER Entity Extraction Test")
print("=" * 70)

drugs_found = 0
dosages_found = 0
conditions_found = 0

for i, text in enumerate(test_cases, 1):
    entities = extract_medical_entities(text)
    
    drugs = [e for e in entities if e["entity_group"] == "DRUG"]
    dosages = [e for e in entities if e["entity_group"] == "DOSAGE"]
    conditions = [e for e in entities if e["entity_group"] == "CONDITION"]
    
    drugs_found += len(drugs)
    dosages_found += len(dosages)
    conditions_found += len(conditions)
    
    print(f"\n  {i}. {text}")
    print(f"     Drugs: {[e['word'] for e in drugs]}")
    print(f"     Dosages: {[e['word'] for e in dosages]}")
    print(f"     Conditions: {[e['word'] for e in conditions]}")
    print(f"     Total entities: {len(entities)}")

print(f"\n{'=' * 70}")
print(f"Summary:")
print(f"  Total test cases: {len(test_cases)}")
print(f"  Drugs found: {drugs_found}")
print(f"  Dosages found: {dosages_found}")
print(f"  Conditions found: {conditions_found}")
print(f"  Average entities per prescription: {(drugs_found + dosages_found + conditions_found) / len(test_cases):.1f}")
print(f"{'=' * 70}")

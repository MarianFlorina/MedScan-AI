"""
Test Suite for New AI Modules
==============================
Tests module8 (ML Anomaly), module9 (Medical QA), module10 (Advanced NER)
"""

import sys
import time

def test_module8_ml_anomaly():
    """Test ML-based clinical anomaly classification."""
    print("=" * 70)
    print("MODULE 8: ML Anomaly Classification (bart-large-mnli)")
    print("=" * 70)
    
    from module8_ml_anomaly import classify_clinical_safety, get_model_status
    
    status = get_model_status()
    print(f"  Model: {status.get('model', 'N/A')}")
    print(f"  Available: {status.get('available', False)}")
    
    test_cases = [
        {
            "name": "Drug interaction (Warfarin + Aspirin + Ibuprofen)",
            "text": "Patient prescribed Warfarin 10mg and Aspirin 81mg. Also taking Ibuprofen 400mg for pain.",
            "expected_severity": "CRITICAL",
        },
        {
            "name": "Allergy conflict (Amoxicillin + Penicillin allergy)",
            "text": "Patient prescribed Amoxicillin 500mg. History of anaphylaxis to penicillin.",
            "expected_severity": "CRITICAL",
        },
        {
            "name": "Diabetes monitoring",
            "text": "Patient taking Insulin but has repeated episodes of low blood sugar.",
            "expected_severity": "HIGH",
        },
        {
            "name": "Mild concern (rash, no breathing difficulty)",
            "text": "Patient reports rash after starting a new medication but has no breathing difficulty.",
            "expected_severity": "MODERATE",
        },
    ]
    
    passed = 0
    for tc in test_cases:
        start = time.time()
        findings = classify_clinical_safety(tc["text"])
        elapsed = time.time() - start
        
        has_expected = any(f["severity"] == tc["expected_severity"] for f in findings)
        status = "PASS" if has_expected else "PARTIAL"
        
        print(f"\n  [{status}] {tc['name']} ({elapsed:.1f}s)")
        for f in findings:
            print(f"    [{f['severity']:8s}] {f['type']:35s} (ML: {f['ml_confidence']:.0%})")
        
        if has_expected:
            passed += 1
    
    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_module9_medical_qa():
    """Test Medical Q&A assistant."""
    print("\n" + "=" * 70)
    print("MODULE 9: Medical Q&A Assistant")
    print("=" * 70)
    
    from module9_medical_qa import answer_clinical_question, get_model_status
    
    status = get_model_status()
    print(f"  ML Model: {status.get('model', 'N/A')}")
    print(f"  ML Available: {status.get('available', False)}")
    print(f"  Rule-based questions: {status.get('rule_based_questions', 0)}")
    
    sample_text = (
        "Patient prescribed 500mg of Amoxicillin three times daily for pneumonia. "
        "Also taking Metformin 500mg for Type 2 Diabetes. "
        "History: Hypertension, Chronic Kidney Disease. "
        "Allergies: Penicillin, Sulfa drugs."
    )
    sample_meds = [
        {"drug": "Amoxicillin", "dosage": "500mg"},
        {"drug": "Metformin", "dosage": "500mg"},
    ]
    sample_entities = [
        {"word": "pneumonia", "entity_group": "CONDITION"},
        {"word": "Type 2 Diabetes", "entity_group": "CONDITION"},
        {"word": "Hypertension", "entity_group": "CONDITION"},
    ]
    
    questions = [
        ("What medications are prescribed?", "Amoxicillin", "list_medications"),
        ("What are the dosages?", "500mg", "list_dosages"),
        ("Are there any drug interactions?", "interaction", "check_interactions"),
        ("Are there any allergies?", "Penicillin", "check_allergies"),
        ("Is this prescription safe?", "safe", "safety_overview"),
        ("What monitoring is needed?", "monitoring", "monitoring_recommendations"),
        ("What conditions does the patient have?", "pneumonia", "list_conditions"),
        ("What should the patient avoid?", "instructions", "patient_instructions"),
    ]
    
    passed = 0
    for q, expected_keyword, expected_method in questions:
        start = time.time()
        result = answer_clinical_question(
            question=q,
            text=sample_text,
            medications=sample_meds,
            entities=sample_entities,
            allergies=["penicillin", "sulfa drugs"],
        )
        elapsed = time.time() - start
        
        method_ok = result["method"] == expected_method
        has_content = len(result["answer"]) > 10
        status = "PASS" if (method_ok and has_content) else "FAIL"
        
        print(f"\n  [{status}] Q: {q}")
        print(f"    Method: {result['method']} (expected: {expected_method})")
        print(f"    Confidence: {result['confidence']:.0%}")
        print(f"    Answer preview: {result['answer'][:120]}...")
        
        if status == "PASS":
            passed += 1
    
    print(f"\n  Results: {passed}/{len(questions)} passed")
    return passed == len(questions)


def test_module10_advanced_ner():
    """Test Advanced NER & Relation Extraction."""
    print("\n" + "=" * 70)
    print("MODULE 10: Advanced Clinical NER & Relation Extraction")
    print("=" * 70)
    
    from module10_clinical_ner_advanced import (
        normalize_drug_name, normalize_allergy_name,
        extract_treatment_relationships, extract_contraindication_relationships,
        analyze_clinical_context, get_model_status,
    )
    
    status = get_model_status()
    print(f"  Drug Ontology: {status['drug_ontology_size']} mappings")
    print(f"  Allergy Ontology: {status['allergy_ontology_size']} mappings")
    print(f"  Treatment Pairs: {status['drug_condition_treatments']}")
    print(f"  Contraindications: {status['drug_condition_contraindications']}")
    
    passed = 0
    total = 4
    
    # Test 1: Drug normalization
    print("\n  Test 1: Drug Name Normalization")
    test_drugs = [
        ("Lipitor", "atorvastatin"),
        ("Tylenol", "acetaminophen"),
        ("Advil", "ibuprofen"),
        ("Glucophage", "metformin"),
        ("Coumadin", "warfarin"),
        ("Zestril", "lisinopril"),
        ("Zoloft", "sertraline"),
        ("Prilosec", "omeprazole"),
    ]
    norm_passed = 0
    for brand, expected_generic in test_drugs:
        result = normalize_drug_name(brand)
        ok = result == expected_generic
        if ok:
            norm_passed += 1
        print(f"    {brand:15s} -> {result:20s} (expected: {expected_generic}) {'OK' if ok else 'FAIL'}")
    
    if norm_passed == len(test_drugs):
        passed += 1
        print(f"  [PASS] Drug normalization: {norm_passed}/{len(test_drugs)}")
    else:
        print(f"  [FAIL] Drug normalization: {norm_passed}/{len(test_drugs)}")
    
    # Test 2: Treatment relationships
    print("\n  Test 2: Treatment Relationships")
    drugs = ["amoxicillin", "metformin", "lisinopril"]
    conditions = ["pneumonia", "Type 2 Diabetes", "hypertension"]
    rels = extract_treatment_relationships(drugs, conditions)
    print(f"    Found {len(rels)} treatment relationships")
    for r in rels:
        print(f"    {r['drug']:15s} -> treats -> {r['condition']:20s} ({r['confidence']:.0%})")
    if len(rels) >= 3:
        passed += 1
        print(f"  [PASS] Treatment relationships: {len(rels)} found")
    else:
        print(f"  [FAIL] Treatment relationships: expected >= 3, got {len(rels)}")
    
    # Test 3: Contraindication relationships
    print("\n  Test 3: Contraindication Relationships")
    contra = extract_contraindication_relationships(
        drugs=["ibuprofen", "lisinopril"],
        conditions=["kidney disease", "gastric ulcer"],
        allergies=["penicillin"],
    )
    print(f"    Found {len(contra)} contraindications")
    for c in contra:
        target = c.get("condition", c.get("allergy", "N/A"))
        print(f"    {c['drug']:15s} -> contraindicated with -> {target}")
    if len(contra) >= 2:
        passed += 1
        print(f"  [PASS] Contraindications: {len(contra)} found")
    else:
        print(f"  [FAIL] Contraindications: expected >= 2, got {len(contra)}")
    
    # Test 4: Clinical context analysis
    print("\n  Test 4: Clinical Context Analysis")
    entities = [
        {"word": "Amoxicillin", "entity_group": "DRUG", "start": 25, "end": 35, "score": 0.9},
        {"word": "Metformin", "entity_group": "DRUG", "start": 80, "end": 89, "score": 0.9},
        {"word": "pneumonia", "entity_group": "CONDITION", "start": 55, "end": 64, "score": 0.85},
        {"word": "Type 2 Diabetes", "entity_group": "CONDITION", "start": 100, "end": 115, "score": 0.85},
    ]
    medications = [
        {"drug": "Amoxicillin", "dosage": "500mg"},
        {"drug": "Metformin", "dosage": "500mg"},
    ]
    context = analyze_clinical_context(
        text="Patient prescribed Amoxicillin 500mg for pneumonia. Also Metformin 500mg for Type 2 Diabetes.",
        entities=entities,
        medications=medications,
    )
    print(f"    Normalized drugs: {context['normalized_drugs']}")
    print(f"    Treatment relationships: {len(context['treatment_relationships'])}")
    print(f"    Contraindications: {len(context['contraindication_relationships'])}")
    print(f"    Clinical confidence: {context['clinical_confidence']:.0%}")
    if context["drug_count"] == 2 and context["condition_count"] == 2:
        passed += 1
        print(f"  [PASS] Clinical context analysis")
    else:
        print(f"  [FAIL] Clinical context analysis: {context['drug_count']} drugs, {context['condition_count']} conditions")
    
    print(f"\n  Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    results = {}
    
    results["Module 8 (ML Anomaly)"] = test_module8_ml_anomaly()
    results["Module 9 (Medical QA)"] = test_module9_medical_qa()
    results["Module 10 (Advanced NER)"] = test_module10_advanced_ner()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for module, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {module}")
        if not passed:
            all_pass = False
    
    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL AI MODULES PASSING")
    else:
        print("SOME MODULES FAILED")
    print(f"{'=' * 70}")

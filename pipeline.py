"""
MedScan-AI Pipeline
====================
The full pipeline that wires all modules together:
Image → Text → Entities → Validation → FHIR
"""

from module1_ocr import extract_text_from_image
from module2_ner import extract_medical_entities, extract_relationships
from module3_fhir import create_fhir_medication_request, create_fhir_bundle, save_fhir_to_file
from module4_validation import validate_entity, validate_dosage, validate_all_entities


def run_pipeline(image_path: str) -> dict:
    """
    Full pipeline: Image → Text → Entities → Validation → FHIR
    
    Args:
        image_path: Path to the clinical document image
        
    Returns:
        Dictionary with all pipeline results
    """
    print("=" * 60)
    print("🏥 MedScan-AI Pipeline")
    print("=" * 60)
    
    # Step 1: OCR
    print("\n📝 Step 1: Extracting text from image...")
    raw_text = extract_text_from_image(image_path)
    print(f"   Extracted: {raw_text}\n")
    
    # Step 2: NER
    print("🏷️  Step 2: Identifying medical entities...")
    entities = extract_medical_entities(raw_text)
    relationships = extract_relationships(entities, raw_text)
    
    for ent in entities:
        print(f"   Found: {ent['word']:20s} | Type: {ent['entity_group']:15s} | Confidence: {ent['score']:.2f}")
    
    # Step 3: Validate
    print("\n🔍 Step 3: Validating against medical databases...")
    validated_results = validate_all_entities(entities)
    
    for ent in validated_results:
        print(f"   {ent['reason']}")
    
    # Step 4: Generate FHIR
    print("\n📋 Step 4: Generating FHIR output...")
    fhir_outputs = []
    for rel in relationships:
        fhir = create_fhir_medication_request(rel['drug'], rel['dosage'], rel['condition'])
        fhir_outputs.append(fhir)
        
        # Save each FHIR resource
        filename = f"medication_{rel['drug'].lower().replace(' ', '_')}.json"
        save_fhir_to_file(fhir, filename)
        print(f"   Saved: {filename}")
    
    # Create FHIR Bundle
    if relationships:
        bundle = create_fhir_bundle(relationships)
        save_fhir_to_file(bundle, "fhir_bundle.json")
        print(f"   Saved: fhir_bundle.json")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)
    
    # Summary
    valid_count = sum(1 for e in validated_results if e["validated"])
    print(f"\n📊 Summary:")
    print(f"   Total entities found: {len(entities)}")
    print(f"   Validated: {valid_count}")
    print(f"   Flagged for review: {len(entities) - valid_count}")
    print(f"   FHIR resources generated: {len(fhir_outputs)}")
    
    return {
        "raw_text": raw_text,
        "entities": entities,
        "relationships": relationships,
        "validated_entities": validated_results,
        "fhir_outputs": fhir_outputs
    }


def run_pipeline_from_text(text: str) -> dict:
    """
    Run pipeline with pre-extracted text (no OCR needed).
    
    Args:
        text: Clinical text to process
        
    Returns:
        Dictionary with all pipeline results
    """
    print("=" * 60)
    print("🏥 MedScan-AI Pipeline (Text Mode)")
    print("=" * 60)
    
    # Step 1: Use provided text
    print(f"\n📝 Input Text: {text}\n")
    
    # Step 2: NER
    print("🏷️  Step 2: Identifying medical entities...")
    entities = extract_medical_entities(text)
    relationships = extract_relationships(entities, text)
    
    for ent in entities:
        print(f"   Found: {ent['word']:20s} | Type: {ent['entity_group']:15s} | Confidence: {ent['score']:.2f}")
    
    # Step 3: Validate
    print("\n🔍 Step 3: Validating against medical databases...")
    validated_results = validate_all_entities(entities)
    
    for ent in validated_results:
        print(f"   {ent['reason']}")
    
    # Step 4: Generate FHIR
    print("\n📋 Step 4: Generating FHIR output...")
    fhir_outputs = []
    for rel in relationships:
        fhir = create_fhir_medication_request(rel['drug'], rel['dosage'], rel['condition'])
        fhir_outputs.append(fhir)
        
        filename = f"medication_{rel['drug'].lower().replace(' ', '_')}.json"
        save_fhir_to_file(fhir, filename)
        print(f"   Saved: {filename}")
    
    if relationships:
        bundle = create_fhir_bundle(relationships)
        save_fhir_to_file(bundle, "fhir_bundle.json")
        print(f"   Saved: fhir_bundle.json")
    
    print("\n✅ Pipeline complete!")
    
    return {
        "raw_text": text,
        "entities": entities,
        "relationships": relationships,
        "validated_entities": validated_results,
        "fhir_outputs": fhir_outputs
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run with image file
        results = run_pipeline(sys.argv[1])
    else:
        # Demo with sample text
        print("No image provided. Running demo with sample text...\n")
        sample_text = (
            "Patient prescribed 500mg of Amoxicillin for pneumonia. "
            "History of Type 2 Diabetes. Allergic to Penicillin."
        )
        results = run_pipeline_from_text(sample_text)

"""
Module 3: Structured FHIR Output
==================================
Converts extracted medical data into FHIR-compliant JSON format.
FHIR is the universal standard that hospital systems use worldwide.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional


def create_fhir_medication_request(
    drug: str,
    dosage: str,
    condition: str,
    patient_id: str = "patient-001"
) -> Dict:
    """
    Create a FHIR-compliant MedicationRequest resource.
    
    This is the EXACT format hospitals use worldwide.
    
    Analogy: Like writing a prescription in the universal language
    that every hospital system can read, regardless of country.
    
    Args:
        drug: Name of the medication
        dosage: Dosage amount (e.g., "500mg")
        condition: Medical condition being treated
        patient_id: Patient identifier
        
    Returns:
        FHIR-compliant MedicationRequest dictionary
    """
    # Extract numeric dosage value
    dosage_value = 0
    if any(c.isdigit() for c in dosage):
        dosage_value = int(''.join(filter(str.isdigit, dosage)))
    
    fhir_data = {
        # 1. What type of FHIR resource is this?
        "resourceType": "MedicationRequest",
        
        # 2. Current status of this prescription
        "status": "active",
        
        # 3. Is this an order, a plan, or a proposal?
        "intent": "order",
        
        # 4. What drug is being prescribed?
        "medicationCodeableConcept": {
            "coding": [{
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "display": drug
            }],
            "text": drug
        },
        
        # 5. Who is the patient?
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        
        # 6. When was this prescribed?
        "authoredOn": datetime.now().isoformat(),
        
        # 7. How should they take it?
        "dosageInstruction": [{
            "text": f"{dosage} for {condition}",
            "doseAndRate": [{
                "doseQuantity": {
                    "value": dosage_value,
                    "unit": "mg",
                    "system": "http://unitsofmeasure.org"
                }
            }]
        }],
        
        # 8. Why was this prescribed?
        "reasonCode": [{
            "text": condition
        }]
    }
    
    return fhir_data


def create_fhir_bundle(relationships: List[Dict]) -> Dict:
    """
    Create a FHIR Bundle containing multiple MedicationRequests.
    
    A Bundle is a container for multiple FHIR resources,
    useful when a patient has multiple prescriptions.
    
    Args:
        relationships: List of relationship dicts from Module 2
        
    Returns:
        FHIR Bundle dictionary
    """
    entries = []
    
    for rel in relationships:
        med_request = create_fhir_medication_request(
            drug=rel['drug'],
            dosage=rel['dosage'],
            condition=rel['condition']
        )
        entries.append({
            "resource": med_request,
            "request": {
                "method": "POST",
                "url": "MedicationRequest"
            }
        })
    
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.now().isoformat(),
        "entry": entries
    }
    
    return bundle


def save_fhir_to_file(fhir_data: Dict, filename: str = "medication_request.json") -> str:
    """
    Save FHIR output to a JSON file.
    
    Args:
        fhir_data: FHIR dictionary to save
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    with open(filename, "w") as f:
        json.dump(fhir_data, f, indent=4)
    return filename


def fhir_to_pretty_json(fhir_data: Dict) -> str:
    """
    Convert FHIR data to pretty-printed JSON string.
    
    Args:
        fhir_data: FHIR dictionary
        
    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(fhir_data, indent=2)


if __name__ == "__main__":
    # Demo: Create a FHIR MedicationRequest
    fhir_output = create_fhir_medication_request(
        drug="Amoxicillin",
        dosage="500mg",
        condition="pneumonia"
    )
    
    # Save it
    save_fhir_to_file(fhir_output)
    print("FHIR output saved to medication_request.json!")
    print()
    print(fhir_to_pretty_json(fhir_output))

"""
Shared helpers for tests and scripts.

build_meds mirrors how app.py turns extracted entities into the medication list that
AnomalyDetector consumes (same function: module2_ner.build_medications), so tests exercise
the same logic as the app.
"""

from module2_ner import build_medications, extract_medical_entities, extract_relationships


def build_meds(text, allergies=None, conditions=None):
    """Extract medications from text the same way the app does."""
    entities = extract_medical_entities(text)
    relationships = extract_relationships(entities, text)
    return build_medications(entities, text, relationships)

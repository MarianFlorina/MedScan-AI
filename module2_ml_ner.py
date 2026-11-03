"""
Module 2A: ML-Based Medical NER
================================
Uses d4data/biomedical-ner-all (66M param biomedical BERT) for entity extraction.
Falls back gracefully to regex-based NER if model unavailable.
"""
import warnings
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*You should probably.*")

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from typing import List, Dict, Optional
import re


class ClinicalNERPipeline:
    """ML-powered medical NER using biomedical BERT model."""
    
    # Entity label mapping from model output to our categories
    LABEL_MAP = {
        "Medication": "DRUG",
        "Drug": "DRUG",
        "Drug_identifier": "DRUG",
        "Dosage": "DOSAGE",
        "Strength": "DOSAGE",
        "Duration": "DURATION",
        "Frequency": "FREQUENCY",
        "Route": "ROUTE",
        "Administration": "ROUTE",  # base-model label used for route (IV, IM, oral ...)
        "Form": "FORM",
        "Disease_disorder": "CONDITION",
        "Sign_symptom": "CONDITION",
        "Clinical_event": "CONDITION",
        "Biological_structure": "BODY_PART",
        "Diagnostic_procedure": "PROCEDURE",
        "Lab_value": "LAB_VALUE",
        "Coreference": None,  # skip coreference tokens
    }
    
    # Subword merge patterns (BERT tokenization produces ## fragments)
    SUBWORD_PREFIX = "##"
    
    def __init__(self):
        self._pipeline = None
        self._loaded = False
        self._error = None
    
    def _load_model(self):
    
        from hf_lock import HF_LOCK
    
        with HF_LOCK:
    
            return self._load_model_unlocked()

    
    def _load_model_unlocked(self):
        """Lazy-load the ML model on first use.
        
        Tries fine-tuned LoRA model first, falls back to base model.
        """
        if self._loaded:
            return
        
        try:
            from transformers import pipeline as hf_pipeline, AutoModelForTokenClassification, AutoTokenizer
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
            base_model_name = "d4data/biomedical-ner-all"
            finetuned_dir = "./finetuned_model"
            
            # Try fine-tuned model first
            import os
            if os.path.exists(os.path.join(finetuned_dir, "adapter_config.json")):
                try:
                    from peft import PeftModel
                    
                    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
                    base_model = AutoModelForTokenClassification.from_pretrained(base_model_name)
                    model = PeftModel.from_pretrained(base_model, finetuned_dir)
                    model = model.merge_and_unload()  # Merge LoRA for faster inference
                    
                    self._pipeline = hf_pipeline(
                        "ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple",
                        device=-1,
                    )
                    self._loaded = True
                    self._error = None
                    print("[NER] Loaded fine-tuned prescription model")
                    return
                except Exception as e:
                    print(f"[NER] Fine-tuned model failed ({e}), using base model")
            
            # Fallback: base model
            self._pipeline = hf_pipeline(
                "ner",
                model=base_model_name,
                tokenizer=base_model_name,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
            self._loaded = True
            self._error = None
            
        except Exception as e:
            self._loaded = True  # prevent retry
            self._error = str(e)
            self._pipeline = None
    
    def _merge_subwords(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge BERT subword tokens into complete words.
        
        BERT tokenizes "amoxicillin" as ["am", "##oxicillin"].
        This function merges them back into "amoxicillin".
        """
        if not entities:
            return []
        
        merged = []
        current = None
        
        for ent in entities:
            word = ent["word"]
            label = ent["entity_group"]
            score = ent["score"]
            
            # Check if this is a continuation of the previous token
            if word.startswith(self.SUBWORD_PREFIX):
                if current:
                    # Merge: remove ## prefix and append
                    current["word"] += word[len(self.SUBWORD_PREFIX):]
                    # Update confidence (average)
                    current["score"] = (current["score"] + score) / 2
                    continue
            else:
                # New entity — save previous if exists
                if current:
                    merged.append(current)
                current = {
                    "word": word,
                    "entity_group": label,
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "score": score,
                }
        
        # Don't forget the last entity
        if current:
            merged.append(current)
        
        return merged
    
    def _map_labels(self, entities: List[Dict]) -> List[Dict]:
        """Map model labels to our standard categories."""
        mapped = []
        for ent in entities:
            our_label = self.LABEL_MAP.get(ent["entity_group"])
            if our_label:  # skip None (e.g., Coreference)
                mapped.append({
                    "text": ent["word"].strip(),
                    "type": our_label,
                    "confidence": round(ent["score"] * 100, 1),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                })
        return mapped
    
    def _deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities, keeping highest confidence."""
        # Same text at DIFFERENT places is kept (e.g. "500 mg" for two different drugs);
        # only repeats of the same span (within a few characters) are collapsed.
        kept = []
        for ent in sorted(entities, key=lambda e: -e["confidence"]):
            if any(k["text"].lower() == ent["text"].lower() and k["type"] == ent["type"]
                   and abs(k.get("start", 0) - ent.get("start", 0)) <= 5 for k in kept):
                continue
            kept.append(ent)
        return sorted(kept, key=lambda e: e.get("start", 0))
    
    def _normalize_dosages(self, entities: List[Dict]) -> List[Dict]:
        """
        Normalize dosage entries.
        
        The ML model sometimes produces:
        - "500mg three times daily" (dosage + frequency combined)
        - "650" (number without unit)
        
        This splits them into proper categories.
        """
        normalized = []
        
        for ent in entities:
            text = ent["text"]
            
            if ent["type"] == "DOSAGE":
                # Split combined dosage + frequency
                # e.g., "500mg three times daily" → "500mg" + "three times daily"
                freq_match = re.search(
                    r'(\d+\s*mg(?:/\w+)?|[\d/]+\s*(?:mg|mcg|g|ml|units?))\s+'
                    r'((?:once|twice|three times|four times|every\s+\d+\s+\w+|daily|bid|tid|qid|prn).*)',
                    text, re.IGNORECASE
                )
                if freq_match:
                    normalized.append({
                        "text": freq_match.group(1).strip(),
                        "type": "DOSAGE",
                        "confidence": ent["confidence"],
                        "start": ent.get("start", 0),
                        "end": ent.get("end", 0),
                    })
                    normalized.append({
                        "text": freq_match.group(2).strip(),
                        "type": "FREQUENCY",
                        "confidence": ent["confidence"],
                        "start": ent.get("start", 0) + len(freq_match.group(1)),
                        "end": ent.get("end", 0),
                    })
                    continue
                
                # Fix bare numbers (e.g., "650" → skip, it's already captured as "650mg")
                if re.match(r'^\d+$', text):
                    continue
            
            normalized.append(ent)
        
        return normalized
    
    @staticmethod
    def _chunks(text: str, max_chars: int = 900):
        """Split text on line boundaries into (offset, chunk) pieces of about `max_chars`."""
        if len(text) <= max_chars:
            return [(0, text)]
        chunks, start, pos = [], 0, 0
        for line in text.splitlines(keepends=True):
            if pos - start + len(line) > max_chars and pos > start:
                chunks.append((start, text[start:pos]))
                start = pos
            pos += len(line)
        chunks.append((start, text[start:pos]))
        return [(o, c) for o, c in chunks if c.strip()]

    def extract(self, text: str) -> List[Dict]:
        """
        Extract medical entities using ML model.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            List of entity dicts with text, type, confidence
        """
        self._load_model()
        
        if not self._pipeline:
            return []  # Signal to caller to use regex fallback
        
        try:
            # Run inference. The model reads at most ~512 tokens, so a full page of OCR text would
            # otherwise be silently truncated (e.g. the diagnosis at the bottom never seen).
            raw_results = []
            for offset, chunk in self._chunks(text):
                for r in self._pipeline(chunk):
                    r = dict(r)
                    r["start"] = r.get("start", 0) + offset
                    r["end"] = r.get("end", 0) + offset
                    raw_results.append(r)
            
            # Merge subword tokens
            merged = self._merge_subwords(raw_results)
            
            # Map labels to our categories
            mapped = self._map_labels(merged)
            
            # Normalize dosages and frequencies
            normalized = self._normalize_dosages(mapped)
            
            # Remove duplicates
            deduped = self._deduplicate(normalized)
            
            return deduped
            
        except Exception as e:
            self._error = str(e)
            return []
    
    @property
    def is_available(self) -> bool:
        """Check if ML model is loaded and ready."""
        self._load_model()
        return self._pipeline is not None
    
    @property
    def error(self) -> Optional[str]:
        """Get last error message."""
        return self._error


# Singleton instance
_ml_ner = None

def get_ml_ner() -> ClinicalNERPipeline:
    """Get or create the ML NER pipeline singleton."""
    global _ml_ner
    if _ml_ner is None:
        _ml_ner = ClinicalNERPipeline()
    return _ml_ner


def extract_entities_ml(text: str) -> List[Dict]:
    """
    Extract medical entities using ML model.
    Returns empty list if model unavailable (caller should use regex fallback).
    """
    ner = get_ml_ner()
    return ner.extract(text)

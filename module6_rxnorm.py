"""
Module 6: RxNorm API Integration
=================================
Integrates with the free NLM RxNorm API for live drug validation.
RxNorm provides normalized names for clinical drugs.

API: https://rxnav.nlm.nih.gov/REST/
No API key required for basic usage.
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import time


class RxNormClient:
    """
    Client for the NLM RxNorm REST API.
    
    Provides:
    - Drug name validation
    - Drug interaction checking
    - Ingredient lookup
    - Brand name to generic conversion
    
    Rate limit: 20 requests/second (unauthenticated)
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Args:
            cache_ttl: Cache time-to-live in seconds (default 1 hour)
        """
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make a rate-limited request to the RxNorm API."""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Check cache
        cache_key = endpoint
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.last_request_time = time.time()
                return cached_data
        
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.cache[cache_key] = (time.time(), data)
                self.last_request_time = time.time()
                return data
        except Exception as e:
            return None
    
    def rxcui_by_name(self, drug_name: str) -> Optional[str]:
        """
        Look up the RxCUI (unique drug identifier) by name.
        
        Args:
            drug_name: Drug name to look up
            
        Returns:
            RxCUI string or None
        """
        encoded = urllib.parse.quote(drug_name)
        data = self._make_request(f"rxcui.json?name={encoded}")
        if data and "idGroup" in data:
            rxcuis = data["idGroup"].get("rxnormId", [])
            if rxcuis:
                return rxcuis[0]
        return None
    
    def drug_name_by_rxcui(self, rxcui: str) -> Optional[str]:
        """Get the normalized drug name from an RxCUI."""
        data = self._make_request(f"rxcui/{rxcui}/properties.json")
        if data and "properties" in data:
            return data["properties"].get("name", None)
        return None
    
    def validate_drug(self, drug_name: str) -> Dict:
        """
        Validate a drug name against the RxNorm database.
        
        Args:
            drug_name: Drug name to validate
            
        Returns:
            Dict with validation results:
            - valid: bool
            - rxcui: str or None
            - normalized_name: str or None
            - confidence: float
            - source: str
        """
        rxcui = self.rxcui_by_name(drug_name)
        
        if rxcui:
            normalized = self.drug_name_by_rxcui(rxcui) or drug_name
            return {
                "valid": True,
                "rxcui": rxcui,
                "normalized_name": normalized,
                "confidence": 0.95,
                "source": "rxnorm"
            }
        
        # Try partial match
        encoded = urllib.parse.quote(drug_name)
        data = self._make_request(f"drugs.json?name={encoded}")
        if data and "drugGroup" in data:
            concept_group = data["drugGroup"].get("conceptGroup", [])
            for group in concept_group:
                concepts = group.get("conceptProperties", [])
                if concepts:
                    best = concepts[0]
                    return {
                        "valid": True,
                        "rxcui": best.get("rxcui"),
                        "normalized_name": best.get("name"),
                        "confidence": 0.80,
                        "source": "rxnorm_partial"
                    }
        
        return {
            "valid": False,
            "rxcui": None,
            "normalized_name": None,
            "confidence": 0.0,
            "source": "rxnorm"
        }
    
    def get_drug_ingredients(self, drug_name: str) -> List[str]:
        """Get the active ingredients for a drug."""
        rxcui = self.rxcui_by_name(drug_name)
        if not rxcui:
            return []
        
        data = self._make_request(f"rxcui/{rxcui}/ingredients.json")
        if data and "ingredientGroupGroup" in data:
            ingredients = []
            for group in data["ingredientGroupGroup"].get("ingredient", []):
                ingredients.append(group.get("name", ""))
            return ingredients
        return []
    
    def check_interactions(self, drug_names: List[str]) -> List[Dict]:
        """
        Check for drug-drug interactions using RxNorm.
        
        Note: The full interaction API requires an API key from NLM.
        This uses the free interaction checker endpoint.
        
        Args:
            drug_names: List of drug names
            
        Returns:
            List of interaction findings
        """
        interactions = []
        
        # Get RxCUIs for all drugs
        rxcuis = {}
        for name in drug_names:
            rxcui = self.rxcui_by_name(name)
            if rxcui:
                rxcuis[name] = rxcui
        
        if len(rxcuis) < 2:
            return interactions
        
        # Check pairwise interactions
        rxcui_list = list(rxcuis.values())
        names_list = list(rxcuis.keys())
        
        for i in range(len(rxcui_list)):
            for j in range(i + 1, len(rxcui_list)):
                rxcui_pair = f"{rxcui_list[i]}/{rxcui_list[j]}"
                data = self._make_request(f"interaction/interaction.json?rxcui={rxcui_pair}")
                
                if data and "interactions" in data:
                    for interaction in data["interactions"].get("interactionType", []):
                        for item in interaction.get("interactionPair", []):
                            interactions.append({
                                "drug1": names_list[i],
                                "drug2": names_list[j],
                                "description": item.get("description", "Unknown interaction"),
                                "severity": "MODERATE",  # RxNorm doesn't provide severity
                                "source": "rxnorm"
                            })
        
        return interactions
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """
        Get comprehensive drug information.
        
        Returns:
            Dict with rxcui, name, ingredients, synonyms, etc.
        """
        rxcui = self.rxcui_by_name(drug_name)
        if not rxcui:
            return {"found": False}
        
        # Get properties
        data = self._make_request(f"rxcui/{rxcui}/properties.json")
        props = data.get("properties", {}) if data else {}
        
        # Get synonyms
        syn_data = self._make_request(f"rxcui/{rxcui}/synonyms.json")
        synonyms = []
        if syn_data and "synonymGroup" in syn_data:
            for syn in syn_data["synonymGroup"].get("synonym", []):
                synonyms.append(syn.get("term", ""))
        
        return {
            "found": True,
            "rxcui": rxcui,
            "name": props.get("name", drug_name),
            "generic_name": props.get("name", drug_name),
            "ndc": props.get("ndcTermList", {}).get("ndcTerm", []),
            "synonyms": synonyms[:10],  # Limit to 10
        }


# Singleton instance
_rxnorm_client = None

def get_rxnorm_client() -> RxNormClient:
    """Get or create the RxNorm client singleton."""
    global _rxnorm_client
    if _rxnorm_client is None:
        _rxnorm_client = RxNormClient()
    return _rxnorm_client


def validate_drug_online(drug_name: str) -> Dict:
    """Quick drug validation using RxNorm API."""
    client = get_rxnorm_client()
    return client.validate_drug(drug_name)


def check_interactions_online(drug_names: List[str]) -> List[Dict]:
    """Quick interaction check using RxNorm API."""
    client = get_rxnorm_client()
    return client.check_interactions(drug_names)


if __name__ == "__main__":
    # Demo
    client = RxNormClient()
    
    test_drugs = ["amoxicillin", "ibuprofen", "warfarin", "asdfghjkl"]
    
    print("Drug Validation:")
    for drug in test_drugs:
        result = client.validate_drug(drug)
        status = "VALID" if result["valid"] else "NOT FOUND"
        print(f"  {drug}: {status} (confidence: {result['confidence']:.0%})")
        if result["normalized_name"]:
            print(f"    Normalized: {result['normalized_name']}")
        if result["rxcui"]:
            print(f"    RxCUI: {result['rxcui']}")
    
    print("\nInteraction Check:")
    interactions = client.check_interactions(["warfarin", "aspirin", "ibuprofen"])
    for interaction in interactions:
        print(f"  {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")

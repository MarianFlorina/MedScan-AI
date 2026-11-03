"""
Module 9: Medical Q&A Assistant (v2)
=====================================
Understands patient and clinician questions about prescriptions and provides
clear, natural-language answers. Handles both structured and free-form queries.

Design principles:
- Understand intent, not just keywords
- Answer exactly what was asked, nothing more
- Be concise and clinically accurate
- Always ground answers in the provided text
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from typing import List, Dict, Optional
import re


# ============================================================
# Intent Classification (rule-based, fast)
# ============================================================

INTENT_PATTERNS = [
    # Medication listing
    {
        "intent": "list_medications",
        "patterns": [
            r"what\s+(?:drugs?|medications?|medicines?|pills?|tablets?|capsules?)\s+(?:are|is|do|did)\s+(?:.*(?:prescribed|given|listed|taken|for|on))",
            r"(?:which|what)\s+(?:medication|drug|medicine)\s+(?:.*(?:prescribed|given|taking))",
            r"(?:list|name)\s+(?:all\s+)?(?:the\s+)?(?:drugs?|medications?|medicines?)",
            r"patient\s+(?:is\s+)?(?:taking|on|prescribed)\s+(?:what|which)",
            r"(?:what|which)\s+(?:meds?|drugs?)",
            r"(?:show|tell)\s+(?:me\s+)?(?:the\s+)?(?:drugs?|medications?|medicines?)",
            r"(?:how\s+many|what)\s+(?:drugs?|medications?)",
        ],
    },
    # Dosage questions
    {
        "intent": "list_dosages",
        "patterns": [
            r"(?:what|how\s+much)\s+(?:is|are|the)\s+(?:the\s+)?(?:dos(?:age|es)|dose|strength|amount|quantity)",
            r"(?:how\s+(?:much|often|many))\s+(?:should|can|does|is)\s+(?:.*(?:take|take|given|prescribed))",
            r"(?:what\s+)?(?:dose|dosage)\s+(?:for|of|is|are)",
            r"(?:how\s+(?:many|much))\s+(?:mg|mcg|ml|tablets?|capsules?)",
            r"(?:what\s+)?(?:strength|form)\s+(?:of|is|are)",
        ],
    },
    # Condition questions
    {
        "intent": "list_conditions",
        "patterns": [
            r"(?:what|which)\s+(?:conditions?|diseases?|diagnoses?|illnesses?|problems?)\s+(?:are|is|does|did|has|have|was|were)",
            r"(?:what|which)\s+(?:is|are)\s+(?:.*(?:diagnosed|suffering|dealing|treating|treating|management))",
            r"(?:patient|they)\s+(?:has|have|had|is\s+suffering|is\s+diagnosed)\s+(?:with\s+)?(?:what|which)",
            r"(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:patient|their)\s+(?:condition|diagnosis|disease)",
            r"(?:why|what\s+for)\s+(?:are|is)\s+(?:.*(?:prescribed|given|taking))",
        ],
    },
    # Interaction questions
    {
        "intent": "check_interactions",
        "patterns": [
            r"(?:are|is)\s+there\s+(?:any\s+)?(?:drug\s+)?interactions?",
            r"(?:can|should)\s+(?:.*(?:take|use|combine|mix|together))",
            r"(?:is\s+it\s+)?safe\s+(?:to\s+)?(?:take|use|combine|mix)\s+(?:these|them|together)",
            r"(?:any\s+)?(?:conflicts?|problems?|issues?|concerns?)\s+(?:between|with|among)",
            r"(?:do|are)\s+(?:these|they|the\s+drugs?)\s+(?:interact|conflict|clash|mix)",
            r"(?:will|can)\s+(?:these|they)\s+(?:.*(?:harm|hurt|affect|cause))",
        ],
    },
    # Allergy questions
    {
        "intent": "check_allergies",
        "patterns": [
            r"(?:are|is)\s+there\s+(?:any\s+)?allerg(?:y|ies)",
            r"(?:any|what)\s+(?:allerg(?:y|ies)|reactions?|sensitivities?)",
            r"(?:patient|they)\s+(?:is|are|has)\s+(?:allerg(?:y|ic|ies))",
            r"(?:is\s+this\s+)?(?:safe|okay)\s+(?:for|with)\s+(?:.*allerg)",
            r"(?:what\s+)?(?:drugs?\s+)?(?:should|must|cannot|can't)\s+(?:.*(?:avoid|not\s+take|allergy))",
        ],
    },
    # Safety questions
    {
        "intent": "safety_overview",
        "patterns": [
            r"is\s+this\s+(?:prescription|medication|drug|regimen|treatment)\s+safe",
            r"(?:is|are)\s+(?:the|this|these)\s+(?:drugs?|medications?|prescriptions?)\s+(?:safe|okay|fine|correct|right)",
            r"(?:any|what)\s+(?:risks?|dangers?|warnings?|concerns?|problems?|issues?|red\s+flags?)",
            r"(?:is|are)\s+there\s+(?:any\s+)?(?:risks?|dangers?|warnings?|concerns?|problems?)",
            r"(?:should|can)\s+(?:i|we|the\s+patient)\s+(?:be\s+)?(?:worried|concerned|afraid|scared)",
            r"(?:is|does)\s+(?:this|it)\s+(?:look|seem)\s+(?:safe|okay|fine|correct|right|normal)",
        ],
    },
    # Instructions
    {
        "intent": "patient_instructions",
        "patterns": [
            r"(?:what|how)\s+(?:should|must|do|does)\s+(?:the\s+)?patient\s+(?:take|use|avoid|eat|drink|not)",
            r"(?:how|when|where)\s+(?:should|must|do)\s+(?:.*(?:take|use|consume|drink|eat))",
            r"(?:what\s+)?(?:should|must)\s+(?:the\s+)?patient\s+(?:avoid|not\s+eat|not\s+drink|not\s+take)",
            r"(?:any|what)\s+(?:food|diet|lifestyle|precautions?|instructions?|directions?)",
            r"(?:how\s+)?(?:to\s+)?(?:take|use|consume)\s+(?:this|it|the\s+medication|the\s+drug)",
            r"(?:any|what)\s+(?:restrictions?|limitations?|guidelines?)",
        ],
    },
    # Monitoring
    {
        "intent": "monitoring_recommendations",
        "patterns": [
            r"(?:what|which)\s+(?:tests?|labs?|monitoring|checks?|screenings?)\s+(?:are|is|do|does|should|need)",
            r"(?:any|what)\s+(?:follow[- ]?up|lab|blood|test|monitoring)\s+(?:needed|required|recommended|necessary)",
            r"(?:when|how\s+often)\s+(?:should|must|do)\s+(?:.*(?:test|check|monitor|lab|blood))",
            r"(?:what|which)\s+(?:vital\s+signs?|parameters?)\s+(?:to|should|need)\s+(?:monitor|check|track)",
        ],
    },
    # Treatment purpose
    {
        "intent": "treatment_purpose",
        "patterns": [
            r"(?:why|what\s+for|what\s+is)\s+(?:.*(?:prescribed|given|taking|taking))",
            r"(?:what|which)\s+(?:condition|disease|problem|illness)\s+(?:is|are)\s+(?:this|it|these)\s+(?:treating|for|prescribed)",
            r"(?:what)\s+(?:does|is|are)\s+(?:this|it|these)\s+(?:medication|drug|treatment)\s+(?:do|for|treat)",
            r"(?:why|how)\s+(?:does|is|are)\s+(?:this|it|these)\s+(?:help|work|treat)",
        ],
    },
    # Dosage safety
    {
        "intent": "dosage_safety",
        "patterns": [
            r"(?:is|are)\s+(?:the\s+)?(?:dos(?:age|es)|dose|amount|strength)\s+(?:safe|correct|right|appropriate|too\s+(?:high|low|much))",
            r"(?:is|are)\s+(?:this|it|the\s+dose)\s+(?:over|above|exceeding|too\s+high|too\s+low|under)",
            r"(?:what\s+is\s+)?(?:the\s+)?(?:max|maximum|safe|correct|usual|standard)\s+(?:dose|dosage)",
            r"(?:can|is)\s+(?:this|it)\s+(?:cause|lead\s+to|result\s+in)\s+(?:overdose|toxicity|side\s+effects?)",
        ],
    },
    # Indian drug questions
    {
        "intent": "indian_drug_info",
        "patterns": [
            r"(?:what\s+is|what\s+does)\s+(?:crocin|combiflam|dolo|omez|asthalin|glycomet|losar|azee|taxim|monocef|betnovate)",
            r"(?:what\s+is)\s+(?:the\s+)?(?:generic|real|actual)\s+(?:name|drug)\s+(?:of|for)",
            r"(?:is|are)\s+(?:crocin|combiflam|dolo|omez)\s+(?:.*(?:same|similar|generic|real))",
        ],
    },
]


def classify_intent(question: str) -> str:
    """Classify the user's question into a clinical intent."""
    q = question.lower().strip()
    
    # Score each intent
    scores = {}
    for intent_group in INTENT_PATTERNS:
        score = 0
        for pattern in intent_group["patterns"]:
            if re.search(pattern, q):
                score += 1
        if score > 0:
            scores[intent_group["intent"]] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    # Default: if question is about the prescription at all, try safety overview
    if any(w in q for w in ["prescription", "medication", "drug", "patient", "take", "this"]):
        return "safety_overview"
    
    return "general"


# ============================================================
# Answer Generators (one per intent)
# ============================================================

def _generate_medication_answer(text, medications, entities, **kwargs):
    if not medications:
        return "No medications were identified in the prescription."
    
    lines = []
    for m in medications:
        drug = m.get("drug", "Unknown")
        dosage = m.get("dosage", "")
        if dosage:
            lines.append(f"**{drug}** — {dosage}")
        else:
            lines.append(f"**{drug}**")
    
    return f"The prescription includes **{len(medications)} medication(s)**:\n\n" + "\n".join(lines)


def _generate_dosage_answer(text, medications, entities, **kwargs):
    if not medications:
        return "No dosage information was identified."
    
    lines = []
    for m in medications:
        drug = m.get("drug", "Unknown")
        dosage = m.get("dosage", "not specified")
        lines.append(f"**{drug}**: {dosage}")
    
    return "**Dosage Information:**\n\n" + "\n".join(lines)


def _generate_condition_answer(text, medications, entities, **kwargs):
    conditions = [e["word"] for e in (entities or []) if e.get("entity_group") == "CONDITION"]
    if not conditions:
        return "No specific medical conditions were mentioned in this prescription."
    
    unique_conditions = list(dict.fromkeys(conditions))  # preserve order, dedupe
    lines = [f"**{c}**" for c in unique_conditions]
    
    return f"The patient has **{len(unique_conditions)} condition(s)** noted:\n\n" + "\n".join(lines)


def _generate_interaction_answer(text, medications, entities, anomalies=None, **kwargs):
    if anomalies and anomalies.get("findings"):
        interaction_findings = [f for f in anomalies["findings"] if "INTERACTION" in f.get("type", "")]
        if interaction_findings:
            lines = []
            for f in interaction_findings:
                sev = f.get("severity", "MODERATE")
                icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "ℹ️"}.get(sev, "•")
                lines.append(f"{icon} **{sev}**: {f['description']}")
            return "**Drug Interactions Detected:**\n\n" + "\n".join(lines)
    
    return "No significant drug interactions were detected among the prescribed medications."


def _generate_allergy_answer(text, medications, entities, allergies=None, anomalies=None, **kwargs):
    if allergies:
        allergy_text = ", ".join(allergies)
        if anomalies and anomalies.get("findings"):
            allergy_findings = [f for f in anomalies["findings"] if "ALLERGY" in f.get("type", "")]
            if allergy_findings:
                lines = [f"🚨 **CONFLICT**: {f['description']}" for f in allergy_findings]
                return f"**Known Allergies**: {allergy_text}\n\n" + "\n".join(lines)
        return f"**Known Allergies**: {allergy_text}. No conflicts detected with the current prescriptions."
    
    return "No allergy information was provided. Please specify any known drug allergies for a complete safety assessment."


def _generate_safety_answer(text, medications, entities, anomalies=None, **kwargs):
    total = anomalies["total_anomalies"] if anomalies else 0
    critical = anomalies["severity_counts"]["CRITICAL"] if anomalies else 0
    high = anomalies["severity_counts"]["HIGH"] if anomalies else 0
    moderate = anomalies["severity_counts"]["MODERATE"] if anomalies else 0
    
    if critical > 0:
        return f"🚨 **Safety Alert**: {critical} critical issue(s) and {high} high-severity concern(s) were detected. **Immediate clinical review is recommended before dispensing this prescription.**"
    elif high > 0:
        return f"⚠️ **Caution**: {high} high-severity concern(s) were identified. A clinical review is recommended before finalizing."
    elif moderate > 0:
        return f"ℹ️ {moderate} moderate concern(s) were noted. The prescription is generally acceptable but should be reviewed."
    elif total > 0:
        return f"💡 {total} minor note(s) were found. The prescription appears generally safe."
    else:
        return "✅ The prescription appears safe based on the current analysis. No concerns were detected."


def _generate_instruction_answer(text, medications, entities, **kwargs):
    instructions = []
    for m in medications:
        drug = m.get("drug", "").lower()
        drug_display = m.get("drug", "Unknown")
        
        if "warfarin" in drug:
            instructions.append(f"**{drug_display}**: Avoid vitamin K-rich foods (leafy greens). Report any unusual bleeding or bruising.")
        elif "metformin" in drug:
            instructions.append(f"**{drug_display}**: Take with meals to reduce stomach upset. Limit alcohol consumption.")
        elif "lisinopril" in drug or "enalapril" in drug or "ramipril" in drug:
            instructions.append(f"**{drug_display}**: Report any persistent dry cough or swelling of face/lips.")
        elif "atorvastatin" in drug or "simvastatin" in drug or "rosuvastatin" in drug:
            instructions.append(f"**{drug_display}**: Report unexplained muscle pain or weakness. Avoid grapefruit juice.")
        elif "amoxicillin" in drug or "azithromycin" in drug or "ciprofloxacin" in drug:
            instructions.append(f"**{drug_display}**: Complete the full course even if you feel better. Do not skip doses.")
        elif "omeprazole" in drug or "pantoprazole" in drug:
            instructions.append(f"**{drug_display}**: Take 30 minutes before meals. Avoid long-term use without medical review.")
        elif "aspirin" in drug:
            instructions.append(f"**{drug_display}**: Take with food. Report any signs of bleeding (black stools, unusual bruising).")
        elif "ibuprofen" in drug or "naproxen" in drug or "diclofenac" in drug:
            instructions.append(f"**{drug_display}**: Take with food. Avoid if you have a history of stomach ulcers or kidney disease.")
        elif "sertraline" in drug or "escitalopram" in drug or "fluoxetine" in drug:
            instructions.append(f"**{drug_display}**: Do not stop suddenly — taper under medical guidance. Avoid alcohol.")
        else:
            instructions.append(f"**{drug_display}**: Take as directed by your prescriber.")
    
    if instructions:
        return "**Patient Instructions:**\n\n" + "\n".join(instructions)
    return "Follow your healthcare provider's instructions for all medications."


def _generate_monitoring_answer(text, medications, entities, **kwargs):
    recs = []
    drug_names = [m.get("drug", "").lower() for m in medications]
    
    monitoring_map = {
        "warfarin": "INR blood test every 2–4 weeks",
        "lithium": "Lithium blood levels every 3–6 months; kidney and thyroid function annually",
        "methotrexate": "CBC, liver function tests, creatinine every 4–8 weeks",
        "metformin": "Kidney function annually; HbA1c every 3–6 months",
        "clozapine": "Absolute neutrophil count (ANC) weekly for 6 months, then biweekly",
        "atorvastatin": "Liver function tests at baseline; CK if muscle pain develops",
        "amiodarone": "Thyroid function, liver function, lung function tests every 6 months",
        "ace_inhibitor": "Potassium and creatinine 1–2 weeks after starting",
        "digoxin": "Digoxin blood levels and potassium levels",
        "carbamazepine": "Drug blood levels, CBC, liver function periodically",
        "levothyroxine": "TSH every 6–8 weeks until stable, then annually",
    }
    
    for i, drug in enumerate(drug_names):
        for key, rec in monitoring_map.items():
            if key in drug:
                med_name = medications[i].get("drug", drug) if medications and i < len(medications) else drug
                recs.append(f"• **{med_name}**: {rec}")
                break
    
    if recs:
        return "**Recommended Monitoring:**\n\n" + "\n".join(recs)
    return "No specific monitoring requirements were identified for the current medications."


def _generate_purpose_answer(text, medications, entities, **kwargs):
    conditions = [e["word"] for e in (entities or []) if e.get("entity_group") == "CONDITION"]
    if not conditions:
        if medications:
            drug_names = ", ".join(m.get("drug", "Unknown") for m in medications)
            return f"The prescription includes {drug_names}. The specific conditions being treated are not explicitly mentioned in the text."
        return "The treatment purpose is not clearly stated in the prescription text."
    
    unique_conditions = list(dict.fromkeys(conditions))
    if medications:
        drug_names = ", ".join(m.get("drug", "Unknown") for m in medications)
        return f"**{drug_names}** is prescribed to treat **{', '.join(unique_conditions)}**."
    return f"The conditions being treated include: **{', '.join(unique_conditions)}**."


def _generate_dosage_safety_answer(text, medications, entities, anomalies=None, **kwargs):
    if anomalies and anomalies.get("findings"):
        dosage_findings = [f for f in anomalies["findings"] if "DOSAGE" in f.get("type", "")]
        if dosage_findings:
            lines = []
            for f in dosage_findings:
                lines.append(f"• **{f.get('drug', 'Unknown')}**: {f['description']}")
            return "**Dosage Assessment:**\n\n" + "\n".join(lines)
    return "All dosages appear to be within recommended ranges."


def _generate_indian_drug_answer(text, medications, entities, **kwargs):
    from module11_indian_rx import INDIAN_DRUG_BRANDS
    
    results = []
    text_lower = text.lower()
    for brand, generic in INDIAN_DRUG_BRANDS.items():
        if generic is None:
            continue
        if brand in text_lower:
            results.append(f"• **{brand.title()}** is the brand name for **{generic}** (generic)")
    
    if results:
        return "**Indian Drug Brand Information:**\n\n" + "\n".join(results)
    return "No Indian drug brand names were detected in the text."


# Intent → handler mapping
INTENT_HANDLERS = {
    "list_medications": _generate_medication_answer,
    "list_dosages": _generate_dosage_answer,
    "list_conditions": _generate_condition_answer,
    "check_interactions": _generate_interaction_answer,
    "check_allergies": _generate_allergy_answer,
    "safety_overview": _generate_safety_answer,
    "patient_instructions": _generate_instruction_answer,
    "monitoring_recommendations": _generate_monitoring_answer,
    "treatment_purpose": _generate_purpose_answer,
    "dosage_safety": _generate_dosage_safety_answer,
    "indian_drug_info": _generate_indian_drug_answer,
}


# ============================================================
# ML Text-Generation Backend (lazy-loaded, enhances answers)
# ============================================================

_ml_generator = None
_ml_available = None
_ml_model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

MEDICAL_SYSTEM_PROMPT = (
    "You are a clinical pharmacy assistant. Answer questions about prescriptions "
    "using ONLY the provided clinical context. Be concise, accurate, and professional. "
    "Never fabricate medical information. Always recommend consulting a healthcare professional."
)


def _get_ml_generator():
    from hf_lock import HF_LOCK
    with HF_LOCK:
        return _get_ml_generator_unlocked()


def _get_ml_generator_unlocked():
    global _ml_generator, _ml_available, _ml_model_name
    if _ml_available is False:
        return None
    if _ml_generator is not None:
        return _ml_generator
    
    try:
        from transformers import pipeline as hf_pipeline
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        _ml_generator = hf_pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM-135M-Instruct",
            device=-1,
        )
        _ml_available = True
        return _ml_generator
    except Exception as e:
        _ml_available = False
        print(f"[MedQA] ML generator failed ({e}). Using rule-based only.")
        return None


def _enhance_with_ml(question, clinical_text, rule_answer, medications=None):
    """Use ML to rephrase a rule-based answer in natural language."""
    generator = _get_ml_generator()
    if generator is None:
        return rule_answer
    
    try:
        med_info = ""
        if medications:
            med_list = [f"{m.get('drug', '')} ({m.get('dosage', '')})" for m in medications if m.get("drug")]
            if med_list:
                med_info = f"\nMedications: {', '.join(med_list)}"
        
        prompt = (
            f"Clinical text: {clinical_text[:1500]}\n"
            f"{med_info}\n\n"
            f"Question: {question}\n\n"
            f"Answer (rewrite this clearly and concisely):\n{rule_answer}\n\n"
            f"Clear answer:"
        )
        
        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        result = generator(messages, max_new_tokens=200, do_sample=False, temperature=0.1)
        generated = result[0]["generated_text"]
        
        if isinstance(generated, list):
            response = generated[-1].get("content", "")
        elif isinstance(generated, str):
            response = generated.split("Clear answer:")[-1].strip()
        else:
            response = str(generated)
        
        response = response.strip()
        if len(response) > 30:
            return response
        return rule_answer
    except Exception:
        return rule_answer


# ============================================================
# Public API
# ============================================================

def _is_faithful(candidate: str, rule_answer: str, medications) -> bool:
    """Accept an LLM rewrite only if it is safe to show in a clinical context.

    A 135M-parameter model can loop or drop facts, so the rewrite must (1) keep every
    medication name from the grounded answer, (2) not be repetitive, (3) not balloon in size.
    """
    if not candidate or len(candidate) > 3 * max(len(rule_answer), 200):
        return False
    low = candidate.lower()
    for m in medications or []:
        name = (m.get("drug") or "").lower()
        if name and name in rule_answer.lower() and name not in low:
            return False
    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if lines and len(set(lines)) / len(lines) < 0.8:
        return False
    return True


def answer_clinical_question(
    question,
    text,
    medications=None,
    entities=None,
    allergies=None,
    anomalies=None,
    use_ml=True,
):
    """
    Answer a clinical question about a prescription.
    
    Understands the question intent, gathers relevant context,
    and returns a clear, concise answer.
    """
    intent = classify_intent(question)
    
    if intent == "general":
        # Try safety overview as default for unrecognized questions
        intent = "safety_overview"
    
    handler = INTENT_HANDLERS.get(intent)
    if handler is None:
        return {
            "answer": "I can help you with questions about medications, dosages, drug interactions, allergies, safety, patient instructions, and monitoring. Please try rephrasing your question.",
            "method": "fallback",
            "confidence": 0.5,
        }
    
    rule_answer = handler(
        text=text,
        medications=medications or [],
        entities=entities or [],
        allergies=allergies or [],
        anomalies=anomalies,
    )
    
    if use_ml:
        enhanced = _enhance_with_ml(question, text, rule_answer, medications)
        if enhanced != rule_answer and _is_faithful(enhanced, rule_answer, medications):
            return {
                "answer": enhanced,
                "method": "ml_enhanced",
                "confidence": 0.88,
                "intent": intent,
            }
    
    return {
        "answer": rule_answer,
        "method": "clinical_engine",
        "confidence": 0.92,
        "intent": intent,
    }


def get_model_status():
    generator = _get_ml_generator()
    return {
        "available": generator is not None,
        "model": _ml_model_name if generator else None,
        "rule_based_questions": len(INTENT_PATTERNS),
    }

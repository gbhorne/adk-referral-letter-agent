import json
import vertexai
from vertexai.generative_models import GenerativeModel
from shared.config import GCP_PROJECT, LOCATION
from shared.models import ReferralContext, UrgencyClassification, UrgencyLevel

vertexai.init(project=GCP_PROJECT, location=LOCATION)


def classify_urgency(context: ReferralContext) -> UrgencyClassification:
    """
    RL-2: Classify referral urgency using Gemini.
    Returns a structured UrgencyClassification with level, rationale, and confidence.
    ROUTINE = routine appointment
    URGENT = within 48 hours
    EMERGENT = same day or ED transfer required
    """
    model = GenerativeModel("gemini-2.5-flash")

    conditions_text = ""
    for c in context.conditions[:10]:
        code_text = ""
        for coding in c.get("code", {}).get("coding", []):
            if coding.get("display"):
                code_text = coding["display"]
                break
        if not code_text:
            code_text = c.get("code", {}).get("text", "Unknown condition")
        conditions_text += f"- {code_text}\n"

    observations_text = ""
    for o in context.observations[:10]:
        code_text = ""
        for coding in o.get("code", {}).get("coding", []):
            if coding.get("display"):
                code_text = coding["display"]
                break
        value = o.get("valueQuantity", {})
        value_str = f"{value.get('value', '')} {value.get('unit', '')}".strip()
        if code_text:
            observations_text += f"- {code_text}: {value_str}\n"

    prompt = f"""You are a clinical triage specialist. Classify the urgency of the following specialist referral.

REFERRAL REASON: {context.referral_reason}
SPECIALTY REQUESTED: {context.specialty}

ACTIVE CONDITIONS:
{conditions_text if conditions_text else "None documented"}

RECENT OBSERVATIONS:
{observations_text if observations_text else "None documented"}

Classify the urgency as exactly one of:
- ROUTINE: Non-urgent, standard appointment scheduling
- URGENT: Requires appointment within 48 hours due to clinical risk
- EMERGENT: Requires same-day assessment or emergency department transfer

Respond with valid JSON only. No preamble, no explanation, no markdown.

{{
  "urgency": "ROUTINE" | "URGENT" | "EMERGENT",
  "clinical_rationale": "One to two sentence clinical justification for this classification",
  "confidence": 0.0 to 1.0
}}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    data = json.loads(raw)

    return UrgencyClassification(
        urgency=UrgencyLevel(data["urgency"]),
        clinical_rationale=data["clinical_rationale"],
        confidence=float(data["confidence"]),
    )

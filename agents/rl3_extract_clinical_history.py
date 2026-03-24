import vertexai
from vertexai.generative_models import GenerativeModel
from shared.config import GCP_PROJECT, LOCATION
from shared.models import ReferralContext

vertexai.init(project=GCP_PROJECT, location=LOCATION)


def extract_clinical_history(context: ReferralContext) -> str:
    """
    RL-3: Extract clinical history scoped to the referral reason.
    Excludes unrelated chronic conditions unless they affect the referral.
    Returns a structured narrative string for inclusion in the referral letter.
    """
    model = GenerativeModel("gemini-2.5-flash")

    conditions_text = ""
    for c in context.conditions[:20]:
        code_text = ""
        for coding in c.get("code", {}).get("coding", []):
            if coding.get("display"):
                code_text = coding["display"]
                break
        if not code_text:
            code_text = c.get("code", {}).get("text", "Unknown condition")
        clinical_status = c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown")
        conditions_text += f"- {code_text} (status: {clinical_status})\n"

    observations_text = ""
    for o in context.observations[:20]:
        code_text = ""
        for coding in o.get("code", {}).get("coding", []):
            if coding.get("display"):
                code_text = coding["display"]
                break
        value = o.get("valueQuantity", {})
        value_str = f"{value.get('value', '')} {value.get('unit', '')}".strip()
        effective = o.get("effectiveDateTime", "")[:10] if o.get("effectiveDateTime") else ""
        if code_text:
            observations_text += f"- {code_text}: {value_str} ({effective})\n"

    medications_text = ""
    for m in context.medication_requests[:20]:
        med_name = ""
        med_ref = m.get("medicationCodeableConcept", {})
        for coding in med_ref.get("coding", []):
            if coding.get("display"):
                med_name = coding["display"]
                break
        if not med_name:
            med_name = med_ref.get("text", "Unknown medication")
        status = m.get("status", "unknown")
        medications_text += f"- {med_name} (status: {status})\n"

    prompt = f"""You are an experienced clinician preparing a specialist referral letter.

REFERRAL REASON: {context.referral_reason}
SPECIALTY REQUESTED: {context.specialty}

Extract and summarize the clinical history relevant to this referral from the data below.

RULES:
- Include only conditions, observations, and medications directly relevant to the referral reason
- Exclude unrelated chronic conditions UNLESS they affect the referral or specialist management
- Organize into three short paragraphs: Relevant History, Relevant Investigations, Current Relevant Medications
- Write in clinical prose suitable for a specialist referral letter
- Be concise -- this section should not exceed 200 words total
- Do not include patient name or identifiers

ALL CONDITIONS ON RECORD:
{conditions_text if conditions_text else "None documented"}

ALL RECENT OBSERVATIONS:
{observations_text if observations_text else "None documented"}

ALL CURRENT MEDICATIONS:
{medications_text if medications_text else "None documented"}

Write the scoped clinical history now:"""

    response = model.generate_content(prompt)
    return response.text.strip()

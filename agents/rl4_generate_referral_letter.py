import vertexai
from vertexai.generative_models import GenerativeModel
from shared.config import GCP_PROJECT, LOCATION
from shared.models import ReferralContext, UrgencyClassification, UrgencyLevel

vertexai.init(project=GCP_PROJECT, location=LOCATION)

URGENCY_HEADERS = {
    UrgencyLevel.ROUTINE: "ROUTINE REFERRAL",
    UrgencyLevel.URGENT: "URGENT REFERRAL -- APPOINTMENT REQUIRED WITHIN 48 HOURS",
    UrgencyLevel.EMERGENT: "EMERGENT REFERRAL -- SAME-DAY ASSESSMENT REQUIRED",
}

URGENCY_CLOSING = {
    UrgencyLevel.ROUTINE: "Please schedule this patient at your earliest routine availability.",
    UrgencyLevel.URGENT: "This referral is classified URGENT. Please arrange assessment within 48 hours.",
    UrgencyLevel.EMERGENT: "This referral is classified EMERGENT. Same-day assessment or emergency department transfer is required. Please contact the referring team immediately upon receipt.",
}


def generate_referral_letter(
    context: ReferralContext,
    urgency: UrgencyClassification,
    history: str,
) -> str:
    """
    RL-4: Generate a LOINC 57133-1 compliant specialist referral letter.
    Sections: reason for referral, relevant history, investigations,
    current medications, clinical question, urgency statement.
    """
    model = GenerativeModel("gemini-2.0-flash-001")

    performer_line = ""
    if context.performer and context.performer.name:
        name = context.performer.name.strip()
        if name.lower().startswith("dr.") or name.lower().startswith("dr "):
            performer_line = f"Dear {name},"
        else:
            performer_line = f"Dear Dr. {name},"
    elif context.performer and context.performer.specialty:
        performer_line = f"Dear {context.performer.specialty} Specialist,"
    else:
        performer_line = f"Dear {context.specialty} Specialist,"

    medications_text = ""
    for m in context.medication_requests[:10]:
        med_name = ""
        med_ref = m.get("medicationCodeableConcept", {})
        for coding in med_ref.get("coding", []):
            if coding.get("display"):
                med_name = coding["display"]
                break
        if not med_name:
            med_name = med_ref.get("text", "Unknown medication")
        status = m.get("status", "unknown")
        if status == "active":
            medications_text += f"- {med_name}\n"

    urgency_header = URGENCY_HEADERS[urgency.urgency]
    urgency_closing = URGENCY_CLOSING[urgency.urgency]

    prompt = f"""You are a clinician generating a formal specialist referral letter.
Generate a complete LOINC 57133-1 compliant referral letter using the information below.

URGENCY CLASSIFICATION: {urgency.urgency.value}
URGENCY RATIONALE: {urgency.clinical_rationale}
REFERRAL REASON: {context.referral_reason}
SPECIALTY: {context.specialty}
SALUTATION: {performer_line}

SCOPED CLINICAL HISTORY (already filtered to referral-relevant content):
{history}

CURRENT ACTIVE MEDICATIONS:
{medications_text if medications_text else "None documented"}

Generate the letter with these exact sections in order:
1. Letter header with urgency classification: {urgency_header}
2. Salutation: {performer_line}
3. REASON FOR REFERRAL: One paragraph stating the referral reason clearly
4. RELEVANT CLINICAL HISTORY: Use the scoped history provided above -- do not add information not present
5. RELEVANT INVESTIGATIONS: Summarize key investigation findings from the history
6. CURRENT MEDICATIONS: List active medications
7. CLINICAL QUESTION: One to two specific questions for the specialist to address
8. URGENCY STATEMENT: {urgency_closing}
9. Closing with "Yours sincerely," followed by a blank line for the referring clinician signature

Write in formal clinical letter style. Do not include patient name or date of birth in the letter body.
Do not add any preamble before the letter header."""

    response = model.generate_content(prompt)
    return response.text.strip()

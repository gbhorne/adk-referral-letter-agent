import json
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from agents.rl1_load_referral_context import load_referral_context
from agents.rl2_classify_urgency import classify_urgency
from agents.rl3_extract_clinical_history import extract_clinical_history
from agents.rl4_generate_referral_letter import generate_referral_letter
from agents.rl5_assemble_document import assemble_document
from agents.rl6_route_and_notify import route_and_notify
from shared.models import ReferralContext, UrgencyClassification


def run_referral_pipeline(service_request_id: str) -> str:
    """
    Executes the full referral letter agent pipeline (RL-1 through RL-6).
    Accepts a FHIR ServiceRequest ID and returns the created DocumentReference ID.
    """
    # RL-1: Load clinical context
    context: ReferralContext = load_referral_context(service_request_id)

    # RL-2: Classify urgency
    urgency: UrgencyClassification = classify_urgency(context)

    # RL-3: Extract scoped clinical history
    history: str = extract_clinical_history(context)

    # RL-4: Generate referral letter
    letter_text: str = generate_referral_letter(context, urgency, history)

    # RL-5: Assemble FHIR DocumentReference
    document: dict = assemble_document(context, urgency, letter_text)

    # RL-6: Write to FHIR, route via Pub/Sub, log to Firestore
    document_id: str = route_and_notify(document, urgency, context)

    return json.dumps({
        "status": "success",
        "document_reference_id": document_id,
        "service_request_id": service_request_id,
        "urgency": urgency.urgency.value,
        "urgency_rationale": urgency.clinical_rationale,
        "confidence": urgency.confidence,
        "specialty": context.specialty,
        "referral_reason": context.referral_reason,
    }, indent=2)


root_agent = Agent(
    name="referral_letter_agent",
    model="gemini-2.5-flash",
    description=(
        "Generates LOINC 57133-1 compliant specialist referral letters from FHIR R4 resources. "
        "Classifies urgency as ROUTINE, URGENT, or EMERGENT and routes via Pub/Sub accordingly."
    ),
    instruction="""You are the Referral Letter Agent. Your job is to generate a complete specialist referral letter from a FHIR ServiceRequest.

When given a ServiceRequest ID, call run_referral_pipeline with that ID.

The pipeline will:
1. Load the clinical context from FHIR (RL-1)
2. Classify referral urgency as ROUTINE, URGENT, or EMERGENT (RL-2)
3. Extract clinical history scoped to the referral reason (RL-3)
4. Generate a LOINC 57133-1 compliant referral letter (RL-4)
5. Assemble a FHIR DocumentReference with urgency metadata (RL-5)
6. Write to FHIR store, route via Pub/Sub, and log audit to Firestore (RL-6)

Report the DocumentReference ID and urgency classification to the user when complete.
If EMERGENT urgency is classified, confirm that the on-call Communication resource was created.""",
    tools=[FunctionTool(run_referral_pipeline)],
)

from shared.fhir_client import (
    get_service_request,
    get_encounter,
    get_conditions,
    get_observations,
    get_medication_requests,
    get_practitioner,
)
from shared.models import ReferralContext, Practitioner


def load_referral_context(service_request_id: str) -> ReferralContext:
    """
    RL-1: Load all clinical context for a referral from FHIR.
    Reads ServiceRequest and all linked resources.
    Returns a ReferralContext passed to all subsequent steps.
    """
    sr = get_service_request(service_request_id)

    # Extract referral reason
    reason_text = ""
    for reason in sr.get("reasonCode", []):
        for coding in reason.get("coding", []):
            if coding.get("display"):
                reason_text = coding["display"]
                break
        if not reason_text:
            reason_text = reason.get("text", "")
        if reason_text:
            break

    # Extract specialty from category
    specialty = ""
    for category in sr.get("category", []):
        for coding in category.get("coding", []):
            if coding.get("display"):
                specialty = coding["display"]
                break
        if not specialty:
            specialty = category.get("text", "")
        if specialty:
            break

    # Extract patient ID
    patient_ref = sr.get("subject", {}).get("reference", "")
    patient_id = patient_ref.replace("Patient/", "") if patient_ref else ""

    # Extract patient name if available
    patient_name = None
    contained = sr.get("contained", [])
    for resource in contained:
        if resource.get("resourceType") == "Patient":
            name_block = resource.get("name", [{}])[0]
            given = " ".join(name_block.get("given", []))
            family = name_block.get("family", "")
            patient_name = f"{given} {family}".strip() or None

    # Extract encounter ID
    encounter_ref = sr.get("encounter", {}).get("reference", "")
    encounter_id = encounter_ref.replace("Encounter/", "") if encounter_ref else None

    # Load encounter if referenced
    if encounter_id:
        try:
            get_encounter(encounter_id)
        except Exception:
            encounter_id = None

    # Load linked clinical resources
    conditions = []
    observations = []
    medication_requests = []

    if patient_id:
        try:
            conditions = get_conditions(patient_id)
        except Exception:
            conditions = []

        try:
            observations = get_observations(patient_id)
        except Exception:
            observations = []

        try:
            medication_requests = get_medication_requests(patient_id)
        except Exception:
            medication_requests = []

    # Resolve performer (may be None)
    performer = None
    performer_refs = sr.get("performer", [])
    if performer_refs:
        performer_ref = performer_refs[0].get("reference", "")
        practitioner_id = performer_ref.replace("Practitioner/", "") if performer_ref else None
        if practitioner_id:
            performer = get_practitioner(practitioner_id)

    return ReferralContext(
        service_request_id=service_request_id,
        referral_reason=reason_text or "Referral reason not specified",
        specialty=specialty or "Specialty not specified",
        patient_id=patient_id,
        patient_name=patient_name,
        encounter_id=encounter_id,
        conditions=conditions,
        observations=observations,
        medication_requests=medication_requests,
        performer=performer,
    )

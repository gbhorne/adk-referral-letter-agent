import base64
from datetime import datetime, timezone
from shared.models import ReferralContext, UrgencyClassification


def assemble_document(
    context: ReferralContext,
    urgency: UrgencyClassification,
    letter_text: str,
) -> dict:
    """
    RL-5: Assemble a FHIR DocumentReference for the referral letter.
    LOINC 57133-1 -- Referral Note
    docStatus: preliminary (requires referring clinician promotion to final)
    """
    encoded_letter = base64.b64encode(letter_text.encode("utf-8")).decode("utf-8")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Resolve performer display
    if context.performer and context.performer.name:
        performer_display = context.performer.name
    elif context.performer and context.performer.specialty:
        performer_display = f"{context.performer.specialty} Specialist (TBD)"
    else:
        performer_display = "performer TBD"

    document = {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "preliminary",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "57133-1",
                    "display": "Referral note",
                }
            ],
            "text": "Referral Note",
        },
        "subject": {
            "reference": f"Patient/{context.patient_id}"
        },
        "date": now,
        "description": f"Referral letter -- {context.specialty} -- {urgency.urgency.value} -- Performer: {performer_display}",
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": encoded_letter,
                    "title": f"Referral Letter -- {context.specialty}",
                    "creation": now,
                }
            }
        ],
        "context": {
            "related": [
                {"reference": f"ServiceRequest/{context.service_request_id}"}
            ]
        },
    }

    if context.encounter_id:
        document["context"]["encounter"] = [
            {"reference": f"Encounter/{context.encounter_id}"}
        ]

    return document

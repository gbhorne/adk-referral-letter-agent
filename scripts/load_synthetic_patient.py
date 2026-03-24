"""
load_synthetic_patient.py
Creates synthetic FHIR resources in the referral-letter-store for testing.

Creates:
- 1 Patient
- 1 Practitioner (for performer-populated test)
- 1 Encounter
- 3 ServiceRequest variants:
    SR-1: ROUTINE referral, performer populated
    SR-2: ROUTINE referral, no performer
    SR-3: EMERGENT-appropriate referral, no performer
"""

import requests
import json
import google.auth
import google.auth.transport.requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import FHIR_STORE_URL


def get_headers():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/fhir+json",
    }


def create_resource(resource_type: str, body: dict) -> dict:
    url = f"{FHIR_STORE_URL}/{resource_type}"
    response = requests.post(url, json=body, headers=get_headers())
    response.raise_for_status()
    return response.json()


def main():
    print("Loading synthetic FHIR resources into referral-letter-store...\n")

    # Create Patient
    patient = create_resource("Patient", {
        "resourceType": "Patient",
        "name": [{"use": "official", "family": "Synthea", "given": ["Test"]}],
        "gender": "male",
        "birthDate": "1965-04-12",
    })
    patient_id = patient["id"]
    print(f"Patient created: {patient_id}")

    # Create Practitioner
    practitioner = create_resource("Practitioner", {
        "resourceType": "Practitioner",
        "name": [{"use": "official", "family": "Cardiology", "given": ["Dr. Jane"]}],
        "qualification": [
            {
                "code": {
                    "coding": [{"display": "Cardiology"}],
                    "text": "Cardiology",
                }
            }
        ],
    })
    practitioner_id = practitioner["id"]
    print(f"Practitioner created: {practitioner_id}")

    # Create Encounter
    encounter = create_resource("Encounter", {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
        "subject": {"reference": f"Patient/{patient_id}"},
    })
    encounter_id = encounter["id"]
    print(f"Encounter created: {encounter_id}")

    # Create Condition (chest pain)
    condition_chest = create_resource("Condition", {
        "resourceType": "Condition",
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
        },
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": "29857009", "display": "Chest pain"}],
            "text": "Chest pain",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
    })
    print(f"Condition (chest pain) created: {condition_chest['id']}")

    # Create Condition (hypertension)
    condition_htn = create_resource("Condition", {
        "resourceType": "Condition",
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
        },
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertensive disorder"}],
            "text": "Hypertension",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
    })
    print(f"Condition (hypertension) created: {condition_htn['id']}")

    # Create Observation (troponin elevated)
    observation = create_resource("Observation", {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "10839-9", "display": "Troponin I"}],
            "text": "Troponin I",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": "2026-03-23T10:00:00Z",
        "valueQuantity": {"value": 0.8, "unit": "ng/mL", "system": "http://unitsofmeasure.org"},
    })
    print(f"Observation (troponin) created: {observation['id']}")

    # Create MedicationRequest (aspirin)
    medication = create_resource("MedicationRequest", {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1191", "display": "Aspirin"}],
            "text": "Aspirin 81mg daily",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
    })
    print(f"MedicationRequest (aspirin) created: {medication['id']}")

    # SR-1: ROUTINE referral, performer populated
    sr1 = create_resource("ServiceRequest", {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "category": [{"coding": [{"display": "Cardiology"}], "text": "Cardiology"}],
        "reasonCode": [{"coding": [{"display": "Chest pain evaluation"}], "text": "Chest pain evaluation"}],
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "performer": [{"reference": f"Practitioner/{practitioner_id}"}],
    })
    print(f"\nSR-1 (ROUTINE, performer populated): {sr1['id']}")

    # SR-2: ROUTINE referral, no performer
    sr2 = create_resource("ServiceRequest", {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "category": [{"coding": [{"display": "Gastroenterology"}], "text": "Gastroenterology"}],
        "reasonCode": [{"coding": [{"display": "Chronic abdominal pain workup"}], "text": "Chronic abdominal pain workup"}],
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
    })
    print(f"SR-2 (ROUTINE, no performer): {sr2['id']}")

    # SR-3: EMERGENT-appropriate referral (elevated troponin + chest pain)
    sr3 = create_resource("ServiceRequest", {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "priority": "urgent",
        "category": [{"coding": [{"display": "Cardiology"}], "text": "Cardiology"}],
        "reasonCode": [
            {
                "coding": [{"display": "Acute chest pain with elevated troponin -- rule out NSTEMI"}],
                "text": "Acute chest pain with elevated troponin -- rule out NSTEMI",
            }
        ],
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
    })
    print(f"SR-3 (EMERGENT-appropriate, no performer): {sr3['id']}")

    print("\nAll resources created successfully.")
    print("\nServiceRequest IDs for testing:")
    print(f"  SR-1 (with performer): {sr1['id']}")
    print(f"  SR-2 (no performer):   {sr2['id']}")
    print(f"  SR-3 (EMERGENT):       {sr3['id']}")


if __name__ == "__main__":
    main()

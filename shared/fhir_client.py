import requests
import google.auth
import google.auth.transport.requests
from shared.config import FHIR_STORE_URL
from shared.models import Practitioner


def _get_headers() -> dict:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/fhir+json",
    }


def get_service_request(service_request_id: str) -> dict:
    url = f"{FHIR_STORE_URL}/ServiceRequest/{service_request_id}"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json()


def get_encounter(encounter_id: str) -> dict:
    url = f"{FHIR_STORE_URL}/Encounter/{encounter_id}"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json()


def get_conditions(patient_id: str) -> list[dict]:
    url = f"{FHIR_STORE_URL}/Condition?patient={patient_id}"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    bundle = response.json()
    return [entry["resource"] for entry in bundle.get("entry", [])]


def get_observations(patient_id: str) -> list[dict]:
    url = f"{FHIR_STORE_URL}/Observation?patient={patient_id}"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    bundle = response.json()
    return [entry["resource"] for entry in bundle.get("entry", [])]


def get_medication_requests(patient_id: str) -> list[dict]:
    url = f"{FHIR_STORE_URL}/MedicationRequest?patient={patient_id}"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    bundle = response.json()
    return [entry["resource"] for entry in bundle.get("entry", [])]


def get_practitioner(practitioner_id: str) -> Practitioner | None:
    if not practitioner_id:
        return None
    try:
        url = f"{FHIR_STORE_URL}/Practitioner/{practitioner_id}"
        response = requests.get(url, headers=_get_headers())
        response.raise_for_status()
        resource = response.json()
        name_block = resource.get("name", [{}])[0]
        given = " ".join(name_block.get("given", []))
        family = name_block.get("family", "")
        full_name = f"{given} {family}".strip() or None
        qualification = resource.get("qualification", [{}])[0]
        specialty = qualification.get("code", {}).get("text", None)
        return Practitioner(
            id=practitioner_id,
            name=full_name,
            specialty=specialty,
        )
    except Exception:
        return None


def create_document_reference(document: dict) -> dict:
    url = f"{FHIR_STORE_URL}/DocumentReference"
    response = requests.post(url, json=document, headers=_get_headers())
    response.raise_for_status()
    return response.json()


def create_communication(communication: dict) -> dict:
    url = f"{FHIR_STORE_URL}/Communication"
    response = requests.post(url, json=communication, headers=_get_headers())
    response.raise_for_status()
    return response.json()

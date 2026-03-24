from enum import Enum
from typing import Optional
from pydantic import BaseModel


class UrgencyLevel(str, Enum):
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    EMERGENT = "EMERGENT"


class UrgencyClassification(BaseModel):
    urgency: UrgencyLevel
    clinical_rationale: str
    confidence: float


class Practitioner(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    specialty: Optional[str] = None


class ReferralContext(BaseModel):
    service_request_id: str
    referral_reason: str
    specialty: str
    patient_id: str
    patient_name: Optional[str] = None
    encounter_id: Optional[str] = None
    conditions: list[dict] = []
    observations: list[dict] = []
    medication_requests: list[dict] = []
    performer: Optional[Practitioner] = None


class ReferralDocument(BaseModel):
    service_request_id: str
    patient_id: str
    urgency: UrgencyLevel
    letter_text: str
    performer_name: Optional[str] = None
    specialty: str

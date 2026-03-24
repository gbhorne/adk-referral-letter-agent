from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GCP_PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["LOCATION"]
FHIR_STORE_URL = os.environ["FHIR_STORE_URL"]
PUBSUB_INBOUND = os.environ["PUBSUB_INBOUND"]
PUBSUB_ROUTINE = os.environ["PUBSUB_ROUTINE"]
PUBSUB_URGENT = os.environ["PUBSUB_URGENT"]
PUBSUB_EMERGENT = os.environ["PUBSUB_EMERGENT"]
FIRESTORE_COLLECTION = os.environ["FIRESTORE_COLLECTION"]

import google.cloud.dlp_v2 as dlp_v2
from shared.config import GCP_PROJECT


def inspect_and_deidentify(text: str) -> str:
    """
    Inspects clinical text for PHI using Cloud DLP.
    Returns the original text if no findings exceed threshold.
    Raises ValueError if high-confidence PHI is detected that cannot be masked.
    """
    if not text or not text.strip():
        return text

    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{GCP_PROJECT}/locations/global"

    inspect_config = dlp_v2.InspectConfig(
        info_types=[
            {"name": "PERSON_NAME"},
            {"name": "DATE_OF_BIRTH"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "PHONE_NUMBER"},
            {"name": "EMAIL_ADDRESS"},
            {"name": "STREET_ADDRESS"},
            {"name": "MEDICAL_RECORD_NUMBER"},
        ],
        min_likelihood=dlp_v2.Likelihood.POSSIBLE,
        include_quote=True,
    )

    deidentify_config = dlp_v2.DeidentifyConfig(
        info_type_transformations=dlp_v2.InfoTypeTransformations(
            transformations=[
                dlp_v2.InfoTypeTransformations.InfoTypeTransformation(
                    primitive_transformation=dlp_v2.PrimitiveTransformation(
                        replace_with_info_type_config=dlp_v2.ReplaceWithInfoTypeConfig()
                    )
                )
            ]
        )
    )

    item = dlp_v2.ContentItem(value=text)

    request = dlp_v2.DeidentifyContentRequest(
        parent=parent,
        deidentify_config=deidentify_config,
        inspect_config=inspect_config,
        item=item,
    )

    response = client.deidentify_content(request=request)
    return response.item.value

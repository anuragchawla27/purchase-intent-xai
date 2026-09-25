"""
Pydantic request/response schemas for the prediction API.
Field definitions are driven by the same config used everywhere else
in the project, so the API schema can never silently drift from the
model's actual expected input.
"""

from pydantic import BaseModel, ConfigDict, Field


class SessionInput(BaseModel):
    """
    A single e-commerce session, matching the raw feature schema
    the trained pipeline expects (before preprocessing).
    """

    Administrative: int = Field(
        ..., ge=0, description="Number of administrative pages visited"
    )
    Administrative_Duration: float = Field(..., ge=0)
    Informational: int = Field(..., ge=0)
    Informational_Duration: float = Field(..., ge=0)
    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0)
    BounceRates: float = Field(..., ge=0, le=1)
    ExitRates: float = Field(..., ge=0, le=1)
    PageValues: float = Field(..., ge=0)
    SpecialDay: float = Field(..., ge=0, le=1)
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Administrative": 1,
                "Administrative_Duration": 14.65,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 19,
                "ProductRelated_Duration": 283.88,
                "BounceRates": 0.008,
                "ExitRates": 0.042,
                "PageValues": 68.58,
                "SpecialDay": 0.0,
                "Month": "May",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    )


class ContributingFeature(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    prediction: str
    purchase_probability: float
    confidence: str
    decision_threshold: float
    top_contributing_features: list[ContributingFeature]

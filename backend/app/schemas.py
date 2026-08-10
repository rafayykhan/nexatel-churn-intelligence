"""Request/response contracts for the NexaTel churn API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerIn(BaseModel):
    """One customer, in the vocabulary the retention team already uses.

    Field names mirror the database columns so an agent filling the form
    and an analyst querying SQL are talking about the same thing.
    """
    gender: Literal["Male", "Female"] = "Female"
    senior_citizen: int = Field(0, ge=0, le=1)
    partner: YesNo = "No"
    dependents: YesNo = "No"
    tenure: int = Field(..., ge=0, le=120, description="months with NexaTel")
    phone_service: YesNo = "Yes"
    multiple_lines: Literal["Yes", "No", "No phone service"] = "No"
    internet_service: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    online_security: YesNoInternet = "No"
    online_backup: YesNoInternet = "No"
    device_protection: YesNoInternet = "No"
    tech_support: YesNoInternet = "No"
    streaming_tv: YesNoInternet = "No"
    streaming_movies: YesNoInternet = "No"
    contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    paperless_billing: YesNo = "Yes"
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"] = "Electronic check"
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: Optional[float] = Field(None, ge=0)
    customer_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "senior_citizen": 0, "partner": "No",
                "dependents": "No", "tenure": 3, "phone_service": "Yes",
                "multiple_lines": "No", "internet_service": "Fiber optic",
                "online_security": "No", "online_backup": "No",
                "device_protection": "No", "tech_support": "No",
                "streaming_tv": "Yes", "streaming_movies": "Yes",
                "contract": "Month-to-month", "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 94.4, "total_charges": 283.2,
            }
        }
    }


class Factor(BaseModel):
    feature: str
    label: str
    impact: float
    value: object = None


class PredictionOut(BaseModel):
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    risk_score: int = Field(..., description="probability as 0-100 for display")
    decision_threshold: float
    flagged_for_outreach: bool
    revenue_at_risk_annual: float
    risk_factors: list[Factor]
    protective_factors: list[Factor]
    recommended_action: str
    model_name: str


class BatchIn(BaseModel):
    customers: list[CustomerIn]


class BatchOut(BaseModel):
    results: list[PredictionOut]
    summary: dict

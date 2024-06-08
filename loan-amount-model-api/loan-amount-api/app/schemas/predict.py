from typing import Any, List, Optional

from pydantic import BaseModel
from regression.processing.validation import LoanDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleLoanDataInput(BaseModel):
    inputs: List[LoanDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "loan_id": "lp001003",
                        "gender": "male",
                        "married": "yes",
                        "dependents": 1,
                        "education": "graduate",
                        "self_employed": "no",
                        "applicantincome": 4583,
                        "coapplicantincome": 1508.0,
                        "loanamount": 128.0,
                        "loan_amount_term": 360.0,
                        "credit_history": 1.0,
                        "property_area": "Rural",
                        "loan_status": "N",
                    }
                ]
            }
        }

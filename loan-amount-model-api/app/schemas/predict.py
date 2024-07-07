from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

class LoanDataInputSchema(BaseModel):
    loan_id: Optional[str]
    gender: Optional[str]
    married: Optional[str]
    dependents: Optional[str]
    education: Optional[str]
    self_employed: Optional[str]
    applicantincome: Optional[int]
    coapplicantincome: Optional[float]
    # loanamount: Optional[float]
    loan_amount_term: Optional[float]
    credit_history: Optional[float]
    property_area: Optional[str]
    loan_status: Optional[str]

    model_config = {
        "json_schema_extra": {
            "examples" : 
                [ 
                     {
                        "loan_id": "lp001003",
                        "gender": "male",
                        "married": "yes",
                        "dependents": "1",
                        "education": "graduate",
                        "self_employed": "no",
                        "applicantincome": 4583,
                        "coapplicantincome": 1508.0,
                        # "loanamount": 128.0,
                        "loan_amount_term": 360.0,
                        "credit_history": 1.0,
                        "property_area": "Rural",
                        "loan_status": "N",
                    }
                ]  
        }
    }

# class MultipleLoanDataInput(BaseModel):
#     input: List[LoanDataInputSchema]

#     model_config = {
#         "json_schema_extra": {
#             "examples" : 
#                 [ 
#                      {
#                         "loan_id": "lp001003",
#                         "gender": "male",
#                         "married": "yes",
#                         "dependents": 1,
#                         "education": "graduate",
#                         "self_employed": "no",
#                         "applicantincome": 4583,
#                         "coapplicantincome": 1508.0,
#                         "loanamount": 128.0,
#                         "loan_amount_term": 360.0,
#                         "credit_history": 1.0,
#                         "property_area": "Rural",
#                         "loan_status": "N",
#                     }
#                 ]  
#         }
#     }


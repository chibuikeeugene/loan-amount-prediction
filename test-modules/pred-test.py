import requests
from requests.exceptions import HTTPError
from pathlib import Path

from sklearn.multiclass import OutputCodeClassifier

test_data = {
                "loan_id": "lp001003",
                "gender": "male",
                "married": "yes",
                "dependents": "1",
                "education": "graduate",
                "self_employed": "no",
                "applicantincome": 4583,
                "coapplicantincome": 1508.0,
                "loan_amount_term": 360.0,
                "credit_history": 1.0,
                "property_area": "Rural",
                "loan_status": "N",
            }

url = "http://127.0.0.1:88/api/v1/predict"

try:
    with requests.request(method='POST', url=url, data = test_data) as response:
        result = response.json()
        print(result.get("predictions"))
except (HTTPError, TimeoutError, OSError) as error:
    print(f"Reason: {error}")
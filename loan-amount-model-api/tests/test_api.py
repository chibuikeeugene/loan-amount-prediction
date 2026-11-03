import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = test_data.dropna().to_dict(orient="records")
    

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200, response.text
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None

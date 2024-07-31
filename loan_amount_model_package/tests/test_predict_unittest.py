# UNIT TEST 1

import numpy as np

from loan_amount_model_package import predict


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 119.27
    expected_no_predictions = 367

    # When
    result = predict.make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert np.round(predictions[0], 2) == expected_first_prediction_value

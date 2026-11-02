from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from loan_amount_model_package.config.core import config
from loan_amount_model_package.processing.data_manager import load_dataset

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return load_dataset(filename=config.app_configs.test_data)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}

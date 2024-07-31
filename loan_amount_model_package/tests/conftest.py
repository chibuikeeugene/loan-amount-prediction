# import our pytest package
import pytest

# load the config data and the dataset
from loan_amount_model_package.config.core import config
from loan_amount_model_package.processing.data_manager import load_dataset    



@pytest.fixture()
def sample_input_data():
    return load_dataset(filename=config.app_configs.test_data)

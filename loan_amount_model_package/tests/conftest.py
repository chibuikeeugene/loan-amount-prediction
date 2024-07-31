# import our pytest package
from json import load
import pytest

# load the config data and the dataset
from loan_amount_model_package.config.core import config
from loan_amount_model_package.processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split



@pytest.fixture()
def sample_input_data():
    return load_dataset(filename=config.app_configs.test_data)

@pytest.fixture
def pipeline_input():
    data = load_dataset(filename=config.app_configs.training_data)

    # split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
                                        data[config.model_configs.features],
                                        data[config.model_configs.target], 
                                        test_size=0.2, 
                                        random_state=1)
    
    return x_train, x_test, y_train, y_test


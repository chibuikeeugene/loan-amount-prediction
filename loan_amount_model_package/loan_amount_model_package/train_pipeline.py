import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from loan_amount_model_package.config.core import config  # noqa: E402
from loan_amount_model_package.pipeline import loan_amount_pipeline  # noqa: E402
from loan_amount_model_package.processing.data_manager import (  # noqa: E402
    load_dataset,
    save_pipeline,
)


def run_training() -> None:
    """ " train the model."""

    logger.info("loading and preprocessing the train data...")
    # read the training data
    data = load_dataset(filename=config.app_configs.training_data)

    logger.info("splitting the dataset into train and test ...")
    # divide the train and val
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_configs.features],
        data[config.model_configs.target],
        test_size=config.model_configs.test_size,
        random_state=config.model_configs.random_state,
    )

    # to address missing values in loanamount, which is  our target variable
    # we can replace nan with 0
    y_train = y_train.fillna(0)

    # we apply logarithm transformation
    y_train = np.log1p(y_train)

    # fit the model
    loan_amount_pipeline.fit(x_train, y_train)

    # persist the trained model
    save_pipeline(pipeline_to_persist=loan_amount_pipeline)
    logger.info("saving and persisting the model pipeline to disk...")


if __name__ == "__main__":
    run_training()

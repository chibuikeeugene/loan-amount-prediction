# INTEGRATION TEST
import numpy as np

# For data encoding
from feature_engine.encoding import OrdinalEncoder

# ===== for feature engineering ===== #
# For imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

# For Data transformation
from feature_engine.transformation import LogCpTransformer

# from sklearn.linear_model import LassoCV
# from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from loan_amount_model_package.config.core import config
from loan_amount_model_package.processing.preprocessor import Mapper

# importing necessary libraries


def test_pipeline_scales_input_data(pipeline_input):
    # given the original data
    x_train, x_test, y_train, y_test = pipeline_input

    # to address missing values in loanamount, which is  our target variable
    # we can replace nan with 0
    y_train = y_train.fillna(0)

    # we apply logarithm transformation
    y_train = np.log1p(y_train)
    loan_amount_pipeline.fit(x_train, y_train)

    # when the data is scaled
    scaled_data = loan_amount_pipeline.transform(x_train)

    # then check the values of a single record and ensure they are scaled between 0 and 1
    # this is a requirement for the model to work correctly
    for value in scaled_data[0]:
        assert 0 <= value <= 1


# we create model pipeline by removing the final estimator so
# as to call the transform method of the pipeline itself
loan_amount_pipeline = Pipeline(
    [
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_configs.num_var_na),
        ),
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_configs.num_var_na
            ),
        ),
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing", variables=config.model_configs.cat_vars_na
            ),
        ),
        (
            "credit_history_mapping",
            Mapper(
                variable=config.model_configs.credit_var_mapper,
                mapping=config.model_configs.mapper,
            ),
        ),
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method="arbitrary", variables=config.model_configs.cat_vars
            ),
        ),
        (
            "log",
            LogCpTransformer(
                variables=config.model_configs.num_cont_vars, C=config.model_configs.C
            ),
        ),
        ("scaler", MinMaxScaler()),
    ]
)

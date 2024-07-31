import os
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel
from strictyaml import YAML, load

# Top level project directories
PACKAGE_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# PACKAGE_ROOT = Path(regression.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    application-level config
    """

    package_name: str
    training_data: str
    test_data: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    all configuration relevant to model
    training and feature engineering
    """

    target: str
    features: List[str]
    test_size: float
    C: int
    random_state: int
    learning_rate: float
    max_depth: int
    max_iter: int
    tol: float
    cat_vars: List[Union[str, int]]
    cat_vars_na: List[Union[str, int]]
    num_var_na: List[Union[str, int]]
    num_cont_vars: List[Union[str, int]]
    mapper: dict[str, str]
    credit_var_mapper: List[str]


class Config(BaseModel):
    """master config object"""

    app_configs: AppConfig
    model_configs: ModelConfig


def find_config_file() -> Path:
    """locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:  # type: ignore
    """parse the YAML containing the package configuration"""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:  # type: ignore
    """run validation on config values"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()  # type: ignore

    # specify the data attribute from the strictyaml YAML type
    _config = Config(
        app_configs=AppConfig(**parsed_config.data),  # type: ignore
        model_configs=ModelConfig(**parsed_config.data),  # type: ignore
    )

    return _config


config = create_and_validate_config()  # type: ignore

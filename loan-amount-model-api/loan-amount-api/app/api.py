
import json
import typing as t

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger



from loan_amount_model_package import __version__ as model_version # type: ignore
from loan_amount_model_package.predict import make_prediction # type: ignore

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, 
        api_version=__version__, 
        model_version=model_version
    )

    return dict(health)


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: t.List[schemas.LoanDataInputSchema]) -> t.Any:
    """
    Make loan amount predictions with the regression model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data))

    logger.info(f"Making prediction on inputs: {input_data}")
    results = make_prediction(input_data=input_df.replace({np.nan: 0}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results

# import libraries
from json import load
import os
from pdb import run
import joblib
import pandas as pd
import uuid
from sklearn.pipeline import Pipeline 
from typing import Union
from datetime import datetime

def generateIdsForEachRecord(n: int) -> list:
    """function to generate unique ids for each record"""
    loan_ids = []

    for i in range(n):
        loan_ids.append(str(uuid.uuid4()))
    return loan_ids


def readData(filename: str) -> pd.DataFrame:
    """function to read dataset"""
    data = pd.read_csv(filename)

    # formatting column names for consistency
    data.columns = data.columns.str.lower()

    # dropping irrelevant columns
    df = data.drop(['loan_id', 'gender', 'loan_status', 'loanamount'], axis=1)

    # converting the credit history variable to categorical
    df['credit_history'] = df['credit_history'].astype('str')
    return df


def loadPipelineModel() -> Pipeline:
    """function to load our model pipeline"""
    current_path = os.path.dirname(__file__)
    for file in os.listdir(current_path):
        if file.endswith('pkl'):
            model_path = os.path.join(current_path, file)
            with open(model_path, 'rb') as f:
                model = joblib.load(f)
    return model


def save_results(df: pd.DataFrame, pred, output_file) -> None:
    """ function to save the prediction results"""
    df_result =  pd.DataFrame()
    df_result['loan_ids'] = generateIdsForEachRecord(len(df))
    df_result['applicantincome'] = df['applicantincome']
    df_result['coapplicantincome'] = df['coapplicantincome']
    df_result['loan_predictions'] =  pred

    df_result.to_csv(output_file, index=False)


def applyModel(input_file: str, output_file: str):
    """obtain predictions from the data"""

    # call the read data function
    refined_dataset = readData(input_file)

    # load the model pipeline
    model = loadPipelineModel()

    # applying our model pipeline
    preds = model.predict(refined_dataset)

    # save the results
    save_results(refined_dataset, preds, output_file)


def getFile(run_date: datetime):
    """ function to get the required file to save results"""
    year = run_date.year
    month = run_date.month
    day = run_date.day
    output_file =f'./output_predictions_{year:04d}_{month:02d}_{day:02d}.csv'
    return output_file


def generatePredictions(run_date:datetime, input_file: str):
    """ function to generate predictions"""
    input_file = input_file
    output_file = getFile(run_date)
    applyModel(input_file, output_file)



def run_program():
    generatePredictions(
        run_date=datetime.now(), 
        input_file='./test.csv'
    )



if __name__ == '__main__':
    run_program()
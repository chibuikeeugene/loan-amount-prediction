from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/chibuikeeugene/loan-amount-prediction.git",
        entrypoint="./batch_deployment_and_orchestration/loan_model_inference.py:run_program",
    ).deploy(
        name="my-first-deployment",
        work_pool_name="loan-app-service-pool",
        cron="0 1 * * *",
    )
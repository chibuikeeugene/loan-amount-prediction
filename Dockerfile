# Define base image
FROM python:3.9.12-slim

# Set environment variables
ENV buildTag=1.0

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

# setting the working directory
WORKDIR /opt/loan-amount-model-project/

# creating a user other than root 
RUN adduser --disabled-password --gecos '' loan-user

# install and update system dependencies
RUN apt-get update \
    && apt-get install

# copy project files and folder to our container directory
COPY . loan-amount-model-api /opt/loan-amount-model-project/

COPY . pyproject.toml /opt/loan-amount-model-project/

RUN pip install --no-cache-dir poetry

# updating PATH to include poetry's bin directory
ENV PATH="${PATH}:~/.poetry/bin"

RUN poetry install

RUN /opt/loan-amount-model-project/.venv/bin/pip install loan_amount_model_package==0.0.7

# Grant ownership and permissions to the user for the application directory
RUN chown -R loan-user:loan-user /opt/loan-amount-model-project/
RUN chmod -R 777 /opt/loan-amount-model-project/
    
USER loan-user 

EXPOSE 8002

CMD ["poetry", "run", "uvicorn", "loan-amount-api.app.main:app", "--host", "localhost", "--port", "8002" ]




# ----------stage: builder------------- #
# Define base image
FROM python:3.9.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# install and update system dependencies
RUN apt-get update \
    && apt-get install

COPY ./requirements/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


# ------------ stage: runtime --------------- #
FROM python:3.9.12-slim AS runtime

# Set environment variables
ENV buildTag=1.0 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PATH="/opt/venv/bin:$PATH"

# setting the working directory
WORKDIR /project

# copy dependency files from build to runtime
COPY --from=builder /opt/venv /opt/venv

# creating a user other than root 
RUN useradd --create-home --uid 10000 loan-user

# Copy,Grant ownership and permissions to the user for the application directory
COPY --chown=loan-user:loan-user /loan-amount-model-api /project/loan-amount-model-api
COPY --chown=loan-user:loan-user /loan_amount_model_package /project/loan_amount_model_package
RUN chmod -R 2755 /project

# set the app user
USER loan-user 

EXPOSE 8001

HEALTHCHECK \
--interval=30s \
--timeout=30s \
--start-period=5s \
--retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://0.0.0.0:8001/health, timeout=30)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001" ]




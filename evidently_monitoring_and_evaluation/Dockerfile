FROM python:3.9.12-slim

WORKDIR /opt/evidently/monitoring/

RUN apt-get update \
    && apt-get installn \
    && apt-get install git-all

RUN adduser --disabled-password --gecos '' loan_admin

COPY . evidently_monitoring_and_evaluation /opt/evidently/monitoring/

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# configure environment path
ENV PYTHONPATH=/opt/evidently/monitoring

# Grant ownership and permissions to the user for the application directory
RUN chown -R loan_admin:loan_admin /opt/evidently/monitoring/
RUN chmod -R 777 /opt/evidently/monitoring/
    
USER loan_admin 

CMD [ "jupyter notebook" ]
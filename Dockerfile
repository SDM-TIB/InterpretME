# use an official Python image as base
FROM python:3.8.11-slim-buster

# install dependencies
COPY requirements.txt /InterpretME/requirements.txt
COPY InterpretME/Predictive_pipeline/validating_models/requirements.txt /InterpretME/requirements-validating_models.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*  && \
    python -m pip install --no-cache-dir --upgrade pip==22.0.* setuptools==58.0.4 && \
    python -m pip install --no-cache-dir -r /InterpretME/requirements.txt && \
    python -m pip install --no-cache-dir -r /InterpretME/requirements-validating_models.txt

# copy the source code
COPY InterpretME /InterpretME
WORKDIR /InterpretME

# keep the container running
# since InterpretME is not yet a service
ENTRYPOINT ["tail", "-f", "/dev/null"]

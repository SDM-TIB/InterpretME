# use an official Python image as base
FROM python:3.8.13-slim-bullseye

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*  && \
    python -m pip install --no-cache-dir --upgrade pip==22.0.* setuptools==58.0.4

# copy the source code and install InterpretME
COPY LIBRARY.md /InterpretME/
COPY LICENSE /InterpretME/
COPY MANIFEST.in /InterpretME/
COPY README.md /InterpretME/
COPY setup.py /InterpretME/
COPY InterpretME /InterpretME/InterpretME
WORKDIR /InterpretME
RUN python -m pip install --no-cache-dir -e .

# keep the container running since InterpretME is not yet a service
ENTRYPOINT ["tail", "-f", "/dev/null"]

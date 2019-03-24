FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3-pip \
    python3-dev \
    virtualenv

COPY requirements.txt .
RUN pip3 install setuptools wheel
RUN pip3 install -r requirements.txt
COPY app.py ./
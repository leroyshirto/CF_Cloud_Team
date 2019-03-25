FROM python:3.5

COPY requirements.txt .
RUN pip install setuptools wheel
RUN pip install -r requirements.txt
COPY app.py ./
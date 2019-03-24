FROM templum/openvino-base:1 as DEV

FROM ubuntu:16.04 as PROD

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3-pip \
    python3-dev \
    virtualenv \
    libgomp1

COPY --from=DEV /dldt-2018_R5/inference-engine/bin/intel64/Release/lib/*.so /usr/local/lib/
COPY --from=DEV /dldt-2018_R5/inference-engine/ie_bridges/python/bin/intel64/Release/python_api/python3.5/openvino/ /usr/local/lib/openvino/
COPY --from=DEV /dldt-2018_R5/mklml_lnx_2019.0.1.20180928/lib/lib*.so /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib
COPY requirements.txt .
RUN pip3 install setuptools wheel
RUN pip3 install -r requirements.txt
COPY predict.py classes.py ./
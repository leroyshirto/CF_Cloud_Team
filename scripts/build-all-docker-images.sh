#! /bin/sh

# Fetcher
docker build -t templum/openvino-fetch:2 -f Fetch/Dockerfile Fetch

# Predictor
docker build -t templum/openvino-serve:7 -f Prediction/Dockerfile Prediction

# Results
docker build -t leroyshirtofh/oisp-results-submission:1 -f Results/Dockerfile Results

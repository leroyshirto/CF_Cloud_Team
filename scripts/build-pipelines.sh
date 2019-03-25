#! /bin/sh

# Prediction Pipeline
dsl-compile --py Pipeline/prediction_pipeline.py --output Pipeline/prediction_pipeline.tar.gz

# Simple Secquce pipline for testing passing output from one step to the next
dsl-compile --py Pipeline/simple_sequence.py --output Pipeline/simple_sequence.tar.gz
# Simple

## Compiling Scripts

Installing `dsl-compile` `pip3 install https://storage.googleapis.com/ml-pipeline/release/0.1.12/kfp.tar.gz --upgrade`

Compiling `simple_sequence.py`:

`dsl-compile --py Pipeline/simple_sequence.py --output Pipeline/simple_sequence.tar.gz`

Compiling `prediction_pipeline.py`:

`dsl-compile --py Pipeline/prediction_pipeline.py --output Pipeline/prediction_pipeline.tar.gz`


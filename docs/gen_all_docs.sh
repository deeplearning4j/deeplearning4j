#!/usr/bin/env bash

# Generate Keras docs
python generate_docs.py \
    --project keras-import \
    --code ../deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/


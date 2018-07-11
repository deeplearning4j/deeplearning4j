#!/usr/bin/env bash

# Generate Keras docs
python generate_docs.py \
    --project keras-import \
    --language java \
    --code ../deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/ \
    --docs_root deeplarning4j.org/keras \
    --out_language en

# Generate ScalNet docs
python generate_docs.py \
    --project scalnet \
    --language scala \
    --code ../scalnet/src/main/scala/org/deeplearning4j/scalnet/ \
    --docs_root deeplarning4j.org/scalnet

# Generate SameDiff docs
python generate_docs.py \
    --project samediff \
    --language java \
    --code ../nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/ \
    --docs_root deeplarning4j.org/samediff
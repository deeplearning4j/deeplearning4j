#!/usr/bin/env bash

################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################


python generate_docs.py \
    --project deeplearning4j \
    --language java \
    --code ../deeplearning4j \
    --out_language en

python generate_docs.py \
    --project deeplearning4j-nn \
    --language java \
    --code ../deeplearning4j \
    --out_language en

python generate_docs.py \
    --project deeplearning4j-nlp \
    --language java \
    --code ../deeplearning4j \
    --out_language en

python generate_docs.py \
    --project deeplearning4j-spark \
    --language java \
    --code ../deeplearning4j \
    --out_language en

python generate_docs.py \
    --project deeplearning4j-zoo \
    --language java \
    --code ../deeplearning4j \
    --out_language en

python generate_docs.py \
    --project datavec \
    --language java \
    --code ../datavec \
    --out_language en

python generate_docs.py \
    --project nd4j \
    --language java \
    --code ../nd4j \
    --out_language en

python generate_docs.py \
    --project nd4j-nn \
    --language java \
    --code ../nd4j \
    --out_language en

python generate_docs.py \
    --project arbiter \
    --language java \
    --code ../arbiter \
    --out_language en

python generate_docs.py \
    --project keras-import \
    --language java \
    --code ../deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/ \
    --docs_root deeplarning4j.org/keras \
    --out_language en

# python generate_docs.py \
#     --project scalnet \
#     --language scala \
#     --code ../scalnet/src/main/scala/org/deeplearning4j/scalnet/ \
#     --docs_root deeplarning4j.org/scalnet

# python generate_docs.py \
#     --project samediff \
#     --language java \
#     --code ../nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/ \
#     --docs_root deeplarning4j.org/samediff
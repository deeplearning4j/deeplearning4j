#!/bin/bash
#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

# Exit on error
set -e

if [ -z "$FLATC_PATH" ]; then
    echo "FLATC_PATH not set"
    exit 1
fi

echo "Using flatc compiler from: $FLATC_PATH"
if [ ! -f "$FLATC_PATH" ]; then
    echo "Error: flatc not found at $FLATC_PATH"
    exit 1
fi

if [ ! -x "$FLATC_PATH" ]; then
    echo "Error: $FLATC_PATH exists but is not executable"
    exit 1
fi

# Ensure output directories exist
mkdir -p ./include/graph/generated

# Generate flatbuffer files using built flatc with correct package
"$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b -t -c \
    --java-package-prefix org.nd4j \
    ./include/graph/scheme/node.fbs \
    ./include/graph/scheme/graph.fbs \
    ./include/graph/scheme/result.fbs \
    ./include/graph/scheme/request.fbs \
    ./include/graph/scheme/config.fbs \
    ./include/graph/scheme/array.fbs \
    ./include/graph/scheme/utils.fbs \
    ./include/graph/scheme/variable.fbs \
    ./include/graph/scheme/properties.fbs \
    ./include/graph/scheme/sequence.fbs

"$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b \
    --java-package-prefix org.nd4j \
    ./include/graph/scheme/node.fbs \
    ./include/graph/scheme/graph.fbs \
    ./include/graph/scheme/result.fbs \
    ./include/graph/scheme/request.fbs \
    ./include/graph/scheme/config.fbs \
    ./include/graph/scheme/array.fbs \
    ./include/graph/scheme/utils.fbs \
    ./include/graph/scheme/variable.fbs \
    ./include/graph/scheme/properties.fbs \
     ./include/graph/scheme/sequence.fbs

"$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b --grpc \
    --java-package-prefix org.nd4j \
    ./include/graph/scheme/uigraphstatic.fbs \
    ./include/graph/scheme/uigraphevents.fbs

# Generate TypeScript files
"$FLATC_PATH" -o ../nd4j/nd4j-web/nd4j-webjar/src/main/typescript -I ./include/graph/scheme --ts \
    --ts-flat-files \
    ./include/graph/scheme/node.fbs \
    ./include/graph/scheme/graph.fbs \
    ./include/graph/scheme/result.fbs \
    ./include/graph/scheme/request.fbs \
    ./include/graph/scheme/config.fbs \
    ./include/graph/scheme/array.fbs \
    ./include/graph/scheme/utils.fbs \
    ./include/graph/scheme/variable.fbs \
    ./include/graph/scheme/properties.fbs \
    ./include/graph/scheme/uigraphstatic.fbs \
    ./include/graph/scheme/uigraphevents.fbs

echo "Flatbuffer sources generated successfully"
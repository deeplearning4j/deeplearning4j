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

# Make sure to set $DL4J_DOCS_DIR to your local copy of https://github.com/deeplearning4j/deeplearning4j-docs
SOURCE_DIR=$(pwd)

# print the current git status
cd $DL4J_DOCS_DIR
git status

cd $SOURCE_DIR

# each release is its own jekyll collection located in docs/<version>
DOCS_DEST=$DL4J_DOCS_DIR/docs/_$DL4J_VERSION
mkdir $DOCS_DEST
echo Copying to $DOCS_DEST

# recursively find all files in doc_sources and copy
find $SOURCE_DIR/*/doc_sources -maxdepth 1 -type f -exec cp '{}' $DOCS_DEST \;
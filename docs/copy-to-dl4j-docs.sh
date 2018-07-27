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
cd $DL4J_DOCS_DIR
git checkout gh-pages

# each release is its own jekyll collection located in docs/<version>
mkdir $DL4J_DOCS_DIR/$DL4J_VERSION
DOCS_DEST='$DL4J_DOCS_DIR/$DL4J_VERSION'

# recursively find all files in doc_sources and copy
find $SOURCE_DIR/*/doc_sources -maxdepth 1 -type f -exec cp -t $DOCS_DEST '{}' +

cp -r $SOURCE_DIR/keras-import/doc_sources $DL4J_DOCS_DIR/keras
cp -r $DL4J_DOCS_DIR/assets $DL4J_DOCS_DIR/keras # workaround, assets are not picked up 2 levels down

cp -r $SOURCE_DIR/scalnet/doc_sources $DL4J_DOCS_DIR/scalnet
cp -r $DL4J_DOCS_DIR/assets $DL4J_DOCS_DIR/scalnet

cp -r $SOURCE_DIR/scalnet/doc_sources $DL4J_DOCS_DIR/samediff
cp -r $DL4J_DOCS_DIR/assets $DL4J_DOCS_DIR/samediff
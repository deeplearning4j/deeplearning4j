#!/usr/bin/env bash

# Make sure to set $DL4J_DOCS_DIR to your local copy of https://github.com/deeplearning4j/deeplearning4j-docs
SOURCE_DIR=$(pwd)
cd $DL4J_DOCS_DIR
git checkout gh-pages

cp -r $SOURCE_DIR/keras-import/doc_sources $DL4J_DOCS_DIR/keras
cp -r $DL4J_DOCS_DIR/assets $DL4J_DOCS_DIR/keras # workaround, assets are not picked up 2 levels down
cp -r $SOURCE_DIR/scalnet/doc_sources $DL4J_DOCS_DIR/scalnet
cp -r $DL4J_DOCS_DIR/assets $DL4J_DOCS_DIR/scalnet # workaround, assets are not picked up 2 levels down

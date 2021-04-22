#!/bin/bash
if ! [[ -z "$LIBND4J_URL" ]]; then
              mkdir libnd4j_home
              cd libnd4j_home
              wget "${LIBND4J_URL}" -o libnd4j.zip
              unzip libnd4j.zip
              export LIBND4J_HOME=${GITHUB_WORKSPACE}/libnd4j_home/libnd4j
              cd ..
fi
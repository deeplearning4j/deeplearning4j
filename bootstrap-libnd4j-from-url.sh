#!/bin/bash
if ! [[ -z "$LIBND4J_URL" ]]; then
              mkdir libnd4j_home
              cd libnd4j_home
              # Set a suffix for the downloaded libnd4j directory
              if [ -z "${LIBND4J_HOME_SUFFIX}" ]; then export LIBND4J_HOME_SUFFIX="cpu"; fi
              wget "${LIBND4J_URL}" -O libnd4j.zip
              unzip libnd4j.zip
              # Note: for builds, the whole source directory is uploaded, but a valid libnd4j home is actually only the compiled output
              export LIBND4J_HOME="${GITHUB_WORKSPACE}/libnd4j_home/libnd4j/blasbuild/${LIBND4J_HOME_SUFFIX}"
              echo "Set libnd4j home to ${LIBND4J_HOME}"
              cd ..
fi
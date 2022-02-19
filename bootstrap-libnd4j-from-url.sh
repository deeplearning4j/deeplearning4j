#!/bin/bash
if ! [[ -z "$LIBND4J_FILE_NAME" ]]; then
    echo "Downloading file with url at $LIBND4J_FILE_NAME"
    curl  "$LIBND4J_FILE_NAME" -o file_url.txt
    export LIBND4J_URL=`cat  file_url.txt`
    echo "Setup LIBND4J_URL to $LIBND4J_URL"
fi

if ! [[ -z "$LIBND4J_URL" ]]; then
              mkdir libnd4j_home
              cd libnd4j_home
              # Set a suffix for the downloaded libnd4j directory
              if [ -z "${LIBND4J_HOME_SUFFIX}" ]; then export LIBND4J_HOME_SUFFIX="cpu"; fi
              wget "${LIBND4J_URL}" -O libnd4j.zip
              unzip libnd4j.zip
              echo "Files in directory $(pwd) are "
              ls
              echo "Copying files to libnd4j directory"
              # Note: for builds, the whole source directory is uploaded, but a valid libnd4j home is actually only the compiled output
              cp -rf libnd4j/blasbuild/ "../libnd4j"
              echo "Files in ${GITHUB_WORKSPACE} are"
              ls "$GITHUB_WORKSPACE"
              echo "Files in libnd4j directory are ${GITHUB_WORKSPACE}/libnd4j"
              ls "${GITHUB_WORKSPACE}/libnd4j"
              cd ..
              # generated files may not exist, use in javacpp compilation
              mkdir -p ${GITHUB_WORKSPACE}/libnd4j/include/generated
              touch ${GITHUB_WORKSPACE}/libnd4j/include/generated/include_ops.h
              # Add flatbuffers include files to libnd4j directory
              git clone https://github.com/KonduitAI/flatbuffers.git
              mkdir -p ${GITHUB_WORKSPACE}/libnd4j/include/graph/generated/flatbuffers
              cp -rf flatbuffers/include/* ${GITHUB_WORKSPACE}/libnd4j/graph/generated/flatbuffers/
fi
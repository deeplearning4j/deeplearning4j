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
              curl "${LIBND4J_URL}" -O libnd4j.zip
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
fi
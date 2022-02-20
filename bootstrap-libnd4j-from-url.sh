#!/bin/bash

file_name=""



function append_args {
   if [ -n "$1"  ]; then
       file_name="${file_name}-$1"
   fi
}

function create_config {
    if [ "$1" == "cuda" ]; then
         engine="ENGINE_CUDA"
    elif [ "$1" == "aurora" ]; then
           engine="ENGINE_AURORA"
    else
            engine="ENGINE_CPU"

    fi

    config="${GITHUB_WORKSPACE}/libnd4j/blasbuild/$1/include/config.h"
    config_copy="${GITHUB_WORKSPACE}/libnd4j/include/config.h"
    if   test -f "${config_copy}"; then
        rm -f "${config_copy}"
    fi

    if   test -f "${config}"; then
      rm -f "${config}"
    fi

    print_config "${config_copy}" "${engine}"
    print_config "${config}" "${engine}"


}

function print_config {
  echo "Generating config.h $1"
  echo "#ifndef LIBND4J_CONFIG_H" >> "$1"
  echo "#define LIBND4J_CONFIG_H" >> "$1"
  echo "#define DEFAULT_ENGINE samediff::$2" >> "$1"
  echo "#endif" >> "$1"
  echo "Generated config.h at $1"
  cat "$1"
}


for var in "$@"
do
  append_args "${var}"
done

# Get rid of the first character
file_name_2="${file_name:1:${#file_name}}"


if ! [[ -z "$LIBND4J_FILE_NAME" ]]; then
    echo "Downloading file with url at $LIBND4J_FILE_NAME/${file_name_2}"
    curl  "${LIBND4J_FILE_NAME}/${file_name_2}" -o file_url.txt
    # shellcheck disable=SC2006
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
              mv flatbuffers flatbuffers-src
              cp -rf flatbuffers-src/include ${GITHUB_WORKSPACE}/libnd4j/
              echo "Copied flatbuffers to ${GITHUB_WORKSPACE}/libnd4j/include"
              if [ "$#" -gt 1 ]; then
                 if [ "$2" == "cuda" ]; then
                        create_config "cuda"
                    elif [ "$2" == "aurora" ]; then
                        create_config "aurora"
                    else
                       create_config "cpu"

                fi
              fi
fi
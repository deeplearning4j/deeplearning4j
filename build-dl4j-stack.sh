#!/bin/bash

# helper function that ensures cmd returns 0 exit code
function checkexit {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "error with $1" >&2
        exit 1
    fi
    return $status
}

# check incoming parameters
while [[ $# > 1 ]]
do
key="$1"
#Build type (release/debug), packaging type, chip: cpu,gpu,lib type (static/dynamic)
case $key in
    -b|--build-type)
    BUILD="$2"
    shift # past argument
    ;;
    -p|--packaging)
    PACKAGING="$2"
    shift # past argument
    ;;
    -c|--chip)
    CHIP="$2"
    shift # past argument
    ;;
    -cc|--compute)
    COMPUTE="$2"
    shift # past argument
    ;;
    -a|--march)
    NATIVE="$2"
    shift # past argument
    ;;
     -l|--libtype)
    LIBTYPE="$2"
    shift # past argument
    ;;
     --scalav)
    SCALAV="$2"
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

# default for chip
if [ -z "$CHIP" ]; then
 CHIP="cpu"
fi


# adjust scala versions
if [ "$SCALAV" == "2.10" ]; then
  SCALA="2.10.6"
fi
# adjust scala versions
if [ "$SCALAV" == "2.11" ]; then
  SCALA="2.11.7"
fi

cd ..

# removes any existing repositories to ensure a clean build
find . -maxdepth 1 -iname "libnd4j" -exec rm -rf "{}" \;
find . -maxdepth 1 -iname "nd4j" -exec rm -rf "{}" \;
find . -maxdepth 1 -iname "datavec" -exec rm -rf "{}" \;
#find . -maxdepth 1 -iname "deeplearning4j" -exec rm -rf "{}" \;

# Report argument values
echo BUILD      = "${BUILD}"
echo PACKAGING  = "${PACKAGING}"
echo CHIP       = "${CHIP}"
echo COMPUTE    = "${COMPUTE}"
echo NATIVE     = "${NATIVE}"
echo LIBTYPE    = "${LIBTYPE}"
echo SCALAV     = "${SCALAV}"

# compile libnd4j
checkexit git clone https://github.com/deeplearning4j/libnd4j.git
cd libnd4j
if [ -z "$NATIVE" ]; then
    checkexit bash buildnativeoperations.sh "$@" -a native
else
    checkexit bash buildnativeoperations.sh "$@"
fi

if [ "$CHIP" == "cuda" ]; then
    if [ -z "$COMPUTE" ]; then
        checkexit bash buildnativeoperations.sh -c cuda
    else
        checkexit bash buildnativeoperations.sh -c cuda --cc $COMPUTE
    fi
fi
export LIBND4J_HOME=`pwd`
cd ..

# build and install nd4j to maven locally
checkexit git clone https://github.com/deeplearning4j/nd4j.git
cd nd4j
if [ "$CHIP" == "cpu" ]; then
  checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'
else
  checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-tests'
fi
cd ..

# build and install datavec
checkexit git clone https://github.com/deeplearning4j/datavec.git
cd datavec
if [ "$SCALAV" == "" ]; then
  checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
else
  checkexit mvn clean install -DskipTests -Dmaven.javadoc.skip=true -Dscala.binary.version=$SCALAV -Dscala.version=$SCALA
fi
cd ..

# build and install deeplearning4j
#checkexit git clone https://github.com/deeplearning4j/deeplearning4j.git
cd deeplearning4j
if [ "$SCALAV" == "" ]; then
  if [ "$CHIP" == "cpu" ]; then
    checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0'
  else
    checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
  fi
else
  if [ "$CHIP" == "cpu" ]; then
    checkexit mvn clean install -DskipTests -Dmaven.javadoc.skip=true -Dscala.binary.version=$SCALAV -Dscala.version=$SCALA  -pl '!:deeplearning4j-cuda-8.0'
  else
    checkexit mvn clean install -DskipTests -Dmaven.javadoc.skip=true -Dscala.binary.version=$SCALAV -Dscala.version=$SCALA
  fi
fi
cd ..

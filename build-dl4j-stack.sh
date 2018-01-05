#!/bin/bash -e

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

function maybeUpdateRepo {
    if [ "$UPDATE_REPOS" == "true" ]; then
        # are there uncommited changes in the repo?
        if ! (git diff-index --quiet HEAD --) then
           echo "Some uncommited changes found in this repo! Stashing..."
           git stash
        fi
        git pull
    fi
}

# check incoming parameters
while [[ $# -gt 1 ]]
do
key="$1"
#Build type (release/debug), packaging type, chip: cpu,gpu,lib type (static/dynamic)
case $key in
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
    -s|--shallow)
    SHALLOW="YES"
    ;;
    -r|--repostrategy)
    REPO_STRATEGY="$2"
    shift # past argument
    ;;
    --deploy)
    DEPLOY="YES"
    ;;
    --testnd4j)
    TEST_ND4J="YES"
    ;;
    --testdatavec)
    TEST_DATAVEC="YES"
    ;;
    --testdl4j)
    TEST_DL4J="YES"
    ;;
    --skiplibnd4j)
    SKIP_LIBND4J="YES"
    ;;
    --skipnd4j)
    SKIP_ND4J="YES"
    ;;
    --skipdatavec)
    SKIP_DATAVEC="YES"
    ;;
    --skipdl4j)
    SKIP_DL4J="YES"
    ;;
    --mvnopts)
    MVN_OPTS="$2"
    shift
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

# default for chip
if [[ -z "$CHIP" ]]; then
    # test for cuda libraries
    if (type ldconfig &> /dev/null); then
       if (ldconfig -p | grep -q libcuda\.so); then
           CHIP="cuda"
       else
           CHIP="cpu"
        fi
    fi
fi

# adjust scala versions
if [ "$SCALAV" == "2.10" ]; then
  SCALA="2.10.6"
fi
# adjust scala versions
if [ "$SCALAV" == "2.11" ]; then
  SCALA="2.11.7"
fi

# set git cloning to a shallow depth if the option says so
if [[ -z "$SHALLOW" ]]; then
    GIT_CLONE="git clone"
else
    GIT_CLONE="git clone --depth 1"
fi

# set git cloning to a shallow depth if the option says so
if [[ -z "$DEPLOY" ]]; then
    MVN_GOAL="install"
else
    MVN_GOAL="deploy"
fi

# Report argument values
echo CHIP          = "${CHIP}"
echo COMPUTE       = "${COMPUTE}"
echo DEPLOY        = "${DEPLOY}"
echo NATIVE        = "${NATIVE}"
echo LIBTYPE       = "${LIBTYPE}"
echo SCALAV        = "${SCALAV}"
echo SHALLOW       = "${SHALLOW}"
echo REPO_STRATEGY = "${REPO_STRATEGY}"
echo TEST_ND4J     = "${TEST_ND4J}"
echo TEST_DATAVEC  = "${TEST_DATAVEC}"
echo TEST_DL4J     = "${TEST_DL4J}"
echo SKIP_LIBND4J  = "${SKIP_LIBND4J}"
echo SKIP_ND4J     = "${SKIP_ND4J}"
echo SKIP_DATAVEC  = "${SKIP_DATAVEC}"
echo SKIP_DL4J     = "${SKIP_DL4J}"
echo MVN_OPTS      = "${MVN_OPTS}"

###########################
# Script execution starts #
###########################

pushd ..

# removes lingering snapshot artifacts from existing maven cache to ensure a
# clean build
JAVA_PROJECTS="nd4j datavec deeplearning4j"
for dirName in $JAVA_PROJECTS; do
    if [ -d "$dirName" ]; then
        pushd "$dirName"
        mvn dependency:purge-local-repository -DreResolve=false -DactTransitively=false
        popd
    fi
done

# What to do with existing repos
case $REPO_STRATEGY in
    "delete")
        DELETE_REPOS="true"
        UPDATE_REPOS=""
        ;;
    "update")
        UPDATE_REPOS="true"
        DELETE_REPOS=""
        ;;
    *) # unknown : do nothing
        UPDATE_REPOS=""
        DELETE_REPOS=""
        ;;
esac

# removes any existing repositories to ensure a clean build
if [[ ! (-z "$DELETE_REPOS") ]]; then
    PROJECTS="libnd4j nd4j datavec deeplearning4j"
    for dirName in $PROJECTS; do
        find . -maxdepth 1 -iname "$dirName" -exec rm -rf "{}" \;
    done
fi

# compile libnd4j
if [[ -z "$SKIP_LIBND4J" ]]; then
    if [[ ! (-z "$DELETE_REPOS") || ! (-d libnd4j) ]]; then
        checkexit $GIT_CLONE https://github.com/deeplearning4j/libnd4j.git
    fi
    pushd libnd4j
    maybeUpdateRepo
    if [[ -z "$NATIVE" ]]; then
        checkexit bash buildnativeoperations.sh "$@" -a native
    else
        checkexit bash buildnativeoperations.sh "$@"
    fi

    if [ "$CHIP" == "cuda" ]; then
        if [[ -z "$COMPUTE" ]]; then
            checkexit bash buildnativeoperations.sh -c cuda
        else
            checkexit bash buildnativeoperations.sh -c cuda -cc "$COMPUTE"
        fi
    fi
    LIBND4J_HOME=$(pwd)
    export LIBND4J_HOME
    popd
fi

# build and install nd4j to maven locally
if [[ -z "$SKIP_ND4J" ]]; then
    if [[ ! (-z "$DELETE_REPOS") || ! (-d nd4j) ]]; then
        checkexit $GIT_CLONE https://github.com/deeplearning4j/nd4j.git
    fi
    if [[ -z "$TEST_ND4J" ]]; then
        ND4J_OPTIONS="-DskipTests"
    else
        ND4J_OPTIONS=""
    fi
    pushd nd4j
    maybeUpdateRepo
    if [ "$CHIP" == "cpu" ]; then
        checkexit bash buildmultiplescalaversions.sh clean $MVN_GOAL -Dmaven.javadoc.skip=true -pl '!nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!nd4j-backends/nd4j-tests' $ND4J_OPTIONS $MVN_OPTS
    else
        checkexit bash buildmultiplescalaversions.sh clean $MVN_GOAL -Dmaven.javadoc.skip=true $ND4J_OPTIONS $MVN_OPTS
    fi
    popd
fi

# build and install datavec
if [[ -z "$SKIP_DATAVEC" ]]; then
    if [[ ! (-z "$DELETE_REPOS") || ! (-d datavec) ]]; then
        checkexit $GIT_CLONE https://github.com/deeplearning4j/datavec.git
    fi
    if [[ -z "$TEST_DATAVEC" ]]; then
        DATAVEC_OPTIONS="-DskipTests"
    else
        if [ "$CHIP" == "cuda" ]; then
            DATAVEC_OPTIONS="-Ptest-nd4j-cuda-8.0"
        else
            DATAVEC_OPTIONS="-Ptest-nd4j-native"
        fi
    fi
    pushd datavec
    maybeUpdateRepo
    if [ "$SCALAV" == "" ]; then
        checkexit bash buildmultiplescalaversions.sh clean $MVN_GOAL -Dmaven.javadoc.skip=true $DATAVEC_OPTIONS $MVN_OPTS
    else
        checkexit mvn clean $MVN_GOAL -Dmaven.javadoc.skip=true -Dscala.binary.version="$SCALAV" -Dscala.version="$SCALA" $DATAVEC_OPTIONS $MVN_OPTS
    fi
    popd
fi

# build and install deeplearning4j
if [[ -z "$SKIP_DL4J" ]]; then
    if [[ ! (-z "$DELETE_REPOS") ||  ! (-d deeplearning4j) ]]; then
        checkexit $GIT_CLONE https://github.com/deeplearning4j/deeplearning4j.git
    fi
    if [[ -z "$TEST_DL4J" ]] ; then
        DL4J_OPTIONS="-DskipTests"
    else
        if [ "$CHIP" == "cuda" ]; then
            DL4J_OPTIONS="-Ptest-nd4j-cuda-8.0"
        else
            DL4J_OPTIONS="-Ptest-nd4j-native"
        fi
    fi
    pushd deeplearning4j
    maybeUpdateRepo
    if [ $DELETE_REPOS == "true" ]; then
        # reset the working diectory to the latest version of the tracking branch
        git remote update
        TRACKING_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')
        git reset --hard "$TRACKING_BRANCH"
    fi
    if [ "$SCALAV" == "" ]; then
        if [ "$CHIP" == "cpu" ]; then
            checkexit bash buildmultiplescalaversions.sh clean $MVN_GOAL -Dmaven.javadoc.skip=true -pl '!./deeplearning4j-cuda/' $DL4J_OPTIONS $MVN_OPTS
        else
            checkexit bash buildmultiplescalaversions.sh clean $MVN_GOAL -Dmaven.javadoc.skip=true $DL4J_OPTIONS $MVN_OPTS
        fi
    else
        checkexit mvn clean $MVN_GOAL -Dmaven.javadoc.skip=true -Dscala.binary.version="$SCALAV" -Dscala.version="$SCALA" $DL4J_OPTIONS $MVN_OPTS
    fi
    popd
fi

popd

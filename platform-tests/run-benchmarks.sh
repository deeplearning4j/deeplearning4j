#!/bin/bash
set -exo pipefail

JAVA_CALL="java"
TEST_RUNNER_PREFIX="valgrind"
# Find libjvm.so
if [[ -n $LIBJVM_SO ]]; then
    LIBJVM_PATH=$LIBJVM_SO
else
    JAVA_REAL_PATH=$(readlink -f $(which java))
    JAVA_HOME=$(dirname $(dirname $JAVA_REAL_PATH))
    LIBJVM_PATH=$(find $JAVA_HOME -type f -name "libjvm.so" | grep "/server/" | head -n 1)
fi

# If libjvm.so not found, terminate
if [[ -z $LIBJVM_PATH ]]; then
    echo "libjvm.so not found"
    exit 1
fi

# If TEST_RUNNER_PREFIX is not empty and contains "valgrind"
if [[ -n $TEST_RUNNER_PREFIX && $TEST_RUNNER_PREFIX =~ "valgrind" ]]; then
    # Create a file to store the suppression information
    SUPPRESSION_FILE="valgrind_suppressions.supp"

    # If suppression file exists, delete it
    if [[ -f $SUPPRESSION_FILE ]]; then
        rm -f $SUPPRESSION_FILE
    fi

    # Generate the suppression file for all memcheck error types except Param
    echo "Generating Valgrind suppression file at $SUPPRESSION_FILE..."
    for error_type in Addr1 Addr2 Addr4 Addr8 Value1 Value2 Value4 Value8 Jump Cond
    do
        cat << EOF >> $SUPPRESSION_FILE
{
    SuppressLibJvm${error_type}
    Memcheck:${error_type}
    ...
    obj:$LIBJVM_PATH
}
EOF
    done

    echo "Valgrind suppression file has been generated."

    # Check if "--suppressions" already exists in TEST_RUNNER_PREFIX
    if [[ $TEST_RUNNER_PREFIX != *"--suppressions"* ]]; then
        TEST_RUNNER_PREFIX="$TEST_RUNNER_PREFIX --suppressions=$SUPPRESSION_FILE --track-origins=yes --keep-stacktraces=alloc-and-free --error-limit=no"
    fi

    JAVA_CALL="${JAVA_CALL} -Djava.compiler=NONE"
fi


# Print the final command
echo "$TEST_RUNNER_PREFIX $JAVA_CALL $@"
export MALLOC_CHECK_=3
# Execute the command

$TEST_RUNNER_PREFIX $JAVA_CALL  -Djava.compiler=NONE  -cp /home/agibsonccc/Downloads/junit-platform-console-standalone-1.9.3.jar \
                                org.junit.platform.console.ConsoleLauncher  -cp=target/platform-tests-1.0.0-SNAPSHOT-shaded.jar \
                                  -c=org.eclipse.deeplearning4j.dl4jcore.gradientcheck.CNN1DGradientCheckTest

# If TEST_RUNNER_PREFIX is not empty and contains "valgrind", remove the suppression file
if [[ -n $TEST_RUNNER_PREFIX && $TEST_RUNNER_PREFIX =~ "valgrind" ]]; then
    rm -f $SUPPRESSION_FILE
fi



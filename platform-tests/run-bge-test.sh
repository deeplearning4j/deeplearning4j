#!/bin/bash
set -e

MVN="/home/agibsonccc/.m2/wrapper/dists/apache-maven-3.9.9-bin/4nf9hui3q3djbarqar9g711ggc/apache-maven-3.9.9/bin/mvn"
PROJECT_ROOT="/home/agibsonccc/Documents/GitHub/deeplearning4j"

echo "============================================"
echo "STEP 1: Building samediff-import modules"
echo "============================================"

cd "$PROJECT_ROOT"

$MVN clean install -pl nd4j/samediff-import/samediff-import-api,nd4j/samediff-import/samediff-import-onnx -DskipTests -am -q

if [ $? -eq 0 ]; then
    echo "BUILD SUCCESS"
else
    echo "BUILD FAILED"
    exit 1
fi

echo ""
echo "============================================"
echo "STEP 2: Running BGE import and test"
echo "============================================"

cd "$PROJECT_ROOT/platform-tests"

# Run the specific test with verbose output
$MVN test -Dtest=BgeModelLoadingTest#testBgeModelInference -Dnd4j.backend=nd4j-native -DforkCount=0 2>&1

echo ""
echo "============================================"
echo "TEST COMPLETE"
echo "============================================"

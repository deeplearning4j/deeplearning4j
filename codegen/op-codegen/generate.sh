#!/bin/bash

if test "$#" -eq 0; then
    echo "No namespaces were specified. One or more namespaces must be provided as an argument"
    echo "Usage example 1 (single namespace):      ./generate.sh math"
    echo "Usage example 2 (multiple namespaces):   ./generate.sh math,random"
    echo "Usage example 2 (all namespaces):        ./generate.sh all"
else
    mvn clean package -DskipTests
    java -cp target/codegen-1.0.0-SNAPSHOT-shaded.jar org.nd4j.codegen.cli.CLI -dir ../../ -namespaces "$@"
fi
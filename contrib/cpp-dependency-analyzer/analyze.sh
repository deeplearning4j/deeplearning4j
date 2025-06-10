#!/bin/bash
cd "$(dirname "$0")"
mvn -q clean compile exec:java -Dexec.mainClass="org.deeplearning4j.tools.CppDependencyAnalyzer" -Dexec.args="$*"

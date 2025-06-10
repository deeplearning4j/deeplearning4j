#!/bin/bash

# C++ Dependency Analyzer Runner Script
# Usage: ./run.sh [options] <root-directory>

# Build the project if JAR doesn't exist
JAR_FILE="target/cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar"

if [ ! -f "$JAR_FILE" ]; then
    echo "Building project..."
    mvn clean package -q
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
fi

# Run the analyzer
java -jar "$JAR_FILE" "$@"

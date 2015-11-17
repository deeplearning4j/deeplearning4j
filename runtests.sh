#!/usr/bin/env bash
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
mvn clean test

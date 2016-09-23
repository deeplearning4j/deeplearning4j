#! /bin/bash
set -eu
mvn -DscalaVersion=2.11 "$@"
mvn -DscalaVersion=2.10 "$@"

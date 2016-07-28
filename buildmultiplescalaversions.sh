#! /bin/bash
set -eu
mvn -Pscala-2.11 "$@"
mvn -Pscala-2.10 "$@"

#! /bin/bash
mvn -Dscala.binary.version=2.10 -Dscala.version=2.10.6 "$@"
mvn -Dscala.binary.version=2.11 -Dscala.version=2.11.7 "$@"

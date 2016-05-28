#! /bin/bash
mvn -Dscala.binary.version=2.10 "$@"
mvn -Dscala.binary.version=2.11 -pl "nd4j-serde/nd4j-kryo" "$@"
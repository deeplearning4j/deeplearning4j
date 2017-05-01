#! /bin/bash
BASEDIR=$(dirname $(readlink -f "$0"))

function whatchanged() {
    cd $BASEDIR
    for i in $(git status -s --porcelain -- $(find ./ -mindepth 2 -name pom.xml)|awk '{print $2}'); do
	echo $(dirname $i)
	cd $BASEDIR
    done
}

set -eu
./change-scala-versions.sh 2.11 # should be idempotent, this is the default
mvn "$@"
./change-spark-versions.sh 2
mvn -Dmaven.clean.skip=true -pl $(whatchanged| tr '\n' ',') -amd "$@"
./change-scala-versions.sh 2.10
./change-spark-versions.sh 1
mvn -Dmaven.clean.skip=true -pl $(whatchanged| tr '\n' ',') -amd "$@"
./change-scala-versions.sh 2.11 # back to the default

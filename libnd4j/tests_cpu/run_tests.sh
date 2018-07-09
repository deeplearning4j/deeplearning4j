#!/usr/bin/env bash

set -exo pipefail

IS_RELEASE='true'
OSARCH=$(arch)

if [[ "$OSTYPE" == "darwin"* ]]; then
    export CC=$(ls -1 /usr/local/bin/gcc-? | head -n 1)
    export CXX=$(ls -1 /usr/local/bin/g++-? | head -n 1)
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
    export CC=$(which gcc)
    export CXX=$(which g++)
fi

parse_commandline ()
{
	while test $# -gt 0
	do
		_key="$1"
		case "$_key" in
			--release=*)
				IS_RELEASE="${_key##--release=}"
				;;
			*)
				;;
		esac
		shift
	done
}

parse_commandline "$@"

if [[ "$IS_RELEASE" == "false" ]]; then
    ../buildnativeoperations.sh -t -b debug
else
    ../buildnativeoperations.sh -t -b release
fi

../blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_output="xml:../target/surefire-reports/TEST-results.xml"

#!/usr/bin/env bash

set -exuo pipefail

IS_RELEASE='true'
OSARCH=$(arch)

if [ -f /etc/redhat-release ]; then
    source /opt/rh/devtoolset-7/enable
fi

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

cmake -G "Unix Makefiles" -D_RELEASE=${IS_RELEASE} && make -j4

if [[ "$OSTYPE" == "linux-gnu" && "$OSARCH" == "x86_64" ]]; then
    # sudo is used as workaround for LeakSanitizer that requires root permissions
    sudo bash -c './layers_tests/runtests --gtest_output="xml:../target/surefire-reports/TEST-results.xml"'
    sudo chown -R "${USER:-jenkins}":"${USER:-jenkins}" ../target
else
    layers_tests/runtests --gtest_output="xml:../target/surefire-reports/TEST-results.xml"
fi

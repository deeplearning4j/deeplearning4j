#!/usr/bin/env bash
# Abort on Error
export PING_SLEEP=30s
export BUILD_OUTPUT=$(mktemp /tmp/nd4jXXX)

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 200 lines of output:
   tail -200 $BUILD_OUTPUT
   echo "Full log in $BUILD_OUTPUT"
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}
# If an error occurs, run our error handler to output a tail of the build
trap 'error_handler' ERR

# Set up a repeating loop to send some output to Travis.

bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!

# nicely terminate the ping output loop
trap "kill $PING_LOOP_PID" EXIT

# My build is using maven, but you could build anything with this, E.g.
mvn clean package -Dmaven.javadoc.skip=true -Dmaven.test.failure.ignore=true -Dmaven.test.error.ignore=true >> $BUILD_OUTPUT 2>&1

# The build finished so dump a tail of the output
dump_output

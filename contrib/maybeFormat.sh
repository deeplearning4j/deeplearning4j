#!/usr/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# swich to build root directory
pushd "$DIR/../"
# Only run in interactive mode, so as not to thwart automated tests
if [[ -t 0 || $1 == "fromMaven" ]]; then
    # test validation
    if !(mvn formatter:validate >/dev/null) then
       echo "Your working directory contains code that does not pass style checks."
       echo "Format it automatically? [y/n]"
       read -n 1 -t 8
       if [[ $REPLY =~ ^[Yy]$ ]]; then
           if ! (git diff-index --quiet HEAD --) then
              echo "Some uncommited changes found in this repo!"
              echo "Adding to git index before automated formatting"
              git add -A
           fi
           mvn formatter:format > /dev/null
      fi
   fi
else
    echo "Omitting syntax checks, as this terminal is not interactive"
fi
popd

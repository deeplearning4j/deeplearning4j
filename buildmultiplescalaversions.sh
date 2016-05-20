#! /bin/sh
for v in 2.10 2.11 ; do
  mvn -Dscala.binary.version=$v $*
done
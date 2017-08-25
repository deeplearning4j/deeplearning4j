#!/bin/bash

DIRS=""
FILES=""

for i in `find tensorflow/core/framework | grep \.proto$`
 do
  FILES+=" ${i}"
 done

echo $FILES

protoc --proto_path=./tensorflow/core/framework --cpp_out=protobuf-generated/ --java_out=protobuf-java/ $FILES
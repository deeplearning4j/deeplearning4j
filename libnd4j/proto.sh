#!/bin/bash

DIRS=""
FILES=""

for i in `find tensorflow/core/framework | grep \.proto$`
 do
  FILES+=" ${i}"
 done

echo $FILES

protoc --proto_path=/Users/raver119/develop/tensorflow --cpp_out=cpp_output/ --java_out=java_output/ $FILES
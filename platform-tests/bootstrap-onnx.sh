#!/bin/bash
echo 'Downloading onnx models...'
mvn clean compile
mvn exec:java -Dexec.mainClass="org.eclipse.deeplearning4j.testinit.DownloadKt" -Dexec.args="./src/main/resources/onnx-models.txt"
echo 'Done downloading onnx models'
for file in `cat src/main/resources/onnx-models.txt`; do
   #echo "$file"
   mvn exec:java -Dexec.mainClass="org.eclipse.deeplearning4j.testinit.OnnxConvertKt" -Dexec.args="$file"
done
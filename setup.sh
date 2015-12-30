#!/bin/sh
cd ..
if [ ! -f nd4j ]; then
    git clone https://github.com/deeplearning4j/nd4j.git
fi

if [ ! -f Canova ]; then
    git clone https://github.com/deeplearning4j/Canova.git
fi
cd nd4j
git pull
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
cd ..
cd Canova
git pull
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
cd ..
cd deeplearning4j
git pull
mvn clean install -DskipTests -Dmaven.javadoc.skip=true

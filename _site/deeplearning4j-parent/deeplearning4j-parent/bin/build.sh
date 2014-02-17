#!/bin/bash
cd deeplearning4j-scaleout/deeplearning4j-scaleout-akka
mvn assembly:single
mv target/*.bz2 ../..
cd ../..

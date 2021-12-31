#!/bin/bash
mvn  clean -Dtest.nogc=true '-Dtest=org.nd4j.samediff.frameworkimport.onnx.**' test

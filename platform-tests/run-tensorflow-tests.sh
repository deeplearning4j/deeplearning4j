#!/bin/bash
mvn  clean -test.nogc=true '-Dtest=org.nd4j.samediff.frameworkimport.tensorflow.**' test

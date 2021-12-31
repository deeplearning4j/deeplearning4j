#!/bin/bash
mvn  clean -test.nogc=true '-Dtest=org.deeplearning4j.nn.modelimport.keras.**' test

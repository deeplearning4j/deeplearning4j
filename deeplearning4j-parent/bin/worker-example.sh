#!/bin/bash
java -Xmx6g -Xms6g -cp "lib/*"  com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ActorNetworkRunnerApp -a dbn -t worker -h training -data mnist 
~          

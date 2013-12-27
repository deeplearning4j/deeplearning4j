#!/bin/bash
java -Xmx6g -Xms6g -cp "lib/*"  -Dakka.remote.netty.tcp.hostname=training com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ActorNetworkRunnerApp -a dbn -data mnist -h training 

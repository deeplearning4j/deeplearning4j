#!/bin/bash

if [ -z "$1" ]
  then
    echo "Please specify a host"
fi
MAX_HEAP="5g"
MIN_HEAP="5g"



java -cp "lib/*" -server -XX:+UseTLAB   -Xmx$MAX_HEAP -Xms$MIN_HEAP -Dakka.remote.netty.tcp.hostname=train -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0   -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.example.mnist.MnistExampleMultiThreaded

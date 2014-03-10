#!/bin/bash

if [ -z "$1" ]
  then
    echo "Please specify a host"
    exit 1
fi

MAX_HEAP="5g"
MIN_HEAP="5g"


java -cp "lib/*"   -Xmx$MAX_HEAP -Xms$MIN_HEAP -server -XX:+UseTLAB   -Dakka.remote.netty.tcp.hostname=train2 -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunnerApp  -h $1 -t worker

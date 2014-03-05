---
title: 
layout: default
---


Running deeplearning4j in distributed mode is accessible with the following.


       java -cp "lib/*"   -Xmx5g -Xms5g -server -XX:+UseTLAB   -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunnerApp -h train1 -t worker


The key here is the -h and -t command line parameters. The -h points at a zookeeper node where the configuration is stored and the -t specifies a worker node.

Service discovery happens when deeplearning4j stores the configuration upon startup. 
[../doc/deeplearning4j/iterativereduce/actor/multilayer/ActorNetworkRunner.html](ActorNetworkRunner)

runs and starts both a local worker node  and a master which stores the configuration specified in the master.

The worker then picks this up from zookeeper and akka actors on the worker will automatically join the cluster

thanks to akka's gossip protocol.
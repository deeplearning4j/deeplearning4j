---
title: Setting up a Deeplearning4j Cluster
layout: default
---

# Setting up a Deeplearning4j Cluster

*Assumptions: AWS experience*

For the moment, a DeepLearning4J cluster consists of one master node set up manually. This will be changed in future releases, when vectorization and dataset creation are a little more refined. 

Before you set up a master node, you need a [deeplearning4j distribution](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/).

If you don't have one, download one of the zip, bz2 or tar files from the page linked to above.

Next, let's tackle setup: We will be using the deeplearning4j-examples demonstration classes as the master nodes.

###master node

A master node coordinates the running of all of the work, and has the dataset iterator, which in turn has a handle on the dataset. This is the seed node for the Akka cluster. We will be setting this up using AWS.

So first, create a node on AWS. We can then create a cluster based on this node.

Setting up workers will be an automated process after the baseline master is setup.

Master setup:

	ssh -i /path/to/your/key ec2-user@ec2-*.compute-1.amazonaws.com

After you're connected, download a DL4J distribution either via Wget and one of the links above or via SFTP.

Untar the DL4J distribution and run:

	rm -f deeplearning4j-*.tar.gz
	mv deeplearning4j-*/lib .

Each dl4j distribution currently depends on Blas, so run:

	sudo yum -y install blas

The, when setting up the master node, ensure that you set the hostname via:

	sudo hostname publicawshostname

Now we are going to run one of the examples.

Use this to create the workers:

	#!/bin/bash
	sudo yum -y update
	sudo yum -y install blas
	wget https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/deeplearning4j-examples-0.0.3.2-20140515.171028-2-bin.tar.gz
	tar xvf *.tar.gz
	rm -f *.tar.gz
	mv deeplearning4j-*/lib .
	nohup java -cp "lib/*"   -Dhazelcast.aws=true -Dhazelcast.access-key=YOURKEY -Dhazelcast.access-secret=YOURSECRET -Xmx5g -Xms5g -server -XX:+UseTLAB   -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunnerApp  -h ec2-*.compute-1.amazonaws.com  -t worker 

Use this to run the workers:

	#!/bin/bash
	java -cp "lib/*"  -Dakka.remote.netty.tcp.hostname=ec2-*.com -Dhazelcast.aws=true -Dhazelcast.access-keyYOURKEY -Dhazelcast.access-secret=YOURSECRET  -Dorg.deeplearning4j.aws.accessKey=YOURKEY -Dorg.deeplearning4j.aws.accessSecretYOURSECERT  -Xms4g -Xmx4g -server -XX:+UseTLAB   -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.example.deepautoencoder.MnistExampleMultiThreaded

This will launch any number of worker nodes.

Here is an example cluster deployment setup:

	java -cp "lib/*" -Dhazelcast.access-key=YOURKEY -Dhazelcast.access-secret=YOURSECRET  -Dorg.deeplearning4j.aws.accessKey=YOURKEY -Dorg.deeplearning4j.aws.accessSecret=YOURSECRET org.deeplearning4j.aws.ec2.provision.ClusterSetup -w  NUMBEROFNODES -sg YOURSECURITYGROUP-kp deploy -kpath /path/to/your/key -wscript /path/to/createworkerscript

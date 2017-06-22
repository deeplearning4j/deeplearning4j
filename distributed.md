---
title: Distributed training: gradients sharing
layout: default
---

# Distributed training: gradients sharing

As of release 0.9.0 (or the 0.8.1-SNAPSHOT), DeepLearning4j also support distributed training in Apache Spark environment, using Aeron for messages.

Idea is simple: individual workers are processing DataSets, and before gradients 
 
 ***NICE SCHEME NEEDED HERE***


# Effective scalability

Network transfers has their price, and this algorithm does some IO as well. Additional overhead comes as `updates encoding time + message serialization time + updates application from other workers`.
So, the longer original iteration time is, less your relative impact coming from sharing is.

Here's simple form, that'll help you with scalability expectations:

# Setting up your cluster

Basically all you need to run training is Spark 1.x/2.x cluster, and at least one open UDP port (both inbound/outbound)

Example configuration:
```
VoidConfiguration voidConfiguration = VoidConfiguration.builder()
            .unicastPort(40123)
            .build();

TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration,batchSizePerWorker)
            .updatesThreshold(1e-3)
            .rddTrainingApproach(RDDTrainingApproach.Exported)
            .batchSizePerWorker(batchSizePerWorker)
            .workersPerNode(4)
            .build();

//Create the Spark network
SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

//Execute training:
for (int i = 0; i < numEpochs; i++) {
    sparkNet.fit(trainData);
    log.info("Completed Epoch {}", i);
}
```
**_PLEASE NOTE_**: this configuration assumes that you have UDP port 40123 open on ALL nodes within your cluster.



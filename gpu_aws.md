---
title: Launching GPUs on AWS With Spark
layout: default
---

# Launching GPUs on AWS With Spark

Scaling out over many cores is the only way to train deep neural networks in a reasonable amount of time. Deeplearning4j can parallelize the massive linear algebra operations necessary for processing large and high-dimensional datasets on multiple GPUs. Indeed, it scales to use an arbitrary number of GPUs, as many as you have available. 

Short of building their own GPU rack, most DL4J users will run neural nets over multiple GPUs via Amazon Web Services, and AWS allows users to access up to 1,536 CUDA cores per card. The [largest AWS instance for GPUs, G2, has four cards](https://aws.amazon.com/ec2/instance-types/), so about 6,000 cores per box. While owning your own chips allows for greater optimization, GPUs on AWS gives a taste of their processing speed. 

[AWS provides instructions for launching GPU instances for a variety of NVIDIA drivers](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html).

Once you have the GPUs installed, you're ready to run DL4J on top of them from Spark. 

Here's an example of how you configure a [deep-belief network to learn MNIST using Spark and GPUs](https://github.com/deeplearning4j/spark-gpu-examples/blob/master/src/main/java/org/deeplearning4j/SparkGpuExample.java). 

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/spark-gpu-examples/blob/master/src/main/java/org/deeplearning4j/SparkGpuExample.java?slice=38:69"></script>

The POM.xml file of your project should [look like this](https://github.com/deeplearning4j/spark-gpu-examples/blob/master/pom.xml). Note the Spark and Jcublas dependencies in the POM. 

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/spark-gpu-examples/blob/master/pom.xml?slice=136:148"></script>

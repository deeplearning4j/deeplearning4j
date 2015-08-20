---
title: 
layout: default
---

# Install Spark

First, see whether you have Spark installed by entering `which spark` in the command line. 

## OSX 

If you don't have it, you can easily install Spark on OSX with `brew`:

        brew install apache-spark

* For other OS, see the instructions on [this page](https://spark.apache.org/downloads.html). We recommend a pre-built download rather than building from source. Below, we'll be working with  Hadoop 2.4 and Spark 1.4.1, but you may have slightly different versions. (Spark dataframes require 1.4.*)

![Alt text](../img/spark_download.png)

* You'll need to set the environmental variable SPARK_HOME. To figure out what the file path should be, search for the spark command you'll need later, `spark-submit`:

        sudo find / -name "spark-submit"

* Take the results (here's what mine look like)

        /Users/cvn/Desktop/spark-1.4.1-bin-hadoop2.4/bin/spark-submit

* Remove `/bin/spark-submit` and feed the rest of the file path into your environment variable SPARK_HOME like so:

        export SPARK_HOME=/users/cvn/Desktop/spark-1.4.1-bin-hadoop2.4

## Linux

On a Linux OS, you'll want to follow these steps:

        wget http://apache.mirrors.lucidnetworks.net/spark/spark-1.4.1/spark-1.4.1-bin-hadoop2.4.tgz
        tar -zxf spark-1.4.1-bin-hadoop2.4.tgz
        cd spark-1.4.1-bin-hadoop2.4
        ./bin/spark-shell //<this is just to test that Spark is installed

Once you have Spark running, you can [build the DL4J examples](../spark.html).

---
title: Install Spark
layout: default
---

# Install Spark

First, see whether you have Spark installed by entering `which spark` in the command line. 

## OSX 

If you don't have it, you can easily install Spark on OSX with [Homebrew](http://brew.sh/), a package manager for OSX.

First install Homebrew, if you don't have it:

        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Spark with `brew`:

        brew install apache-spark

## Linux

* For other OS, see the instructions on [this page](https://spark.apache.org/downloads.html). We recommend a pre-built download rather than building from source. Below, we'll be working with  Hadoop 2.4 and Spark 1.4.1, but you may have slightly different versions. (Spark dataframes require 1.4.*)

![Alt text](../img/spark_download.png)

On Linux OS, you can follow steps similar to these (the URL will depend on the Spark and Hadoop versions you choose.):

        wget http://apache.mirrors.lucidnetworks.net/spark/spark-1.4.1/spark-1.4.1-bin-hadoop2.4.tgz
        tar -zxf spark-1.4.1-bin-hadoop2.4.tgz
        cd spark-1.4.1-bin-hadoop2.4
        ./bin/spark-shell //<this is just to test that Spark is installed

<!-- * You'll need to set the environmental variable SPARK_HOME. To figure out what the file path should be, search for the spark command you'll need later, `spark-submit`:

        sudo find / -name "spark-submit"

* Take the results (here's what mine look like)

        /Users/cvn/Desktop/spark-1.4.1-bin-hadoop2.4/bin/spark-submit

* Remove `/bin/spark-submit` and feed the rest of the file path into your environment variable SPARK_HOME like so:

        export SPARK_HOME=/users/cvn/Desktop/spark-1.4.1-bin-hadoop2.4

-->

* Once you have Spark running, you can [build the DL4J examples](../spark.html).

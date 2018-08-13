---
title: Quickstart Guide for DL4J
short_title: Quickstart
description: Get started fast with Eclipse Deeplearning4j.
category: Get Started
weight: 0
---

## Welcome to DL4J

Welcome to Eclipse Deeplearning4j, a deep learning framework supported the Eclipse Foundation! This is everything you need to run Deeplearning4j, our examples, and our tutorials. In this quickstart guide you will create your first deep neural network with Deeplearning4j and learn how to train, infer, and evaluate it.

Are you a regular user who isn't familiar with the Java Virtual Environment? Help us guide you to the correct quickstart:

<div class="btn-group" role="group">
  <button type="button" class="btn btn-default">Yes (keep reading <i class="arrow-down"></i>)</button>
  <button type="button" class="btn btn-default">No, I'm JVM expert</button>
</div>

If you're having difficulty, we recommend that you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j). Gitter is where you can request help and give feedback, but please do use this guide before asking questions we've answered below. If you are new to deep learning, we've included [a road map for beginners](./deeplearningforbeginners.html) with links to courses, readings and other resources. If you need an end to end tutorial to get started (including setup), then please go to our [getting started](http://deeplearning4j.org/gettingstarted).


### Prepare your environment

For this quickstart tutorial, you will learn how to set up an environment that will allow you create and execute code in an interactive environment. There are a couple of open source tools that can help you do this, and these instructions will help you set them up step-by-step.

Notebooks are interactive coding environments that can be easily installed on a computer and exported and shared with others. This quickstart will use an open-source notebook tool known as [Apache Zeppelin](https://zeppelin.apache.org/). This tutorial and others like it are already prepackaged on a community edition of [SKIL](https://skymind.ai/platform) from Skymind (the creators of Deeplearning4j), and we will be using a tool known as [Docker](https://docs.docker.com/install/) to download and run the tutorial.

#### Download Docker CE
If you don't already have Docker on your system, visit the [Docker install page](ttps://docs.docker.com/install/) and install the version of Docker that works best for you. You can test if you have Docker installed by opening a terminal and running `docker --version`.

#### Pull the SKIL-CE image
With a single command you can download the SKIL-CE image that has this tutorial (and other deep learning guides) directly to your machine. Use the `docker pull` command to copy an image from Docker's website to your own machine. To pull the SKIL-CE image, run the following command:

```shell
docker pull skymindops/skil-ce:1.1.0
```

<div class="alert alert-info" role="alert">
  Alternatively, if you already have Apache Zeppelin you can import our notebooks directly from [our tutorials repository](https://github.com/deeplearning4j/deeplearning4j/blob/master/dl4j-examples/tutorials/).
</div>

#### Run the SKIL image</h5>
Now you're ready to run the SKIL image and open Zeppelin and the quickstart notebook. Run the following command and you'll see the image "boot up":

```shell
docker run -it --rm --name skil -p 9008:9008 -p 8080:8080 skymindops/skil-ce:1.1.0 bash /start-skil.sh
```


### Running the Quickstart tutorial

Once you have finished installing the prerequisite software and have used the "docker run" command above, open Zeppelin by visiting `http://localhost:8080` in your web browser. You will need to login using the username `admin` and the password `admin`. Click the login button at the top right of the web page to enter your credentials.

### Deeplearning4j and EMNIST

Success! Now you should be logged in and be able to see all of the Deeplearning4j tutorials. Open the <b>Deeplearning4j Quickstart</b> and you will see a screen that looks like this:

<img src="" alt="Deeplearning4j EMNIST MNIST Deep learning tutorial">



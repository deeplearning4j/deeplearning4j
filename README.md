Deeplearning4J: Neural Net Platform
=========================
 
[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.deeplearning4j/deeplearning4j-core/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.deeplearning4j/deeplearning4j-core)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.deeplearning4j/deeplearning4j-core/badge.svg)](http://deeplearning4j.org/doc)

Deeplearning4J is an Apache 2.0-licensed, open-source, distributed neural net library written in Java and Scala.

Deeplearning4J integrates with Hadoop and Spark and runs on several backends that enable use of CPUs and GPUs. The aim is to create a plug-and-play solution that is more convention than configuration, and which allows for fast prototyping. 

The most recent stable release in Maven Central is `0.7.2`, and the current master is `0.7.3-SNAPSHOT`.

---
## Using Deeplearning4j

To get started using Deeplearning4j, please go to our [Quickstart](http://deeplearning4j.org/quickstart.html). You'll need to be familiar with a Java automated build tool such as Maven and an IDE such as IntelliJ. 

## Main Features
- Versatile n-dimensional array class
- GPU integration(Supports devices starting from Kepler,cc3.0. You can check your device's compute compatibility [here](https://developer.nvidia.com/cuda-gpus).)


---
## Modules
- datavec = Library for converting images, text and CSV data into format suitable for Deep Learning
- nn = core neural net structures MultiLayer Network and Computation graph for designing Neural Net structures
- core = additional functionality building on deeplearning4j-nn
- modelimport = functionality to import models from Keras
- nlp = natural language processing components including vectorizers, models, sample datasets and renderers
- scaleout = integrations
    - spark = integration with Apache Spark versions 1.3 to 1.6 (Spark 2.0 coming soon)
    - parallel-wraper = Single machine model parallelism (for multi-GPU systems, etc) 
    - aws = loading data to and from aws resources EC2 and S3
- ui = provides visual interfaces for tuning models [Details here](https://deeplearning4j.org/visualization)

---
## Documentation
Documentation is available at [deeplearning4j.org](https://deeplearning4j.org/overview) and [JavaDocs](http://deeplearning4j.org/doc).

## Support

We are not supporting Stackoverflow right now. Github issues should focus on bug reports and feature requests. Please join the community on [Gitter](https://gitter.im/deepelearning4j/deeplearning4j), where we field questions about how to install the software and work with neural nets. For support from Skymind, please see our [contact page](https://skymind.io/contact). 

## Installation

To install Deeplearning4J, there are a couple approaches briefly described on our [Quickstart](http://deeplearning4j.org/quickstart.html) and below. More information can be found on the [ND4J web site](http://nd4j.org/getstarted.html) as well as [here](http://deeplearning4j.org/gettingstarted.html).

#### Use Maven Central Repository

Search Maven Central for [deeplearning4j](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j) to get a list of dependencies.

Add the dependency information to your `pom.xml` file. **We highly recommend downloading via Maven unless you plan to help us develop DL4J.**

<!--
#### Yum Install / Load RPM (Fedora or CentOS)
Create a yum repo and run yum install to load the Red Hat Package Management (RPM) files. First create the repo file to setup the configuration locally.

    $ sudo vi /etc/yum.repos.d/dl4j.repo 

Add the following to the `dl4j.repo` file:

    [dl4j.repo]

    name=dl4j-repo
    baseurl=http://ec2-52-5-255-24.compute-1.amazonaws.com/repo/RPMS
    enabled=1
    gpgcheck=0

Then run the following command on the dl4j repo packages to install them on your machine:

    $ sudo yum install [package name] -y
    $ sudo yum install DL4J-Distro -y 

Note, be sure to install the ND4J modules you need first, especially the backend and then install DataVec and DL4J.

-->
---
## Contribute

1. Check for [open issues](https://github.com/deeplearning4j/deeplearning4j/issues) or open a fresh one to start a discussion around a feature idea or a bug. 
2. If you feel uncomfortable or uncertain about an issue or your changes, don't hesitate to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/deeplearning4j/deeplearning4j.git) on GitHub to start making your changes to the **master** branch (or branch off of it).
4. Write a test that shows the bug was fixed or the feature works as expected.
5. Send a pull request and bug us on Gitter until it gets merged and published. :)

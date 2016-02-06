Deeplearning4J: Neural Net Platform
=========================
 
[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Deeplearning4J is an Apache 2.0-licensed, open-source, distributed neural net library written in Java and Scala.

Deeplearning4J integrates with Hadoop and Spark and runs on several backends that enable use of CPUs and GPUs. The aim is to create a plug-and-play solution that is more convention than configuration, and which allows for fast prototyping. 

Current version in maven central is `0.4-rc3.8`.

---
## Using Deeplearning4j

To get started using Deeplearning4j, please go to our [Quickstart](http://deeplearning4j.org/quickstart.html). You'll need to be familiar with a Java automated build tool such as Maven and an IDE such as IntelliJ. 

## Main Features
- Versatile n-dimensional array class
- GPU integration
- Scalable on Hadoop, Spark and Akka + AWS et al

---
## Modules
- cli = command line interface for deeplearning4j
- core = core neural net structures and supporting components such as datasets, iterators, clustering algorithms, optimization methods, evaluation tools and plots.
- scaleout = integrations
    - aws = loading data to and from aws resources EC2 and S3
    - nlp = natural language processing components including vectorizers, models, sample datasets and renderers
    - akka = setup concurrent and distributed applications on the JVM
    - api = core components like workers and multi-threading
    - zookeeper = maintain configuration for distributed systems
    - hadoop-yarn = common map-reduce distributed system
    - spark = integration with spark
        - dl4j-spark = spark 1.2-compatible
        - dl4j-spark-ml = spark 1.4-compatible, based on ML pipeline
- ui = provides visual interfaces with models like nearest neighbors
- test-resources = datasets and supporting components for tests

---
## Documentation
Documentation is available at [deeplearning4j.org](http://deeplearning4j.org/) and [JavaDocs](http://deeplearning4j.org/doc/).

## Support

We are not supporting Stackoverflow right now. Github issues should be limited to bug reports. Please join the community on [Gitter](https:://gitter.im/deepelearning4j/deeplearning4j), where we field questions about how to install the software and work with neural nets. 

## Installation
To install Deeplearning4J, there are a couple approaches (briefly described on our [Quickstart](http://deeplearning4j.org/quickstart.html) and below). More information can be found on the [ND4J website](http://nd4j.org/getstarted.html) and [here](http://deeplearning4j.org/gettingstarted.html).

#### Use Maven Central Repository

Search for [deeplearning4j](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j) to get a list of jars you can use.

Add the dependency information into your `pom.xml`. **We highly recommend downloading via Maven unless you plan to help us develop DL4J.**

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

Note, be sure to install the nd4j modules you need first, especially the backend and then install Canova and dl4j.

---
## Contribute

1. Check for open issues or open a fresh one to start a discussion around a feature idea or a bug. 
2. If you feel uncomfortable or uncertain about an issue or your changes, don't hesitate to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/deeplearning4j/deeplearning4j.git) on GitHub to start making your changes to the **master** branch (or branch off of it).
4. Write a test which shows that the bug was fixed or that the feature works as expected.
5. Send a pull request and bug us on Gitter until it gets merged and published. :)

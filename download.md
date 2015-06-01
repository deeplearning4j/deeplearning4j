---
title: "Downloads"
layout: default
---

# Downloads

To install Deeplearning4J, there are a couple approaches (briefly described below). More information can be found on the  [ND4J website](http://nd4j.org/getstarted.html).

#### Use Maven Central Repository

1. Search for [deeplearning4j](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j) to get a list of jars you can use

2. Add the dependency information into your pom.xml

#### Clone from the GitHub Repo
Deeplearning4J is being actively developed and you can clone the repository, compile it and reference it in your project.

Clone the repository:

    $ git clone git://github.com/deeplearning4j/deeplearning4j.git

Compile the project:

    $ cd deeplearning4j && mvn clean install -DskipTests -Dmaven.javadoc.skip=true

Add the local compiled file dependencies to your pom.xml file. Here's an example of what they'll look like:

    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-cli</artifactId>
        <version>0.0.3.3.4.alpha1-SNAPSHOT</version>
    </dependency>

#### Yum Install / Load RPM (Fedora or CentOS)
Create a yum repo and run yum install to load the Red Hat Package Management (RPM) files. First create the repo file to setup the configuration locally.

    $ sudo vi /etc/yum.repos.d/dl4j.repo 

Add the following to the dl4j.repo file:



    [dl4j.repo]

    name=dl4j-repo
    baseurl=http://ec2-52-5-255-24.compute-1.amazonaws.com/repo/RPMS
    enabled=1
    gpgcheck=0


Then run the following command on the dl4j repo packages to install them on your machine:

    $ sudo yum install [package name] -y
    $ sudo yum install DL4J-Distro -y 

Note, be sure to install the nd4j modules you need first, especially the backend and then install Canova and dl4j.

ND4J: Scientific Computing on the JVM
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

ND4J is an Apache2 Licensed open-sourced scientific computing library for the JVM. It is meant to be used in production environments
rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements.

Please search for the latest version on search.maven.org.

Or use the versions displayed in:
https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml


---
## Main Features

- Versatile n-dimensional array object
- Multiplatform functionality including GPUs
- Linear algebra and signal processing functions

Specifics

- Supports GPUs via CUDA and Native via Jblas and Netlib Blas.
- All of this is wrapped in a unifying interface.
- The API mimics the semantics of Numpy, Matlab and scikit-learn.

---
## Modules
Several of these modules are different backend options for ND4J (including GPUs with JCublas).

- api = core
- instrumentation
- java = java backend
- jblas = jblas backend
- jcublas-parent = jcublas backend (GPUs)
- jdbc = Java Database Connectivity
- jocl-parent = Java bindings for OpenCL
- netlib-blas = netlib blas backend
- scala-api = API for Scala users
- scala-notebook = Integration with Scala Notebook

---
## Documentation

Documentation is available at [nd4j.org](http://nd4j.org/). Access the [JavaDocs](http://nd4j.org/doc/) for more detail.

---
## Installation

To install ND4J, there are a couple of approaches, and more information can be found on the [ND4J website](http://nd4j.org/getstarted.html).

#### Install from Maven Central

1. Search for nd4j in the [Maven Central Repository](http://mvnrepository.com/search?q=nd4j) to find the available nd4j jars.
2. Include the appropriate dependency in your pom.xml.

#### Clone from the GitHub Repo

ND4J is actively developed. You can clone the repository, compile it, and reference it in your project.

Clone the repository:

    $ git clone https://github.com/deeplearning4j/nd4j.git

Compile the project:

    $ cd nd4j
    $ mvn clean install -DskipTests -Dmaven.javadoc.skip=true

Add the local compiled file dependency (choose the module for your backend) to your pom.xml file:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-x86</artifactId>
        <version>${nd4j.version}</version>
    </dependency>

#### Yum Install / Load RPM (Fedora or CentOS)
Create a yum repo and run yum install to load the Red Hat Package Management (RPM) files. First create the repo file to setup the configuration locally.

    $ sudo vi /etc/yum.repos.d/dl4j.repo 

Add the following to the dl4j.repo file:

'''

    [dl4j.repo]

    name=dl4j-repo
    baseurl=http://ec2-52-5-255-24.compute-1.amazonaws.com/repo/RPMS
    enabled=1
    gpgcheck=0
'''

Then run the following command on the dl4j repo packages to install them on your machine:

    $ sudo yum install [package name] -y
    $ sudo yum install nd4j-cli -y # for example

Note, be sure to install the nd4j modules you need first, especially the backend and then install Canova and dl4j.

---
## Tests

Run the following command to execute all tests at once.

    mvn test

Or, run the following command to execute TestSuite with only specified backend e.g. jcublas on GPU.

    mvn test -pl nd4j-XXX

- nd4j-java
- nd4j-jblas
- nd4j-jcublas-parent/nd4j-jcublas-X.X
- nd4j-netlib-blas
- nd4j-x86

Or, run the following command to execute only specified tests in TestSuite with only specified backend.

     mvn test -pl nd4j-XXX -Dorg.nd4j.linalg.tests.classestorun=org.nd4j.linalg.YYY -Dorg.nd4j.linalg.tests.methods=ZZZ

---
## Contribute

1. Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/deeplearning4j/nd4j.git) on GitHub to start making your changes to the **master** branch (or branch off of it).
4. Write a test, which shows that the bug was fixed or that the feature works as expected.
5. Send a pull request, and bug us on Gitter until it gets merged and published. 

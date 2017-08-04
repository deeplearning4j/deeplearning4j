<<<<<<< HEAD
# WEBSITE GUIDELINE

## PERMALINK

**DEFAULT ENGLISH** = deeplearning4j.org/**title**

**OTHER LANGUAGE** = deeplearning4j.org/*lang*/**title**

**EXAMPLE:**

1. English  = deeplearning4j.org/**title**
2. Chinese  = deeplearning4j.org/*cn*/**title**
3. Korean   = deeplearning4j.org/*kr*/**title**
4. Japanese = deeplearning4j.org/*jp*/**title**

---

## WEBSITE STRUCTURE

The deeplearning4j.org website divided into the following sections:

1. Landing Page (deeplearning4j.org)
2. Content Page (deeplearning4j.org/xxx)
  * header
  * navigation
  * sidebar
  * footer

---

### Landing Page

To edit the Landing Page, edit the html file located at **"_layouts/index.html"**

---

### Content Page

The default content layout is located at **"_layouts/default.html"**. The default layout contains html blocks from **"_includes/"**:

1. header.html
2. navigation.html
3. sidebar.html
4. footer.html


![WEBSITE LAYOUT](/img/website-layout.jpg)

To write a post with the layout you want, you should add this in the beginning of your post:

**---**

**title: YOUR TITLE**

**layout: default (or other layout you preferred)**

**---**

In example:

![MD LAYOUT](/img/sample-layout-theme.jpg)

---

### Header

Header html file located at **"_includes/header.html"**

**NOTE:** You shouldn't edit this file unless needed.

---

### Navigation

Navigation html file located at **"_includes/navigation.html"**

If you are modifying the Navigation file for other language, then duplicated the **navigation.html** and rename it to **"lang-navigation.html"** (i.e. cn-navigation.html)

---

### Sidebar

Navigation html file located at **"_includes/sidebar.html"**

If you are modifying the Navigation file for other language, then duplicated the **sidebar.html** and rename it to **"lang-sidebar.html"** (i.e. cn-sidebar.html)

---

### Footer

Footer html file located at **"_includes/footer.html"**

**NOTE:** You shouldn't edit this file unless needed.

---

## STYLING (Standard)

### For Image Use the following code:

&lt;img class="img-responsive center-block" src="../img/**your-image-name-here.png**" alt="deeplearning4j"&gt;

### If you are putting up the dl4j or other code, use the following:

**More than one line:**

&lt;pre class="line-numbers"&gt;&lt;code class="language-java"&gt;

*your code here*

*your code here*

*...*

&lt;/code&gt;&lt;/pr&gt;



**Only one line:**

&lt;pre&gt;&lt;code class="language-java"&gt;

*your code here*

&lt;/code&gt;&lt;/pr&gt;

**NOTE**: Please use **&amp;lt;** for **&lt;**, and **&amp;gt;** for **&gt;** in order not to mess up with the html coding.
=======
Deeplearning4J: Neural Networks for Java/JVM
=========================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.deeplearning4j/deeplearning4j-core/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.deeplearning4j/deeplearning4j-core)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.deeplearning4j/deeplearning4j-core/badge.svg)](http://deeplearning4j.org/doc)

Deeplearning4J is an Apache 2.0-licensed, open-source, distributed neural net library written in Java and Scala. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

Deeplearning4J integrates with Hadoop and Spark and runs on several backends that enable use of CPUs and GPUs. The aim is to create a plug-and-play solution that is more convention than configuration, and which allows for fast prototyping.

The most recent stable release in Maven Central is `0.8.0`, and the current master is `0.8.1-SNAPSHOT`.

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
3. Fork [the repository](https://github.com/deeplearning4j/deeplearning4j.git)
   on GitHub to start making your changes (branch off of the master branch).
4. Write a test that shows the bug was fixed or the feature works as expected.
5. Note the repository follows
   the [Google Java style](https://google.github.io/styleguide/javaguide.html)
   with two modifications: 120-char column wrap and 4-spaces indentation. You
   can format your code to this format by typing `mvn formatter:format` in the
   subproject you work on, by using the `contrib/formatter.xml` at the root of
   the repository to configure the Eclipse formatter, or by [using the INtellij
   plugin](https://github.com/HPI-Information-Systems/Metanome/wiki/Installing-the-google-styleguide-settings-in-intellij-and-eclipse).
6. Send a pull request and bug us on Gitter until it gets merged and published. :)
>>>>>>> master

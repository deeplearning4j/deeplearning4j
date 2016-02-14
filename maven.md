---
title: Maven for Python Programmers
layout: default
---

# Maven for Python Programmers

[Maven](https://en.wikipedia.org/wiki/Apache_Maven) is the most commonly used build automation tool for Java programmers. While there is no Python tool that matches Maven feature for feature, it is analogous to a package management system like [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) in Python, or PyBuilder, or [Distutils](http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html). 

It is also the single most convenient way to get up and running with Deeplearning4j, which offers a [Scala API](http://nd4j.org/scala.html) whose syntax will strike many Python programmers as eerily familiar, while offering them powerful concurrent features. 

As a build automation tool, Maven compiles source to byte code and links object files to executables, library files etc. Its deliverable is a JAR file, created from Java source, as well as resources for deployment. 

(A [JAR](https://en.wikipedia.org/wiki/JAR_%28file_format%29) is a *Java ARchive*, a package file format that aggregates many Java class files, associated metadata and resources such as text and images. It's a compressed file format that helps Java runtimes  deploy a set of classes and their resources.) 

Maven dynamically downloads Java libraries and Maven plug-ins from Maven Central Repository which are specified in an XML file that stores a Project Object Model, which you'll find in the file POM.xml. 

![Alt text](../img/maven_schema.png)

To quote *Maven: The Complete Reference*: 

		Running mvn install from the command line will process resources, compile source, execute unit tests, create a JAR and install the JAR in a local repository for reuse in other projects. 

Like Deeplearning4j, Maven relies on convention over configuration, which means that it provides default values that allow it to run without the programmer having to specify each parameter for each new project. 

If you have both IntelliJ and Maven installed, IntelliJ will allow you to choose Maven when creating a new project in the IDE, and will then take you through the wizard (we comment more thoroughly on the process [in our Getting Started page](http://nd4j.org/getstarted.html#maven)). That is, you can make the build happen from within IntelliJ, without going anywhere else. 

Alternatively, you can use Maven from your project's root directory in the command prompt to freshly install it:

		mvn clean install -DskipTests -Dmaven.javadoc.skip=true

Several useful books have been written about Apache Maven. They are available on the website of Sonatype, the company that supports the open-source project. 

### Troubleshooting Maven

* Older versions of Maven, such as 3.0.4, are likely to throw exceptions like a NoSuchMethodError. This can be fixed by upgrading to the latest version of Maven. 
* After you install Maven, you may receive a message like this: *'mvn is not recognised as an internal or external command, operable program or batch file.'* That means you need Maven in your [PATH variable](https://www.java.com/en/download/help/path.xml), which you can change like any other environmental variable. 
* As the DL4J code base grows, installing from source requires more memory. If you encounter a Permgen error during the DL4J build, you may need to add more heap space. To do that, you'll need to find and alter your hidden .bash_profile file, which adds environmental variables to bash. To see those variables, enter *env* in the command line. To add more heap space, enter this command in your console:
      echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile

### Further reading:

* [Maven by Example](https://books.sonatype.com/mvnex-book/reference/public-book.html)
* [Maven: The Complete Reference](https://books.sonatype.com/mvnref-book/reference/public-book.html)
* [Developing with Eclipse and Maven](https://books.sonatype.com/m2eclipse-book/reference/)


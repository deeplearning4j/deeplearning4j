---
title: 
layout: default
---

# Maven for Python Programmers

[Maven](https://en.wikipedia.org/wiki/Apache_Maven) is the most commonly used build automation tool for Java programmers, analogous to a package management system like [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) in Python, or PyBuilder. 

It is also the single most convenient way to get up and running with Deeplearning4j, which offers a [Scala API](../scala.html) whose syntax will strike many Python programmers as eerily familiar, while offering them powerful concurrent features. 

As a build automation tool, Maven compiles source to byte code and links object files to executables, library files etc. Its deliverable is a JAR file, created from Java source, as well as resources for deployment. 

(A [JAR](https://en.wikipedia.org/wiki/JAR_%28file_format%29) is a *Java ARchive*, a package file format that aggregates many Java class files, associated metadata and resources such as text and images. It's a compressed file format that helps Java runtimes  deploy a set of classes and their resources.) 

Maven dynamically downloads Java libraries and Maven plug-ins from Maven Central Repository which are specified in an XML file that stores a Project Object Model, which you'll find in the file POM.xml. 

![Alt text](../img/how_maven_works.png)

To quote *Maven: The Complete Reference*: 

		Running mvn install from the command line will process resources, compile source, execute unit tests, create a JAR and install the JAR in a local repository for reuse in other projects. 

Like Deeplearning4j, Maven relies on convention over configuration, which means that it provides default values that allow it to run without the programmer having to specify each parameter for each new project. 

If you have both IntelliJ and Maven installed, IntelliJ will allow you to choose Maven when creating a new project in the IDE, and will then take you through the wizard (we comment more thoroughly on the process [here](http://nd4j.org/getstarted.html#maven)). That is, you can make the build happen from within IntelliJ, without going anywhere else. 

Several useful books have been written about Apache Maven. They are available on the website of Sonatype, the company that supports the open-source project. 

Further reading:

* [Maven by Example](https://books.sonatype.com/mvnex-book/reference/public-book.html)
* [Maven: The Complete Reference](https://books.sonatype.com/mvnref-book/reference/public-book.html)
* [Developing with Eclipse and Maven](https://books.sonatype.com/m2eclipse-book/reference/)


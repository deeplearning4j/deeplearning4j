---
layout: default
title: Working With Source
---

# Working With Source

If you are not planning to contribute to Deeplearning4j as a committer, or don't need the latest alpha version, we recommend downloading the most recent stable release of Deeplearning4j from [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j), 0.4-rc*. That's as simple as adding dependencies to your POM.xml file in IntelliJ.

On the other hand, our [Github repo is here](https://github.com/deeplearning4j/deeplearning4j/). Install [Github](http://nd4j.org/getstarted.html) for [Mac](https://mac.github.com/) or [Windows](https://windows.github.com/). Then 'git clone' the repository, and run this command for Maven:

      mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true

If you want to run Deeplearning4j examples after installing from trunk, you should *git clone* ND4J, Canova and Deeplearning4j, in that order, and then install all from source using Maven with the command above.

Following these steps, you should be able to run the 0.4-rc* examples. 

If you have an existing project, you can build Deeplearning4j's source files yourself and then add dependencies as JAR files to your project. Each dependency used with Deeplearning4j and [ND4J](http://nd4j.org/dependencies.html) can be included in your project's POM.xml as a jar like this, specifying the most recent version of ND4J or Deeplearning4j between the `properties` tags. 

        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-ui</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-nlp</artifactId>
            <version>${dl4j.version}</version>
        </dependency>

To work with source, you need to install a [project Lombok plugin](https://projectlombok.org/download.html) for IntelliJ or Eclipse.

To learn more about contributing to Deeplearning4j, please read our [Dev Guide](../devguide.html).

<!-- #### <a name="one">Magical One-Line Install</a>

For users who have never `git cloned` Deeplearning4j before, you should be able to install the framework, along with ND4J and Canova, by entering one line in your command prompt:

      git clone https://github.com/deeplearning4j/deeplearning4j/; cd deeplearning4j;./setup.sh -->

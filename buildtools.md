---
title: Configuring Automated Build Tools
layout: default
---

## Configuring Automated Build Tools

While we encourage Deeplearning4j, ND4J and DataVec users to employ Maven, it's worthwhile documenting how to configure build files for other tools, like Ivy, Gradle and SBT -- particularly since Google prefers Gradle over Maven for Android projects. 

The instructions below apply to all DL4J and ND4J submodules, such as deeplearning4j-api, deeplearning4j-scaleout, and ND4J backends. You can find the **latest version** of any project or submodule on [Maven Central](https://search.maven.org/). As of January 2017, the latest version is `0.7.2`. Building from source, the latest version is `0.7.3-SNAPSHOT`.

## Maven

You can use Deeplearning4j with Maven by adding the following to your POM.xml:

    <dependencies>
      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>deeplearning4j-core</artifactId>
          <version>${FIND THE VERSION FROM OUR EXAMPLES http://github.com/deeplearning4j/dl4j-examples}</version>
      </dependency>
    </dependencies>

Note that deeplearning4J will have dependencies on nd4J and DataVec, for a working example of proper Maven configurations please see our [examples](http://github.com/deeplearning4j/dl4j-examples)

## Ivy

You can use lombok with ivy by adding the following to your ivy.xml:

    <dependency org="org.deeplearning4j" name="deeplearning4j-core" rev="${FIND THE VERSION FROM OUR EXAMPLES http://github.com/deeplearning4j/dl4j-examples}" conf="build" />

## SBT

You can use Deeplearning4j with SBT by adding the following to your build.sbt:

    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "${FIND THE VERSION FROM OUR EXAMPLES http://github.com/deeplearning4j/dl4j-examples}"

## Gradle

You can use Deeplearning4j with Gradle by adding the following to your build.gradle in the dependencies block:

    compile "org.deeplearning4j:deeplearning4j-core:${FIND THE VERSION FROM OUR EXAMPLES http://github.com/deeplearning4j/dl4j-examples}"

## Leiningen

Clojure programmers may want to use [Leiningen](https://github.com/technomancy/leiningen/) or [Boot](http://boot-clj.com/) to work with Maven. A [Leiningen tutorial is here](https://github.com/technomancy/leiningen/blob/master/doc/TUTORIAL.md).

NOTE: You'll still need to download ND4J, DataVec and Deeplearning4j, or doubleclick on the their respective JAR files file downloaded by Maven / Ivy / Gradle, to install them in your Eclipse installation.

## Backends

[Backends](http://nd4j.org/backend) and other [dependencies](http://nd4j.org/dependencies) are explained on the ND4J website.

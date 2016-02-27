---
title: Configuring Automated Build Tools
layout: default
---

## Configuring Automated Build Tools

While we encourage Deeplearning4j, ND4J and Canova users to employ Maven, it's worthwhile documenting how to configure build files for other tools, like Ivy, Gradle and SBT -- particularly since Google prefers Gradle over Maven for Android projects. 

The instructions below apply to all DL4J and ND4J submodules, such as *deeplearning4j-api, deeplearning4j-scaleout, and ND4J backends. You can find the **latest version** of any project or submodule on [Maven Central](https://search.maven.org/). As of February 2016, the latest version is `rc3.8`.

## Maven

You can use Deeplearning4j with Maven by adding the following to your POM.xml:

    <dependencies>
      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>deeplearning4j-core</artifactId>
          <version>0.4-rc3.8</version>
          <scope>provided</scope>
      </dependency>
    </dependencies>

## Ivy

You can use lombok with ivy by adding the following to your ivy.xml:

    <dependency org="org.deeplearning4j" name="deeplearning4j-core" rev="0.4-rc3.8" conf="build" />

## SBT

You can use Deeplearning4j with SBT by adding the following to your build.sbt:

    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8"

## Gradle

You can use Deeplearning4j with Gradle by adding the following to your build.gradle in the dependencies block:

    provided "org.deeplearning4j:deeplearning4j-core:0.4-rc3.8"

## Leiningen

Clojure programmers may want to use [Leiningen](https://github.com/technomancy/leiningen/) or [Boot](http://boot-clj.com/) to work with Maven. A [Leiningen tutorial is here](https://github.com/technomancy/leiningen/blob/master/doc/TUTORIAL.md).

NOTE: You'll still need to download ND4J, Canova and Deeplearning4j, or doubleclick on the their respective JAR files file downloaded by Maven / Ivy / Gradle, to install them in your Eclipse installation.

## Backends

[Backends](http://nd4j.org/backend) and other [dependencies](http://nd4j.org/dependencies) are explained on the ND4J website.

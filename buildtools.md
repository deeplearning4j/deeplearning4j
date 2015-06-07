---
title: 
layout: default
---

## Configuring Automated Build Tools

While we encourage Deeplearning4j, ND4J and Canova users to employ Maven, it's worthwhile documenting how to configure build files for other tools, like Ivy, Gradle and SBT. Particularly since Google prefers Gradle over Maven for Android projects. The instructions below apply to all DL4J and ND4J submodules, such as deeplearning4j-api, deeplearning4j-scaleout, nd4j-jblas, etc.

## Maven

You can use Deeplearning4j with Maven by adding the following to your POM.xml:

    <dependencies>
    	<dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>0.0.3.3.4.alpha2</version> //<--find the latest version on Maven Central: search.maven.org
    		<scope>provided</scope>
    	</dependency>
    </dependencies>

## Ivy

You can use lombok with ivy by adding the following to your ivy.xml:

    <dependency org="org.deeplearning4j" name="deeplearning4j-core" rev="0.0.3.3.4.alpha2" conf="build" />

## SBT

You can use Deeplearning4j with SBT by adding the following to your build.sbt:

    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.0.3.3.4.alpha2"

## Gradle

You can use Deeplearning4j with Gradle by adding the following to your build.gradle in the dependencies block:

    provided "org.deeplearning4j:deeplearning4j-core:0.0.3.3.4.alpha2"

NOTE: You'll still need to download ND4J, Canova and Deeplearning4j, or doubleclick on the their respective JAR files file downloaded by Maven / Ivy / Gradle, to install them into your Eclipse installation.

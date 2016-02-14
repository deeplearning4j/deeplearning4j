---
title: Using Deeplearning4j, ND4J and Canova SNAPSHOTS 
layout: default
---

# Using Deeplearning4j, ND4J and Canova SNAPSHOTS 

This page will explain how to update Deeplearning4j, ND4J and Canova to the current master in development, rather than the latest stable version of the libraries. 

(To verify the stable version of Deeplearning4j, [go here](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j). As of this writing, it was rc3.8).

Deeplearning4j users who would like to explore the libraries' latest features need to work with a SNAPSHOT. We have made those SNAPSHOTS available via Maven. If you update your POM.xml file in your Deeplearning4j project, Maven will download and build the current SNAPSHOT automatically. You need to include the SNAPSHOT repository information. 

    <repositories>
      <repository>
        <id>snapshots-repo</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots</url>
        <releases><enabled>false</enabled></releases>
        <snapshots><enabled>true</enabled></snapshots>
      </repository>
    </repositories>

Then, between the `properties` tags, where you specify the version of dependency, you will need to update the versions for ND4J, Canova and Deeplearning4j to the latest SNAPSHOT. As of this writing, it was 0.4-rc3.9-SNAPSHOT for ND4J and DL4J, and 0.0.0.15-SNAPSHOT for Canova.

    <properties>
      <nd4j.version>LatestVersionPastedFromMavenCentralHere-SNAPSHOT</nd4j.version>
      <canova.version>LatestVersionPastedFromMavenCentralHere-SNAPSHOT</canova.version>
      <dl4j.version>LatestVersionPastedFromMavenCentralHere-SNAPSHOT</dl4j.version>
    </properties>

To verify the number of the latest version, please see the following pages on Sonatype:

* [ND4J](https://oss.sonatype.org/content/repositories/snapshots/org/nd4j/nd4j-api/)
* [Canova](https://oss.sonatype.org/content/repositories/snapshots/org/nd4j/canova-api/)
* [Deeplearning4j](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-core/)

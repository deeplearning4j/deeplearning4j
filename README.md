Canova
=========================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Canova is an Apache2 Licensed open-sourced tool for vectorizing raw data into usable vector formats across machine learning tools. Canova provides both an API and a command line interface (CLI).

## Why Would I Use Canova?

Canova allows a practitioner to take raw data and produce open standard compliant vectorized data (svmLight, etc) in under 5 minutes. Current input data types supported out of the box:

* CSV Data
* Raw Text Data (Tweets, Text Documents, etc)
* Image Data
* Custom File Formats (MNIST)

# Installation

We have several options to work with Canova. The most common way would be to download the last stable released tarball.

## Tarball

Download our latest release at: [ Coming Soon ]

## Clone from the Github Repository

Canova is actively developed and you can clone the repository, compile it and reference it in your project. First clone the [ND4J repo](https://github.com/deeplearning4j/nd4j) and build compile prior to building Canova.

Clone the repository:

    $ git clone https://github.com/deeplearning4j/Canova.git

Compile the project:

    $ cd canova && mvn clean install -DskipTests -Dmaven.javadoc.skip=true


## Use Maven Central Repository

    Search for [canova](https://search.maven.org/#search%7Cga%7C1%7CCanova) to get a list of jars you can use

    Add the dependency information into your pom.xml

Add the local compiled file dependencies to your pom.xml file like the following example:

	<dependency>
	    <groupId>org.nd4j</groupId>
	    <artifactId>canova-api-SNAPSHOT</artifactId>
	    <version>0.0.0.3</version>
	</dependency>


# Example Uses

* [Using Canova to Vectorize CSV Data from the CLI](https://github.com/deeplearning4j/Canova/wiki/Vectorizing-CSV-Data-With-Canova-and-the-CLI)
 
# Contribute

1. Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the link above.
3. Fork the repository on GitHub to start making your changes to the master branch (or branch off of it).
4. Write a test which shows that the bug was fixed or that the feature works as expected.
5. Send a pull request and bug us on Gitter until it gets merged and published.

## Future Directions / Roadmap

* Adding Pipelines for
    * Timeseries
    * Audio
    * Video
* Parallel engine support
    * Hadoop / MapReduce
* More Text Processing Techniques
    * Kernel Hashing for Text Pipeline


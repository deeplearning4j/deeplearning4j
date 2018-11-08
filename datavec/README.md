# DataVec

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.datavec/datavec-api/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.datavec/datavec-api)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.datavec/datavec-api/badge.svg)](http://deeplearning4j.org/datavecdoc)

DataVec is an Apache 2.0-licensed library for machine-learning ETL (Extract, Transform, Load) operations. DataVec's purpose is to transform raw data into usable vector formats that can be fed to machine learning algorithms. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

## Why Would I Use DataVec?

Data handling is sometimes messy, and we believe it should be distinct from high-performance algebra libraries (such
as [nd4j](https://nd4j.org) or [Deeplearning4j](https://deeplearning4j.org)).

DataVec allows a practitioner to take raw data and produce open standard compliant vectorized data (svmLight, etc)
quickly. Current input data types supported out of the box:

* CSV Data
* Raw Text Data (Tweets, Text Documents, etc)
* Image Data
* LibSVM
* SVMLight
* MatLab (MAT) format
* JSON, XML, YAML, XML

Datavec draws inspiration from a lot of the Hadoop ecosystem tools, and in particular accesses data on disk through the
Hadoop API (like Spark does), which means it's compatible with many records.

DataVec also includes sophisticated functionality for feature engineering, data cleaning and data normalization both for
static data and for sequences (time series). Such operations can be executed on [Apache Spark](https://spark.apache.org/) using DataVec-Spark.

## Datavec's architecture : API, transforms and filters, and schema management

Apart from obviously providing readers for classic data formats, DataVec also provides an interface. So if you wanted to
ingest specific custom data, you wouldn't have to build the whole pipeline. You would just have to write the very first step. For example, if you describe through the API how your data fits into a common format that complies with the interface, DataVec
would return a list of Writables for each record. You'll find more detail on the API in the corresponding [module](https://github.com/deeplearning4j/DataVec/tree/master/datavec-api).

Another thing you can do with DataVec is data cleaning. Instead of having clean, ready-to-go data, let's say you start with data in different forms or from different sources. You might need to do sampling, filtering, or several incredibly messy ETL tasks needed to prepare data in the real world. DataVec offers filters and transformations that help with curating, preparing and massaging your data. It leverages Apache Spark to do this at scale.

Finally, DataVec tracks a schema for your columnar data, across all transformations. This schema is actively checked
through probing, and DataVec will raise exceptions if your data does not match the schema. You can specify filters as
well: you can attach a regular expression to an input column of type `String`, for example, and DataVec will only keep
data that matches this filter.

## On Distribution

Distributed treatment through Apache Spark is optional, including running Spark in local-mode (where your
cluster is emulated with multi-threading) when necessary. Datavec aims to abstract away from the actual execution, and
create at compile time, a logical set of operations to execute. While we have some code that uses Spark, we do not want
to be locked into a single tool, and using [Apache Flink](https://flink.apache.org/) or [Beam](https://beam.apache.org/) are possibilities - projects on which we would welcome collaboration.

## Examples

Examples for using DataVec are available
here: [https://github.com/deeplearning4j/dl4j-examples](https://github.com/deeplearning4j/dl4j-examples)


---
## Contribute

### Where to contribute?

We have a lot in the pipeline, and we'd love to receive contributions. We want to support representing data as
more than a collection of simple types ("*writables*"), and rather as binary data — that will help with GC pressure
across our pipelines and fit better with media-based use cases, where columnar data is not essential. We also expect it
will streamline a lot of the specialized operations we now do on primitive types.

That being said, an area that could use a first contribution is the implementations of the `RecordReader`
interface, since this is relatively self-contained. Of note, to support most of the distributed file formats of the
Hadoop ecosystem, we use [Apache Camel](https://camel.apache.org/). Camel supports
a [pluggable DataFormat](https://camel.apache.org/data-format.html) to allow messages to be marshalled to and from
binary or text formats to support a kind of Message Translator.

Another area that is relatively self-contained is transformations, where you might find a filter or data munging
operation that has not been implemented yet, and provide it in a self-contained way.

## Which maintainers to contact?

It's useful to know which maintainers to contact to get information on a particular part of the code, including reviewing your pull requests, or asking questions on our [gitter channel](https://gitter.im/deeplearning4j/deeplearning4j). For this you can use the following, indicative mapping:

- `RecordReader` implementations:
   @saudet and @agibsonccc
- Transformations and their API:
   @agibsonccc and @AlexDBlack
- Spark and distributed processing:
   @AlexDBlack, @agibsonccc and @huitseeker
- Native formats, geodata:
   @saudet

### How to contribute

1. Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the
   link above.
3. Fork [the repository](https://github.com/deeplearning4j/datavec.git) on GitHub to start making your changes.
4. Write a test, which shows that the bug was fixed or that the feature works as expected.
5. Note the repository follows the [Google Java style](https://google.github.io/styleguide/javaguide.html) with two
   modifications: 120-char column wrap and 4-spaces indentation. You can format your code to this format by typing `mvn
   formatter:format` in the subproject you work on, by using the `contrib/formatter.xml` at the root of the repository
   to configure the Eclipse formatter, or by
   [using the INtellij plugin](https://github.com/HPI-Information-Systems/Metanome/wiki/Installing-the-google-styleguide-settings-in-intellij-and-eclipse).

6. Send a pull request, and bug us on Gitter until it gets merged and published.

## Eclipse Setup

1. Downloading the latest JAR from https://projectlombok.org/download
2. Double-click the JAR file to install the plugin for Eclipse
3. Clone Datavec to your system
4. Import the project as a Maven project
5. You will also need clone and build ND4J and libnd4j

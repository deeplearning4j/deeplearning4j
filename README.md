DataVec
=========================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.datavec/datavec-api/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.datavec/datavec-api)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.datavec/datavec-api/badge.svg)](http://deeplearning4j.org/datavecdoc)

DataVec is an Apache2 Licensed open-sourced tool for machine learning ETL (Extract, Transform, Load) operations. The goal of DataVec is to transform raw data into usable vector formats across machine learning tools.

## Why Would I Use DataVec?

DataVec allows a practitioner to take raw data and produce open standard compliant vectorized data (svmLight, etc) quickly. Current input data types supported out of the box:

* CSV Data
* Raw Text Data (Tweets, Text Documents, etc)
* Image Data


DataVec also includes sophisticated functionality for feature engineering, data cleaning and data normalization both for static data and for sequences (time series).
Such operations can be executed on spark using DataVec-Spark.


## Examples

Examples for using DataVec are available here: [https://github.com/deeplearning4j/dl4j-0.4-examples/tree/master/datavec-examples/src/main](https://github.com/deeplearning4j/dl4j-0.4-examples/tree/master/datavec-examples/src/main)

---
## Contribute

1. Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/deeplearning4j/datavec.git) on GitHub to start making your changes.
4. Write a test, which shows that the bug was fixed or that the feature works as expected.
5. Note the repository follows
   the [Google Java style](https://google.github.io/styleguide/javaguide.html)
   with two modifications: 120-char column wrap and 4-spaces indentation. You
   can format your code to this format by typing `mvn formatter:format` in the
   subproject you work on, by using the `contrib/formatter.xml` at the root of
   the repository to configure the Eclipse formatter, or by [using the INtellij
   plugin](https://github.com/HPI-Information-Systems/Metanome/wiki/Installing-the-google-styleguide-settings-in-intellij-and-eclipse).

6. Send a pull request, and bug us on Gitter until it gets merged and published.

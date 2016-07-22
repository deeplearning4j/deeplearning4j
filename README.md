DataVec
=========================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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
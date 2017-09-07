---
title: Jumpy: Numpy Arrays for the JVM
layout: default
---

# Jumpy: Numpy Arrays for the JVM

Deeplearning4j and ND4J's Python interface has three main components:

* Autodiff (WIP)
* Model import from Keras
* Jumpy

[Jumpy](https://github.com/deeplearning4j/jumpy) is a Python interface for the scientific computing library [ND4J](http://nd4j.org/) (n-dimensional arrays for the JVM) via pointers, which means no network communication is required, unlike other Python tools. 

Jumpy accepts Numpy arrays and allows us to work with those arrays and tensors without copying data. In short, it is a better interface for anyone currently using MLlib or PySpark, because sidestepping data copying makes it faster and more efficient. 

Jumpy is a thin wrapper around Numpy and [Pyjnius](https://pyjnius.readthedocs.io/en/latest/). Jumpy gives you autocomplete for the JVM while working in Python, just pass it the class path and JAR files, as well as dynamic Java class creation. While PySpark is consistently behind the latest developments in Scala, Jumpy allows developers to dynamically extend bindings themselves.

Think of Jumpy as one way to get tensors into the JVM where you can easily work with Spark and other big data frameworks there. 

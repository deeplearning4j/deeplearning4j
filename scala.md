---
title: Scala, Apache Spark and Deeplearning4j
layout: default
---

# Scala, Apache Spark and Deeplearning4j

Scala programmers seeking to build deep learning solutions can use Deeplearning4j's Scala API [ScalNet](https://github.com/deeplearning4j/scalnet) or work with the Java framework using the `Builder` pattern. Skymind's numerical computing library, [ND4J](http://nd4j.org/) (n-dimensional arrays for the JVM), comes with a Scala API, [ND4S](https://github.com/deeplearning4j/nd4s). Our full walkthrough of Deeplearning4j's Apache Spark integration is [here](https://deeplearning4j.org/spark).

## Scala

Scala is one of the most exciting languages to be created in the 21st century. It is a multi-paradigm language that fully supports functional, object-oriented, imperative and concurrent programming. It also has a strong type system, and from our point of view, strong type is a convenient form of self-documenting code.

Scala works on the JVM and has access to the riches of the Java ecosystem, but it is less verbose than Java. As we employ it for ND4J, its syntax is strikingly similar to Python, a language that many data scientists are comfortable with. Like Python, Scala makes programmers happy, but like Java, it is quite fast. 

Finally, [Apache Spark](./spark.html) is written in Scala, and any library that purports to work on distributed run times should at the very least be able to interface with Spark. Deeplearning4j and ND4J go a step further, because they work in a Spark cluster, and boast Scala APIs called ScalNet and ND4S. 

We believe Scala's many strengths will lead it to dominate numerical computing, as well as deep learning. We think that will happen on Spark. And we have tried to build the tools to make it happen now. 

## Spark

Deeplearning4j depends on Apache Spark for fast ETL. While many machine-learning tools rely on Spark for computation, this is in fact quite inefficient, and slows down neural net training. The trick to using Apache Spark is pushing the computation to a numerical computing library like ND4J, and its underlying C++ code. 

### See also

* [Docs: Deeplearning4j on Spark](./spark.html)
* [Course: Atomic Scala](http://www.atomicscala.com/) - a recommended beginner's course
* [Martin Odersky's Coursera course on Scala](https://www.coursera.org/learn/progfun1)
* [Book: Scala for Data Science](https://www.amazon.com/Scala-Data-Science-Pascal-Bugnion/dp/1785281372)
* [Video Course: Problem-solving using Scala](https://www.youtube.com/user/DrMarkCLewis)
* [Learn: The Scala Programming Language](http://www.scala-lang.org/documentation/)
* [A Scala Tutorial for Java programmers](http://www.scala-lang.org/docu/files/ScalaTutorial.pdf) (PDF)
* [Scala By Example, by Martin Odersky](http://www.scala-lang.org/docu/files/ScalaByExample.pdf) (PDF) 
* [An Intro to Scala on ND4J](http://nd4j.org/scala.html)
* [Our early-stage Scala API](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-scala-api/src/main/scala/org/nd4j/api/linalg): ([One example on Github](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-scala-api/src/test/scala/org/nd4j/api/linalg/TestNDArray.scala#L18))
* SF Spark Talk: [Deeplearning4j on Spark, and Data Science on the JVM, with ND4J](https://www.youtube.com/watch?v=LCsc1hFuNac&feature=youtu.be)
* [Q&A with Adam Gibson about Spark with Alexy Khrabrov](https://www.youtube.com/watch?v=LJPL8sL0Daw&feature=youtu.be)
* [Our Spark integration](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark)
* [ND4J: Scientific Computing for the JVM](http://nd4j.org)
* [Scala Basics for Python Developers](https://bugra.github.io/work/notes/2014-10-18/scala-basics-for-python-developers/)
* [Why We Love Scala at Coursera](https://tech.coursera.org/blog/2014/02/18/why-we-love-scala-at-coursera/)

A non-exhaustive list of [organizations using Scala](http://alvinalexander.com/scala/whos-using-scala-akka-play-framework):

* AirBnB
* Amazon
* Apple
* Ask.com
* AT&T
* Autodesk
* Bank of America
* Bloomberg
* Credit Suisse
* eBay
* Foursquare
* (The) Guardian
* IBM
* Klout
* LinkedIn
* NASA
* Netflix
* precog
* Siemens
* Sony
* Twitter
* Tumblr
* UBS
* (The) Weather Channel
* Xerox
* Yammer

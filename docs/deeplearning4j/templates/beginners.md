---
title: Deep Learning for Beginners
short_title: Beginners
description: Road map for beginners new to deep learning.
category: Get Started
weight: 10
---

## How Do I Start Using Deep Learning?

Where you start depends on what you already know. 

The prerequisites for really understanding deep learning are linear algebra, calculus and statistics, as well as programming and some machine learning. The prerequisites for applying it are just learning how to deploy a model. 

In the case of Deeplearning4j, you should know Java well and be comfortable with tools like the IntelliJ IDE and the automated build tool Maven. [Skymind's SKIL](https://docs.skymind.ai/) also includes a managed Conda environment for machine learning tools using Python. 

Below you'll find a list of resources. The sections are roughly organized in the order they will be useful. 

## Free Machine- and Deep-learning Courses Online

* [Andrew Ng's Machine-Learning Class on YouTube](https://www.youtube.com/watch?v=qeHZOdmJvFU) 
* [Geoff Hinton's Neural Networks Class on YouTube](https://youtu.be/2fRnHVVLf1Y) 
* [Patrick Winston's Introduction to Artificial Intelligence @MIT](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/) (For those interested in a survey of artificial intelligence.)
* [Andrej Karpathy's Convolutional Neural Networks Class at Stanford](http://cs231n.github.io) (For those interested in image recognition.)
* [ML@B: Machine Learning Crash Course: Part 1](https://ml.berkeley.edu/blog/2016/11/06/tutorial-1/)
* [ML@B: Machine Learning Crash Course: Part 2](https://ml.berkeley.edu/blog/2016/12/24/tutorial-2/)
* [Gradient descent, how neural networks learn, Deep learning, part 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&feature=youtu.be)

## Math

The math involved with deep learning is basically linear algebra, calculus and probility, and if you have studied those at the undergraduate level, you will be able to understand most of the ideas and notation in deep-learning papers. If haven't studied those in college, never fear. There are many free resources available (and some on this website).

* [Seeing Theory: A Visual Introduction to Probability and Statistics](http://students.brown.edu/seeing-theory/)
* [Andrew Ng's 6-Part Review of Linear Algebra](https://www.youtube.com/playlist?list=PLnnr1O8OWc6boN4WHeuisJWmeQHH9D_Vg)
* [Khan Academy's Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra)
* [Linear Algebra for Machine Learning](https://www.youtube.com/watch?v=ZumgfOei0Ak); Patrick van der Smagt
* [CMU's Linear Algebra Review](http://www.cs.cmu.edu/~zkolter/course/linalg/outline.html)
* [Math for Machine Learning](https://www.umiacs.umd.edu/~hal/courses/2013S_ML/math4ml.pdf)
* [Immersive Linear Algebra](http://immersivemath.com/ila/learnmore.html)
* [Probability Cheatsheet](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf)
* [The best linear algebra books](https://begriffs.com/posts/2016-07-24-best-linear-algebra-books.html)
* [Markov Chains, Visually Explained](http://setosa.io/ev/markov-chains/)
* [An Introduction to MCMC for Machine Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7133&rep=rep1&type=pdf)
* [Eigenvectors, Eigenvalues, PCA, Covariance and Entropy](/eigenvector)
* [Markov Chain Monte Carlo (MCMC) & Machine Learning](/markovchainmontecarlo)

## Programming

If you do not know how to program yet, you can start with Java, but you might find other languages easier. Python and Ruby resources can convey the basic ideas in a faster feedback loop. "Learn Python the Hard Way" and "Learn to Program (Ruby)" are two great places to start. 

* [Learn Java The Hard Way](https://learnjavathehardway.org/)
* [Learn Python the Hard Way](http://learnpythonthehardway.org/)
* [Pyret: A Python Learning Environment](https://www.pyret.org/)
* [Scratch: A Visual Programming Environment From MIT](https://scratch.mit.edu/)
* [Learn to Program (Ruby)](https://pine.fm/LearnToProgram/)
* [Intro to the Command Line](http://cli.learncodethehardway.org/book/)
* [Additional command-line tutorial](http://www.learnenough.com/command-line)
* [A Vim Tutorial and Primer](https://danielmiessler.com/study/vim/) (Vim is an editor accessible from the command line.)
* [Intro to Computer Science (CS50 @Harvard edX)](https://www.edx.org/course/introduction-computer-science-harvardx-cs50x)
* [A Gentle Introduction to Machine Fundamentals](https://marijnhaverbeke.nl/turtle/)

If you want to jump into deep-learning from here without Java, we recommend [Theano](http://deeplearning.net/) and the various Python frameworks built atop it, including [Keras](https://github.com/fchollet/keras) and [Lasagne](https://github.com/Lasagne/Lasagne).

## Java

Once you have programming basics down, tackle Java, the world's most widely used programming language. Most large organizations in the world operate on huge Java code bases. (There will always be Java jobs.) The big data stack -- Hadoop, Spark, Kafka, Lucene, Solr, Cassandra, Flink -- have largely been written for Java's compute environment, the JVM.

* [Think Java: Interactive Web-based Dev Environment](https://books.trinket.io/thinkjava/)
* [Learn Java The Hard Way](https://learnjavathehardway.org/)
* [Java Resources](http://wiht.link/java-resources)
* [Java Ranch: A Community for Java Beginners](http://javaranch.com/)
* [Intro to Programming in Java @Princeton](http://introcs.cs.princeton.edu/java/home/)
* [Head First Java](http://www.amazon.com/gp/product/0596009208)
* [Java in a Nutshell](http://www.amazon.com/gp/product/1449370829)
* [Java Programming for Complete Beginners in 250 Steps](https://www.udemy.com/java-programming-tutorial-for-beginners/?siteID=JVFxdTr9V80-nE4LGc8755WIfh0f9e7Jqw&LSNPUBID=JVFxdTr9V80)

## Deeplearning4j

With that under your belt, we recommend you approach Deeplearning4j through its [examples](https://github.com/deeplearning4j/dl4j-examples). 

* [Quickstart](./quickstart.html)

You can also download a [free version of the Skymind Intelligence Layer](https://docs.skymind.ai/), which supports Python, Java and Scala machine-learning and data science tools. SKIL is a machine-learning backend that works on prem and in the cloud, and can ship with your software to provide a machine learning model server. 

## Other Resources

Most of what we know about deep learning is contained in academic papers. We've linked to a number of them [here](./deeplearningpapers). 

While individual courses have limits on what they can teach, the Internet does not. Most math and programming questions can be answered by Googling and searching sites like [Stackoverflow](http://stackoverflow.com) and [Math Stackexchange](https://math.stackexchange.com/).
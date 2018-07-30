---
title: Contributor's Guide
short_title: Contribute
description: How to contribute to the Eclipse Deeplearning4j source code.
category: Get Started
weight: 10
---

## Prerequisites

Before contributing, make sure you know the structure of all of the Eclipse Deeplearning4j libraries. As of early 2018, all libraries now live in the Deeplearning4j [monorepo](https://github.com/deeplearning4j/deeplearning4j). These include:

- DeepLearning4J: Contains all of the code for learning neural networks, both on a single machine and distributed.
- ND4J: “N-Dimensional Arrays for Java”. ND4J is the mathematical backend upon which DL4J is built. All of DL4J’s neural networks are built using the operations (matrix multiplications, vector operations, etc) in ND4J. ND4J is how DL4J supports both CPU and GPU training of networks, without any changes to the networks themselves. Without ND4J, there would be no DL4J.
- DataVec: DataVec handles the data import and conversion side of the pipeline. If you want to import images, video, audio or simply CSV data into DL4J: you probably want to use DataVec to do this.
- Arbiter: Arbiter is a package for (amongst other things) hyperparameter optimization of neural networks. Hyperparameter optimization refers to the process of automating the selection of network hyperparameters (learning rate, number of layers, etc) in order to obtain good performance.

We also have an extensive examples respository at [dl4j-examples](https://github.com/deeplearning4j/dl4j-examples).


## Ways to contribute

There are numerous ways to contribute to to DeepLearning4J (and related projects), depending on your interests and experince. Here’s some ideas:

- Add new types of neural network layers (for example: different types of RNNs, locally connected networks, etc)
- Add a new training feature
- Bug fixes
- DL4J examples: Is there an application or network architecture that we don’t have examples for?
- Testing performance and identifying bottlenecks or areas to improve
- Improve website documentation (or write tutorials, etc)
- Improve the JavaDocs


There are a number of different ways to find things to work on. These include:

- Looking at the issue trackers:
https://github.com/deeplearning4j/deeplearning4j/issues
https://github.com/deeplearning4j/dl4j-examples/issues
- Reviewing our Roadmap
- Talking to the developers on Gitter, especially our early adopters channel
- Reviewing recent papers and blog posts on training features, network architectures and applications
- Reviewing the website and examples - what seems missing, incomplete, or would simply be useful (or cool) to have?

## General guidelines

Before you dive in, there’s a few things you need to know. In particular, the tools we use:

- Maven: a dependency management and build tool, used for all of our projects. See this for details on Maven.
- Git: the version control system we use
- Project Lombok: Project Lombok is a code generation/annotation tool that is aimed to reduce the amount of ‘boilerplate’ code (i.e., standard repeated code) needed in Java. To work with source, you’ll need to install the Project Lombok plugin for your IDE
- VisualVM: A profiling tool, most useful to identify performance issues and bottlenecks.
- IntelliJ IDEA: This is our IDE of choice, though you may of course use alternatives such as Eclipse and NetBeans. You may find it easier to use the same IDE as the developers in case you run into any issues. But this is up to you.

Things to keep in mind:

- Code should be Java 7 compliant
- If you are adding a new method or class: add JavaDocs
- You are welcome to add an author tag for significant additions of functionality. This can also help future contributors, in case they need to ask questions of the original author. If multiple authors are present for a class: provide details on who did what (“original implementation”, “added feature x” etc)
- Provide informative comments throughout your code. This helps to keep all code maintainable.
- Any new functionality should include unit tests (using JUnit) to test your code. This should include edge cases.
- If you add a new layer type, you must include numerical gradient checks, as per these unit tests. These are necessary to confirm that the calculated gradients are correct
- If you are adding significant new functionality, consider also updating the relevant section(s) of the website, and providing an example. After all, functionality that nobody knows about (or nobody knows how to use) isn’t that helpful. Adding documentation is definitely encouraged when appropriate, but strictly not required.
- If you are unsure about something - ask us on Gitter!
---
title: Developer Guide for Deeplearning4j
layout: default
---

# Developer Guide

## DeepLearning4J and Related Projects - An Overview

DeepLearning4j is perhaps the more visible project, there are a number of other projects that you should be familiar with - and may consider contributing to. These include:

* [DeepLearning4J](https://github.com/deeplearning4j/deeplearning4j): Contains all of the code for learning neural networks, both on a single machine and distributed.
* [ND4J](https://github.com/deeplearning4j/nd4j): "N-Dimensional Arrays for Java". ND4J is the mathematical backend upon which DL4J is built. All of DL4J's neural networks are built using the operations (matrix multiplications, vector operations, etc) in ND4J. ND4J is how DL4J supports both CPU and GPU training of networks, without any changes to the networks themselves. Without ND4J, there would be no DL4J.
* [Canova](https://github.com/deeplearning4j/Canova): Canova handles the data import and conversion side of the pipeline. If you want to import images, video, audio or simply CSV data into DL4J: you probably want to use Canova to do this.
* [Arbiter](https://github.com/deeplearning4j/Arbiter): Arbiter is a package for (amongst other things) hyperparameter optimization of neural networks. Hyperparameter optimization refers to the process of automating the selection of network hyperparameters (learning rate, number of layers, etc) in order to obtain good performance.
* [DL4J Examples](https://github.com/deeplearning4j/dl4j-0.4-examples)

Deeplearning4j and ND4J are distributed under an Apache 2.0 license.

## Ways to Contribute

There are numerous ways to contribute to to DeepLearning4J (and related projects), depending on your interests and experince. Here's some ideas:

* Add new types of neural network layers (for example: different types of RNNs, locally connected networks, etc)
* Add a new training feature
* Bug fixes
* DL4J examples: Is there an application or network architecture that we don't have examples for?
* Testing performance and identifying bottlenecks or areas to improve
* Improve website documentation (or write tutorials, etc)
* Improve the JavaDocs

There are a number of different ways to find things to work on. These include:

* Looking at the issue trackers:
  * [https://github.com/deeplearning4j/deeplearning4j/issues](https://github.com/deeplearning4j/deeplearning4j/issues)
  * [https://github.com/deeplearning4j/nd4j/issues](https://github.com/deeplearning4j/nd4j/issues)
  * [https://github.com/deeplearning4j/Canova/issues](https://github.com/deeplearning4j/Canova/issues)
  * [https://github.com/deeplearning4j/dl4j-0.4-examples/issues](https://github.com/deeplearning4j/dl4j-0.4-examples/issues)
* Reviewing our [Roadmap](http://deeplearning4j.org/roadmap.html)
* Talking to the developers on [Gitter](https://gitter.im/deeplearning4j/deeplearning4j)
* Reviewing recent papers and blog posts on training features, network architectures and applications
* Reviewing the [website](http://deeplearning4j.org/documentation.html) and [examples](https://github.com/deeplearning4j/dl4j-0.4-examples/) - what seems missing, incomplete, or would simply be useful (or cool) to have?




## Working on DL4J/ND4J and Other Projects - The Basics

Before you dive in, there's a few things you need to know. In particular, the tools we use:

* Maven: a dependency management and build tool, used for all of our projects. See [this](http://deeplearning4j.org/maven.html) for details on Maven.
* Git: the version control system we use
* [Project Lombok](https://projectlombok.org/): Project Lombok is a code generation/annotation tool that is aimed to reduce the amount of 'boilerplate' code (i.e., standard repeated code) needed in Java. To work with source, you'll need to install the [Project Lombok plugin](https://projectlombok.org/download.html) for your IDE
* [Travis](https://travis-ci.org/): Travis is a continuous integration service that we use to provide automatic testing. Any pull requests that are made to our repositories will be automatically tested by Travis, which will attempt to build you code and then run *all* of the unit tests in the project. This can help to automatically identify any issues in a pull request.

Some other tools you might find useful:

* [VisualVM](https://visualvm.java.net/): A profiling tool, most useful to identify performance issues and bottlenecks.
* [IntelliJ IDEA](https://www.jetbrains.com/idea/): This is our IDE of choice, though you may of course use alternatives such as Eclipse and NetBeans. You may find it easier to use the same IDE as the developers in case you run into any issues. But this is up to you.

And here's the typical workflow for contributing to any of our projects:

1. Pick something to work on. If it has an [open issue](https://github.com/deeplearning4j/deeplearning4j/issues): consider assigning yourself to it. This can help with coordination and avoiding duplication of work. If there is no open issue: it may make sense to open one, for example for bugs or new features.
2. If you haven't forked DL4J (or ND4J/Canova/whatever-you-are-working-on) yet: you can fork it from the main repository page for that project
3. Start each day by syncing your local copy of the repository (or repositories) you are working on.
  * For details, see [Syncing a Fork](https://help.github.com/articles/syncing-a-fork/)
  * (Note: if you have write access to the deeplearning4j repos (most people don't) instead do a 'git pull' and 'mvn clean install' on ND4J, Canova and DL4J)
4. Make your changes in your local copy of the code. Add test to ensure your new functionality works, or the bug you fixed is actually fixed.
5. Create a [Pull Request](https://help.github.com/articles/using-pull-requests/)
  * Provide an informative title and description. Reference any issues relevant to the pull request.
6. Wait for Travis to run.
  * If Travis fails: review the logs and check why. Is there a particular test that failed? If so, review this locally, and make any changes as necessary. Any changes pushed up to your local repository will automatically added to the pull request.
7. Once Travis succeeds one of two things will happen
  * Your pull request will be reviewed and merged. Hooray!
  * Or, after review: You might be asked to make some final changes before we can merge your code. Usually only minor alterations (if any) are required.


Here's some guidelines and things to keep in mind:

* Code should be Java 7 compliant
* If you are adding a new method or class: add JavaDocs
  * For example: what each of the arguments are, and their assumptions (nullable? always positive?)
  * You are welcome to add an author tag for significant additions of functionality. This can also help future contributors, in case they need to ask questions of the original author. If multiple authors are present for a class: provide details on who did what ("original implementation", "added feature x" etc)
  * JavaDocs can be quite detaled, and include various formatting options (code, bold/italic text, links, etc): see [this, for further details](http://docs.oracle.com/javase/7/docs/technotes/tools/windows/javadoc.html)
* Provide informative comments throughout your code. This helps to keep all code maintainable.
* Any new functionality should include unit tests (using [JUnit](http://junit.org/)) to test your code. This should include edge cases.
* If you add a new layer type, you must include numerical gradient checks, as per [these unit tests](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/GradientCheckTests.java). These are necessary to confirm that the calculated gradients are correct
* If you are adding significant new functionality, consider also updating the relevant section(s) of the website, and providing an example. After all, functionality that nobody knows about (or nobody knows how to use) isn't that helpful. Adding documentation is definitely encouraged when appropriate, but strictly not required.
* If you are unsure about something - ask!



## Contributing Documentation: How the Websites Work

So you want to contribute to one of the websites, such as [deeplearning4j.org](http://deeplearning4j.org/) and [nd4j.org](http://nd4j.org/)? Great. These are also open source, and we welcome pull requests for them.

How does the website actually work? This is actually pretty straightforward. Consider for example deeplearning4j.org:

* The website itself is hosted on GitHub, specifically in the [gh-pages](https://github.com/deeplearning4j/deeplearning4j/tree/gh-pages) branch
* Website code is in [Markdown](https://help.github.com/articles/markdown-basics/) format/language, and is automatically converted to a HTML page. For example, this whole page you are reading is written in markdown.
  * There are a few exceptions to this, such as the documentation.html page, which are in HTML format.

Markdown itself is relatively simple to pick up. If you are unfamiliar with it, have a look at the existing pages (.md files) in the [gh-pages](https://github.com/deeplearning4j/deeplearning4j/tree/gh-pages) branch. You can use these as a guide to get started.



## DeepLearning4J: Some Things to Know

DL4J is a very large and complicated piece of software. Giving a complete overview of how DL4J works is quite difficult. Though that is what this section attempts to do.

Let's start with the main packages:

* deeplearning4j-core: Contains all of the layers, configuration and optimization code.
* deeplearning4j-scaleout: distributed training (Spark), plus some other models such as Word2Vec
* deeplearning4j-ui: user-interface functionality, such as the [HistogramIterationListener](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/HistogramIterationListener.java) ([see also](http://deeplearning4j.org/visualization.html)) and others. DL4J's user interface functionality is based on [Dropwizard](http://www.dropwizard.io/), [FreeMarker](http://freemarker.incubator.apache.org/) and [D3](http://d3js.org/). In short, these components allow the UI Javascript code to use the outputs of DL4J while your network is training.




### Adding New Layers to DL4J

Suppose you want to add a new type of layer to DL4J. Here's what you need to know.

First, network configuration and network implementation (i.e., the math) are separated. Confusingly, they are both called layer:

* [org.deeplearning4j.nn.api.Layer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Layer.java) for the implementation, and
* [org.deeplearning4j.nn.conf.layers.Layer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/layers/Layer.java) for the configuration

Now, to implement a new layer type, you need to implement all of the following:

* A Layer (configuration) class, with a Builder class. Follow the design of [these classes](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/layers)
* A Layer (implementation) class. Again, follow the design of [these classes](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers)
* A [ParameterInitializer](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params) for your layer (which is responsible for initializing the initial parameters, given the configuration)
* A [LayerFactory](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/factory) which extends DefaultLayerFactory, plus add your layer to DefaultLayerFactory.getInstance()

In DL4J, we do not currently have symbolic automatic differentiation. This means that both the forward pass (predictions) and backward pass (backpropagation) code must be implemented manually.

Some other things you should be aware of:

* DL4J has a numerical gradient checking utility [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java) with unit tests using this [here](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck).
  * The idea behind numerical gradient checks is to check that all gradients, calculated analytically (i.e., in your Layer) are approximately the same as those calculated numerically. For more info, see [this javadoc](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java)
  * Gradient checks are necessary for any new layer type
* Parameters and gradients (as discussed in the next section) are flattened to a row vector. It is important that both the parameters and the gradients are flattened in the same order. In practice, this usually comes down to the order in which you add your gradients to your [Gradient object](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/gradient) vs. the order in which layer parameters are flattened into a row vector (i.e., [Model.params()](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Model.java)). This is one common reason for gradient checks failing.

### How Backpropagation is Implemented in DL4J

So, backprop. Let's start with the basics - an overview of the classes you need to know about:

* MultiLayerNetwork: a neural network with multiple layers
* Layer: a single layer
* Updater: for example, AdaGrad, momentum and RMSProp.
* Optimizer: An abstraction that allows DL4J to support stochastic gradient descent, conjugate gradient, LBFGS, etc in conjunction with any other options (or updaters)

Now, let's step through what happens when you call MultiLayerNetwork.fit(DataSet) or MultiLayerNet.fit(DataSetIterator). We'll assume that we are doing backprop (not unsupervised pretraining).

1. First: the MultiLayerNetwork inputs and outputs (both INDArrays) are set
2. A [Solver](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/Solver.java) object is created if one does not exist
3. Solver.optimize() is called. This calls [ConvexOptimizer.optimize()](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/api/ConvexOptimizer.java)
	What exactly is a ConvexOptimizer? Well, this is the abstraction we use to implement multiple optimization algorithms, including [StochasticGradientDescent](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/StochasticGradie), [LineGradientDescent](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/LineGradientDesc), [ConjugateGradient](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/ConjugateGradient.java) and [LBFGS](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/LBFGS.java)
    Note that each of these ConvexOptimizer classes extend [BaseOptimizer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/BaseOptimizer.java)
    For the next step, let's assume we are using StochasticGradientDescent.
4. StochasticGradientDescent.optimize(): this does two things. First: it initates the calculation of the gradients, by calling BaseOptimizer.GradientAndScore(). Second: it applies the updates to the parameters.
5. BaseOptimizer.gradientAndScore():
  * Calls MultiLayerNetwork.computeGradientAndScore() - this calculates the gradients, then:
  * Calls BaseOptimizer.updateGradientAccordingToParams() - this applies things like learning rates, adagrad, etc
6. Back in StochasticGradientDescent: both the updates (i.e., post-modification gradient) and the network parameters are flattened into a 1d row vector. They are then added, network parameters are set
7. Done! The network parameters have been updated, and we return to do the process


Now, we glossed over two important components: the calculation of the gradients, and their updating/modification.


**Gradient Calculations**
Picking up at MultiLayerNetwork.computeGradientAndScore():

* The MultiLayerNetwork first does a full forward pass through the network, using the input set earlier
  * Ultimately, this ends up calling the [Layer.activate(INDArray,boolean)](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Layer.java#L200-200) method for each layer from the input to the output of the network.
  * Within each layer, the input activations are stored. These are needed during backprop.
* Then, the MultiLayerNetwork does the backward pass through the network, from the OutputLayer to the input.
  * MultiLayerNetwork.calcBackpropGradients(INDArray,boolean) gets called
  * Gradient calculations start with the OutputLayer, which calculates gradients based on the network predictions/output, the labels, and the loss function set in the configuration, [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/BaseOutputLayer.java)
  * Then, gradients are calculated progressively for each layer, using the error from the layer above.
  * Ultimately, the MultiLayerNetwork.gradient field is set, which is essentially a ```Map<String,INDArray>``` containing the gradients for each layer. This is later retrieved by the optimizer.


**Updating Gradients**
Updating gradients involves going from gradients for each parameter, to updates for each parameter. An 'update' is the gradients after we apply things like learning rates, momentum, L1/L2 regularization, gradient clipping and division by the minibatch size.

This functionality is implemented in [BaseUpdater](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/BaseUpdater.java), and the [various updater classes](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater).







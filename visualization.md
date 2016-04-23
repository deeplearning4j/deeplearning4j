---
title: How to Visualize, Monitor and Debug Neural Network Learning
layout: default
---

# Visualize, Monitor and Debug Network Learning

The deeplearning4j-ui repository can display T-SNE, histograms, filters, error and activations. 

Contents

* [Visualizing Network Training with HistogramIterationListener](#histogram)
* [Using the UI to Tune Your Network](#usingui)
* [TSNE and Word2Vec](#tsne)


## <a name="histogram">Visualizing Network Training with HistogramIterationListener</a>

DL4J provides the HistogramIterationListener as a method of visualizing in your  browser (in real time) the progress of network training. You can add a histogram iteration listener using the following code:


    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new HistogramIterationListener(1));

Once network training starts (specifically, after the first parameter update) your browser should automatically open to display the network. (If the browser does not open: check the console output for an address, and go to that address manually.)

Once the UI opens, you will be presented with a display as follows:


![Alt text](../img/DL4J_UI.png)
(Versions of DL4J before 0.4-rc3.9 will have a less sophisticated display).


There are four components to this display:

- Top left: score (loss function) of the current minibatch vs. iteration
- Top right: histogram of parameter values, for each parameter type
- Bottom left: histogram of updates (updates are the gradient values after applying learning rate, momentum, etc)
- Bottom right: line chart of the absolute value of parameters and updates, for each parameter type

*Note: parameter(param) refers to weights(W) and biases(b) and the number tells you what layer it is related to.*

## <a name="usingui">Using the UI to Tune Your Network</a>

Here's an excellent [web page by Andrej Karpathy](http://cs231n.github.io/neural-networks-3/#baby) about visualizing neural net training.

Here's some basics to get you started:

- Score vs. iteration should (overall) go down over time. If the score increases, your learning rate may be set too high
- The histograms of parameters (top right) and the updates (bottom left) should  have an approximately Gaussian (normal) distribution. Keep an eye out for parameters that are diverging to +/- infinity: this may be due to too high a learning rate, or insufficient regularization (try adding some L2 regularization to your network).
- For tuning the learning rate, the ratio of parameters to updates should be somewhere in the order of 1000:1 - but note that is a rough guide only, and may not be appropriate for all networks.
  - If the ratio diverges significantly from this, your parameters may be too unstable to learn useful features, or may change too slowly to learn useful features
  - To change this ratio, adjust your learning rate (or sometimes, parameter initialization). In some networks, you may need to set the learning rate differently for different layers.


## <a name="tsne">TSNE and Word2vec</a>

We rely on [TSNE](https://lvdmaaten.github.io/tsne/) to reduce the dimensionality of [word feature vectors](../word2vec.html) and project words into a two or three-dimensional space. Here's some code for using TSNE with Word2Vec:

        log.info("Plot TSNE....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);

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
(Versions of DL4J before 0.4-rc3.9 will have a less sophisticated display, but contains the same information).


There are four components to this display:

- Top left: score (loss function) of the current minibatch vs. iteration
- Top right: histogram of parameter values, for each parameter type
- Bottom left: histogram of updates (updates are the gradient values after applying learning rate, momentum, etc)
- Bottom right: line chart of the absolute value of parameters and updates, for each parameter type

*Note: parameter(param) refers to weights(W) and biases(b) and the number tells you what layer it is related to. For recurrent neural networks, W refers to the weights connecting the layer to the layer below, and RW refers to the recurrent weights (i.e., those between time steps).*

## <a name="usingui">Using the UI to Tune Your Network</a>

Here's an excellent [web page by Andrej Karpathy](http://cs231n.github.io/neural-networks-3/#baby) about visualizing neural net training. It is worth reading that page first.

Tuning neural networks is often more an art than a science. However, here's some ideas that may be useful:

**Using the score vs. iteration graph: (top left)**

- Score vs. iteration should (overall) go down over time
    - If the score increases consistently, your learning rate is likely set too high. Try reducing it until scores become more stable.
    - Increasing scores can also be indicative of other network issues, such as incorrect data normalization
    - If the score is flat or decreases very slowly (over a few hundred iteratons) (a) your learning rate may be too low, or (b) you might be having diffulties with optimization. In the latter case, if you are using the SGD updater, try a different updater such as momentum, RMSProp or Adagrad.
    - Note that data that isn't shuffled (i.e., each minibatch contains only one class, for classification) can result in very rough or abnormal-looking score vs. iteration graphs
- Some noise in this line chart is expected (i.e., the line will go up and down within a small range). However, if the scores vary quite significantly between runs variation is very large, this can be a problem
    - The issues mentioned above (learning rate, normalization, data shuffling) may contribute to this.
    - Setting the minibatch size to a very small number of examples can also contribute to noisy score vs. iteration graphs, and *might* lead to optimization difficulties

**Using the histogram of parameters graphs: (top right)**

- At the top right is a histogram of the weights in the neural network (at the last iteration), split up by layer and the type of parameter. For example, "param_0_W" refers to the weight parameters for the first layer.
- For weights, these histograms should  have an approximately Gaussian (normal) distribution, after some time
- For biases, these histograms will generally start at 0, and will usually end up being approximately Gaussian
    - One exception to this is for LSTM recurrent neural network layers: by default, the biases for one gate (the forget gate) are set to 1.0 (by default, though this is configurable), to help in learning dependencies across long time periods. This results in the bias graphs initially having many biases around 0.0, with another set of biases around 1.0
- Keep an eye out for parameters that are diverging to +/- infinity: this may be due to too high a learning rate, or insufficient regularization (try adding some L2 regularization to your network).
- Keep an eye out for biases that become very large. This can sometimes occur in the output layer for classification, if the distribution of classes is very imbalanced

**Using the histogram of gradients (updates) graphs: (bottom left)**

- At the bottom left is the histogram of updates for the neural network (at the last iteration), also split up by layer and type of parameter
    - Note that these are the updates - i.e., the gradients *after* appling learning rate, momentum, regularization etc
- As with the parameter graphs, these should have an approximately Gaussian (normal) distribution
- Keep an eye out for very large values: this can indicate exploding gradients in your network
    - Exploding gradients are problematic as they can 'mess up' the parameters of your network
    - In this case, it may indicate a weight initialization, learning rate or input/labels data normalization issue
    - In the case of recurrent neural networks, adding some [gradient normalization or gradient clipping](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/GradientNormalization.java) can frequently help

**Using the mean magnitudes over time graphs: (bottom right)**

- At the bottom right is a line chart of the mean magnitude of both the parameters and the updates in the neural network
    - "Mean magnitude" = the average of the absolute value of the parameters or updates
- For tuning the learning rate, the ratio of parameters to updates for a layer should be somewhere in the order of 1000:1 - but note that is a rough guide only, and may not be appropriate for all networks. It's often a good starting point, however.
  - If the ratio diverges significantly from this, your parameters may be too unstable to learn useful features, or may change too slowly to learn useful features
  - To change this ratio, adjust your learning rate (or sometimes, parameter initialization). In some networks, you may need to set the learning rate differently for different layers.
- Keep an eye out for unusually large spikes in the updates: this may indicate exploding gradients (see discussion in the "histogram of gradients" section above)


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

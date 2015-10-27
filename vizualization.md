---
title: 
layout: default
---

# Deeplearning4j's Vizualization and UI

To support visualizations that will help you monitor neural networks as they learn, and therefore debug them, you must set up an iteration listener. This is done when you instantiate and initialize a new MultiLayerNetwork.

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

The first line above passes a configuration you will have specified previously into an instance of a MultiLayerNetwork model. The second initializes the model. The third sets iteration listeners. Remember, an iteration is simply one update of a network's weights: you may decide to update the weights after a batch of examples is processed, or you may update them after a full pass through the dataset, known as an epoch.

An *iterationListener* is a hook, a plugin, which monitors the iterations and reacts to what's happening. 

A typical pattern for an iterationListener would be asking it to do something every two to five iterations. For example, you might ask it to print the error associated with your net's latest guess. You might ask it to plot either the latest weight distribution or the latest reconstructions your RBM imagines match the input data or the activations in the net itself. In addition, an iterationListener logs activity associated with the iteration, and helps you debug. 

        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

In this line of code, the ScoreIterationListener is passed the parameter specifying a number of iterations -- let's say you specify two -- and after every two iterations, it will print out the error or cost. (The higher the frequency, the more you slow things down).

## UI

M deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/HistogramIterationListener.java 
M deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/ModelAndGradient.java 
M deeplearning4j-ui/src/main/resources/org/deeplearning4j/ui/weights/render.ftl 

https://github.com/deeplearning4j/deeplearning4j/blob/9ca18d8f0b4828a55f381d50e32b6eebcb3444e0/deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/HistogramIterationListener.java#L35-34

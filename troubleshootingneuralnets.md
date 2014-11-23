---
title: 
layout: default
---

#Troubleshooting Neural Net Training

Neural networks are notoriously difficult to tune. When training with any neural net framework, you want to be able to understand how to use it. Below are some baseline steps you should take when tuning your network to train:

##Data

What's distribution of your data? Are you scaling it properly? With Deeplearning4j, there are a different scaling and normalization techniques to keep in mind.

      DataSet d = ...;
      //zero mean and unit variance
      d.normalizeZeroMeanAndUnitVariance();
      Scale between 0 and 1:
      d.scale();
      or:
      d.scaleMinAndMax(0,1);

The data transforms that you'll perform are relative to the problem you're solving, and will vary according to the data. Let's consider a configuration now:
 
       List<NeuralNetConfiguration> conf = new NeuralNetConfiguration.Builder()
	    .iterations(1)
	    .weightInit(WeightInit.DISTRIBUTION)
	    .dist(Distributions.normal(gen, 1e-2))
	    .activationFunction(Activations.tanh())
	    .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
	    .hiddenUnit(RBM.HiddenUnit.RECTIFIED).constrainGradient
	    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
	    .optimizationAlgo(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
	    .rng(gen)
	    .learningRate(1e-2f)
	    .nIn(4).nOut(3).list(2)
	    .override(new NeuralNetConfiguration.ConfOverride() {
            @Override
            public void override(int i, NeuralNetConfiguration.Builder builder) {
            if (i == 1) {
                builder.weightInit(WeightInit.ZERO);
                builder.activationFunction(Activations.softMaxRows());
                builder.lossFunction(LossFunctions.LossFunction.MCXENT);
              }
            }
        }).build();
        
        DBN d = new DBN.Builder().layerWiseConfiguration(conf)
        .hiddenLayerSizes(new int[]{3}).build();

There's a lot going on here -- I'll cover each of the facets of the configuration and how it relates to troubleshooting.

## weightInit

Deeplearning4j supports several different kinds of weight initializations with the weightInit parameter.

You need to make sure your weights are neither too big nor too small. If you're running a classifier, ensure that your weights are initialized to zero on the output layer (as with the above example).

##Iterations

Note the number of iterations you're training on. Early stopping can help prevent neural-network weights from diverging and will help a net generalize better.

##Learning Rate

The learning rate is either used as the master step size in Adagrad (default) or the step size used for calculating your gradient. With continuous data, you will want a smaller learning rate. With binary, you can often go up to a rate of 1e-1.

If, when visualizing weights and gradients, you notice the magnitudes are too high, try shrinking the learning rate or the weights.

##Activation Function

Different data has different optimal activation functions. 

Using tanh, you can get a bounded range of -1,1 -- and with a combination of zero mean and unit variance, that may be a good nonlinear transform for the data. When dealing with binary data, sigmoid is better.

##Visible/Hidden Unit

When using a deep-belief network, pay close attention here. An RBM (the component of the DBN used for feature extraction) is stochastic and will sample from different probability distributions relative to the visible or hidden units specified. 

See Geoff Hinton's definitive work, [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), for a list of all of the different probability distributions.

##Loss Function

Loss functions for each neural network layer can either be used in pretraining, to learn better weights, or in classification (on the output layer) for achieving some result. (In the example above, classification happens in the override section.)

Your net's purpose will determine the loss funtion you use. For pretraining, choose reconstruction entropy. For classification, use multiclass cross entropy.

##Gradient Constraining

When training a neural network, it can sometimes be in your interest to constrain the gradient to unit norm. This will prevent overfitting and has a very similar effect to regularization. If your neural network isn't working, this can help. However, over penalizing can also prevent learning from happening. I would advise enabling this when the magnitude of your gradient is large (this can be seen if you add render(n) to your neural network configuration)
or the neural network at the beginning (due to high weights among other things) has near all white activiations (the grey/white/black plot you see pop up after the histograms).

You can enable this by adding:
 
      .constrainGradientToUnitNorm(true)


to your configuration.


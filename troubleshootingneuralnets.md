---
title: 
layout: default
---

###Troubleshooting baseline Neural Network Training

Neural networks are notoriously difficult to tune.

When training with any neural net framework, you want to be able to understand how to use the framework.

Below are some baseline steps you should take when tuning your network for training:

###Data

What distribution is your data? Are you scaling it properly? In deeplearning4j, there are a few different scaling/normalization 
techniques to be aware of.

```
DataSet d = ...;
//zero mean and unit variance
d.normalizeZeroMeanAndUnitVariance();
Scale between 0 and 1:
d.scale();
or:
d.scaleMinAndMax(0,1);
```

The data transforms you should be doing are relative to the problem you're solving and what works best with your data.


Let's consider a configuration now:


```
   List<NeuralNetConfiguration> conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen, 1e-2))
                .activationFunction(Activations.tanh())
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .optimizationAlgo(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
                .rng(gen)
                .learningRate(1e-2f)
                .nIn(4).nOut(3).list(2).override(new NeuralNetConfiguration.ConfOverride() {
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
```

There's a lot going on here. I will cover each of the facets of the conf and how it relates to trouble shooting:


##weightInit:

Deep learning4j supports several different kinds of weight initializations with the weightInit knob.

With neural network weights, be sure your weights aren't too big/too small.

If you are running a classifier, ensure your weights are initialized to zero on the output layer (an example of this above)

##Iterations:

Note the number of iterations you are training on. Early stopping can help prevent neural network weights
from diverging and generalize better.

##Learning Rate:

Learning rate is either used as the master step size in adagrad (default) or the step size used for calculating
your gradient. With continuous data, you will want a smaller learning rate. With binary, you can often go up to 1e-1.

If you notice your magnitude is too high when you are visualizing weights and gradients, you may either want to shrink
your learning rate or your weights.

##Activaion Function:

Depending on your data, different activation functions will tend to be better. If you use tanh, you can 
get a bounded range of -1,1 and with a combination of zero mean and unit variance can give you a good
non linear transform of your data.

If you are dealing with binary data sigmoid will be better.

##Visible/Hidden Unit

If you are using a DBN, you will want to pay attention to this. An RBM (the component of the DBN used
for feature extraction) is stochastic and will sample from different probability distributions
relative to the visible or hidden units specified. See [Practical Guide to RBMs](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
for a list of all of the different probability distributions and the like.

##Loss Function

Loss Functions for each neural network layer can either be used in pretraining (to learn better weights)
or in classification (on the output layer) for achieving some result. (I do the classification above in the override part)

The loss function you specify is going to be relative to your objective.

For pretraining, it is good to leave it at reconstruction entropy.

For classification, use multi class cross entropy.




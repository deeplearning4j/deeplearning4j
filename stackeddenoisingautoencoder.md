---
title: Stacked Denoising AutoEncoders
layout: default
---

# Stacked Denoising Autoencoders

A stacked denoising autoencoder is to a denoising autoencoder what a [deep-belief network](/deepbeliefnetwork.html) is to a [restricted Boltzmann machine](../restrictedboltzmannmachine.html). A key function of SDAs, and deep learning more generally, is unsupervised pre-training, layer by layer, as input is fed through. Once each layer is pre-trained to conduct feature selection and extraction on the input from the preceding layer, a second stage of supervised fine-tuning can follow. 

A word on stochastic corruption in SDAs: Denoising autoencoders shuffle data around and learn about that data by attempting to reconstruct it. The act of shuffling is the noise, and the job of the network is to recognize the features within the noise that will allow it to classify the input. When a network is being trained, it generates a model, and measures the distance between that model and the benchmark through a loss function. Its attempts to minimize the loss function involve resampling the shuffled inputs and re-reconstructing the data, until it finds those inputs which bring its model closest to what it has been told is true. 

The serial resamplings are based on a generative model to randomly provide data to be processed. This is known as a Markov Chain, and more specifically, a Markov Chain Monte Carlo algorithm that steps through the data set seeking a representative sampling of indicators that can be used to construct more and more complex features.

In Deeplearning4j, stacked denoising autoencoders are built by creating a `MultiLayerNetwork` that has autoencoders for its hidden layers. Those autoencoders have a `corruptionLevel`. That's the "noise"; the neural network learns to denoise the signal. Notice how `pretrain` is set to "true".

By the same token, deep-belief networks are created as a `MultiLayerNetwork` that has restricted Boltzmann machines at each hidden layer. More generally, you can think of Deeplearning4j as usuing "primitives" such as RBMs and autoencoders that allow you to construct various deep neural networks.

## Just Give Me the Code


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(4)
           .layer(0, new AutoEncoder.Builder().nIn(numRows * numColumns).nOut(500)
                   .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                   .corruptionLevel(0.3)
                   .build())
                .layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)

                        .build())
                .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
                .build();


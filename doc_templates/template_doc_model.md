---
title: Template for Model (feature) content. 
Note: Your changes can be viewed by anyone using this document’s public sharing link.
layout: default
---

# Title_of_the_Doc

Template for Model Doc Content.

This appears to be based on a topic description of an existing Model. 
Content listed is for example purposes. Headings are probably the primary intent.

# Learning Rate Policy

## Contents

### <a href="#description">Description</a>
### <a href="#examples"> Examples and Use Cases</a>
### <a href="#setup"> How It Works</a>
   <b><a href="#prereqs"> 0. Prerequisites</a></b>
   
   <b><a href="#step1"> 1. Configure your neural network to use a LearningRatePolicy</a></b>
   
   <b><a href="#step2"> 2. Train your network to use a LearningRatePolicy</a></b>
### <a href="#troubleshooting"> Troubleshooting</a>
### <a href="#further"> Further Reading</a>

## <a name="description">Description</a>

LearningRatePolicy provides decay alternatives during training. [[Explain what decay alternatives are for.]]

## <a name="examples">Examples and Use Cases</a>

<b>Exponential</b>

[[Brief description]]
```
double lr = 1e-2;
double decayRate = 2;
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
.learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Exponential)
.lrPolicyDecayRate(decayRate).iterations(iterations)
.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
.updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
.build();
```

<b>Inverse</b>

[[Brief description]]
```
double lr = 1e-2;
double decayRate = 2;
double power = 3;
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Inverse)
.lrPolicyDecayRate(decayRate).lrPolicyPower(power).iterations(iterations)
.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
.updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
.build();
```

<b>Poly</b>

[[Brief description]]
```
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Poly).lrPolicyPower(power)
.iterations(iterations)
.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)    
.updater(org.deeplearning4j.nn.conf.Updater.SGD).build())                                      
.build();
```

<b>Sigmoid</b>

[[Brief description]]
```
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Sigmoid)
.lrPolicyDecayRate(decayRate).lrPolicySteps(steps).iterations(iterations)
.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
.updater(org.deeplearning4j.nn.conf.Updater.SGD).build())                                
.build();
```

<b>Step</b>

[[Brief description]]
```
double lr = 1e-2;
double decayRate = 2;
double steps = 3;
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(decayRate)
.lrPolicySteps(steps).iterations(iterations)
.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
.updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
.build();
```

<b>Schedule</b>

Allows you to specify a schedule [[explain]].
```
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
.learningRateDecayPolicy(LearningRatePolicy.Schedule)                            
.learningRateSchedule(learningRateSchedule)
```

<b>Score</b>

[[Brief description]]
```
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
.learningRate(lr)
.learningRateDecayPolicy(LearningRatePolicy.Score).lrPolicyDecayRate(lrScoreDecay).list()
```


## <a name="setup">How It Works</a>

   <b><a href="#prereqs">0. Prerequisites</a></b>
   
   <b><a href="#step1">1. Configure your neural network to use a LearningRatePolicy</a></b>
   
   <b><a href="#step2">2. Train your network to use a LearningRatePolicy</a></b>


### <a name="prereqs">0. Prerequisites</a>

Versions 0.7.0 and above support LearningRateDecayPolicy

### <a name="step1">1. Configure your neural network to use a LearningRatePolicy</a>

[[Describe the process of configuration.]]

### <a name="step2">2. Train your network to use a LearningRatePolicy</a>

[[Describe the process of training.]]


## <a name="#troubleshooting">Troubleshooting</a>

Avoid choosing values that would cause the training to take longer than it would with a standard updater. Verify that your network is making reasonable progress. [[Explain/give examples. e.g., Here's what a neural net behaves like when the learning rate is too high vs. too low.]] 

## <a name="#further">Further Reading</a>

<b>[Tutorial on Keras implemetation of learning rate schedules](http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras "Tutorial on Keras implemetation of learning rate schedules")</b>

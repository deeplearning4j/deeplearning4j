---
title: 
layout: default
---

# Recurrent Neural Networks in DL4J

This document outlines the specifics training features and the practicalities of how to use them in DeepLearning4J. This document assumes some familiarity with recurrent neural networks and their use - see  It is not an introduction to recurrent neural networks, and assumes some familiarity with their use and terminology. If you are new to RNNs, see [A Beginner's Guide to Recurrent Networks and LSTMs](/lstm.html) for an introduction.


Contents
* [The Basics: Network Configuration, Data and Learning](#basics)
* [RNN Training Features](#trainingfeatures)
 * [Truncated Back Propagation Through Time](#tbptt)
 * [Masking: One-to-Many, Many-to-One, and Sequence Classification](#masking)
 * [Combining RNN Layers with Other Layer Types](#otherlayertypes)
* [Test Time: Prediction One Step at a Time](#rnntimestep)
* [Importing Time Series Data](#data)
* [Examples](#examples)

## <a name="basics">The Basics: Network Configuration, Data and Fitting</a>
DL4J currently supports one main type of recurrent neural network – the LSTM (Long Short-Term Memory) model, though many more are planned for the future.

#### Data for RNNs
Consider for the moment a standard feed-forward network (a multi-layer perceptron or 'DenseLayer' in DL4J). These networks expect input and output data that is two-dimensional: that is, data with "shape" [numExamples,inputSize]. This means that the data into a feed-forward network has ‘numExamples’ rows/examples, where each row each row consists of ‘inputSize’ columns. A single example would have shape [1,inputSize]; in practice, we generally use multiple examples for computational and optimization efficiency. Similarly, output data is also two dimensional, with shape [numExamples,outputSize].

Conversely, data for RNNs are time series. Thus, they have 3 dimensions: one additional dimension for time. Input data thus has shape [numExamples,inputSize,timeSeriesLength], and output data has shape [numExamples,outputSize,timeSeriesLength]. This means that the data in our INDArray is laid out such that the value at position (i,j,k) is the jth value at the kth time step of the ith example in the minibatch.

#### RnnOutputLayer
RnnOutputLayer is a class used with many recurrent neural network systems. RnnOutputLayer is a type of layer in DL4J that handles things like score calculation, and error calculation (of prediction vs. actual) given a loss function etc. Functionally, is is very similar to the 'standard' OutputLayer (which is used with feed-forward networks); however it both outputs (and expects as labels/targets) 3d time series data sets.



#### Network Configuration


## <a name="trainingfeatures">RNN Training Features</a>

### <a name="tbptt">Truncated Back Propagation Through Time</a>
Training neural networks (including RNNs) can be quite computationally demanding. For recurrent neural networks, this is especially the case when we are dealing with long sequences - i.e., training data with many time steps.


Truncated backpropagation through time (BPTT) was developed in order to reduce the computational complexity of each parameter update in a recurrent neural network. In summary, it allows us to train networks faster (by performing more frequent parameter updates), for a given amount of computational power. It is recommended to use truncated BPTT when your input sequences are long (typically, more than a few hundred time steps).


Consider what happens when training a recurrent neural network with a time series of length 12 time steps. Here, we need to do a forward pass, compare the predictions, and do a backward pass:
![Standard Backprop Training](../img/rnn_tbptt_1.png)

For 12 time steps, in the example above, this is not a problem. Consider, however, that instead the input time series was 10,000 or more time steps. In this case, standard backpropagation through time would require 10,000 time steps - something very computationally demanding.

In practice, truncated BPTT splits the forward and backward passes into a set of smaller forward/backward pass operations. The specific length of these forward/backward pass segments is a parameter set by the user. For example, if we use truncated BPTT of length 4 time steps, learning looks like the following:

![Truncated BPTT](../img/rnn_tbptt_2.png)

Note that the overall complexity for truncated BPTT and standard BPTT are approximately the same - both do the same number of time step during forward/backward pass. However, the cost is not exactly the same there is a small amount of overhead per parameter update.

The downside of truncated BPTT is that the length of the dependencies learned in truncated BPTT can be shorter than in full BPTT. This is easy to see: consider the images above, with a TBPTT length of 4. Suppose that at time step 10, the network needs to store some information from time step 0. In standard BPTT, this is ok: the gradients can flow backwards all the way along the unrolled network. In truncated BPTT, this is problematic: the gradients from time step 10 simply don't flow back far enough to cause the required parameter updates that would store the required information.

Using truncated BPTT in DL4J is quite simple: just add the following code to your network configuration (at the end, before the final .build() on your configuration)

```java
.backpropType(BackpropType.TruncatedBPTT)
.tBPTTForwardLength(100)
.tBPTTBackwardLength(100)
```
The above code snippet will cause any networ training (i.e., calls to MultiLayerNetwork.fit() methods) to use truncated BPTT with equal length forward and backward passes, of length 100.

Some things of note:
* By default (if a backprop type is not manually specified), DL4J will use BackpropType.Standard (i.e., full BPTT).
* The tBPTTForwardLength and tBPTTBackwardLength options set the length of the truncated BPTT passes. Typically, this is somewhere on the order of 50 to 200 time steps, though depends on the application. Typically both forward pass and backward pass will be the same length (though tBPTTBackwardLength may be shorter, but not longer)
* The truncated  BPTT lengths must be shorter than or equal to the total time series length



### <a name="masking">Masking: One-to-Many, Many-to-One, and Sequence Classification</a>

DL4J supports a number of related training features for RNNs, based on the idea of padding and masking. Padding and masking allows us to support training situations including one-to-many, many-to-one, as also support variable length time series (in the same mini-batch).

Suppose we want to train a recurrent neural network with inputs or outputs that don't occur at every time step. Examples of this (for a single example) are shown in the image below. DL4J supports all of these situations:
![RNN Training Types](../img/rnn_masking_1.png)

Without masking and padding, we are restricted to the many-to-one case (above, left): that is, (a) All examples are of the same length, and (b) Examples have both inputs and outputs at all time steps.

The idea behind padding is simple. Consider two time series of lengths 50 and 100 time steps, in the same mini-batch. The training data is a rectangular array; thus, we pad (i.e., add zeros to) the shorter time series (for both input and output), such that the input and output are both the same length (in this example: 100 time steps).
Of course, if this was all we did, it would cause problems during training. Thus, in addition to padding, we use a masking mechanism. The idea behind masing is simple: we have an additional two arrays that record an input or output is actually present for a given time step and example, or whether the input/output is just padding.

Recall that with RNNs, our minibatch data has 3 dimensions, with shape [miniBatchSize,inputSize,timeSeriesLength] and [miniBatchSize,outputSize,timeSeriesLength] for the input and output respectively. The padding arrays are then 2 dimensional, with shape [miniBatchSize,timeSeriesLength] for both the input and output, with values of 0 ('absent') or 1 ('present') for each time series and example. The masking arrays for the input and output are stored in separate arrays.

For a single example, the input and output masking arrays are shown below:
![RNN Training Types](../img/rnn_masking_2.png)

For the “Masking not required” cases, we could equivalently use a masking array of all 1s, which will give the same result as not having a mask array at all. Also note that it is possible to use zero, one or two masking arrays when learning RNNs - for example, the many-to-one case could have a masking array for the output only.

In practice: these padding arrays are generally created during the data import stage (for example, by the SequenceRecordReaderDatasetIterator – discussed later), and are contained within the DataSet object. If a DataSet contains masking arrays, the MultiLayerNetwork fit will automatically use them during training. If they are absent, no masking functionality is used.



#### Evaluation and Scoring with Masking
Mask arrays are also important when doing scoring and evaluation (i.e., when evaluating the accuracy of a RNN classifier). Consider for example the many-to-one case: there is only a single output for each example, and any evaluation should take this into account.

Evaluation using the (output) mask arrays can be used during evaluation by passing it to the following method:
```java
Evaluation.evalTimeSeries(INDArray labels, INDArray predicted, INDArray outputMask)
```
where labels are the actual output (3d time series), predicted is the network predictions (3d time series, same shape as labels), and outputMask is the 2d mask array for the output. Note that the input mask array is not required for evaluation.

Score calculation will also make use of the mask arrays, via the MultiLayerNetwork.score(DataSet) method. Again, if the DataSet contains an output masking array, it will automatically be used when calculating the score (loss function - mean squared error, negative log likelihood etc) for the network.


### <a name="otherlayertypes">Combining RNN Layers with Other Layer Types</a>

(TODO)


## <a name="rnntimestep">Test Time: Predictions One Step at a Time</a>
As with other types of neural networks, predictions can be generated for RNNs using the MultiLayerNetwork.output() and MultiLayerNetwork.feedForward() methods. These methods can be useful in many circumstances; however, they have the limitation that we can only generate predictions for time series, starting from scratch each and every time.

Consider for example the case where we want to generate predictions in a real-time system, where these predictions are based on say 10,000 (or more) time steps of history. It this case, it is impractical to use the output/feedForward methods, as they conduct the full forward pass over the entire data history, each time they are called. If we wish to make a prediction for a single time step, at every time step, these methods can be both (a) very costly, and (b) wasteful, as they do the same calculations over and over.

For these situations, MultiLayerNetwork provides 4 methods of note:
* rnnTimeStep(INDArray)
* rnnClearPreviousState()
* rnnGetPreviousState(int layer)
* rnnSetPreviousState(int layer, Map<String,INDArray> state)

The rnnTimeStep() method is designed to allow forward pass (predictions) to be conducted efficiently, one or more steps at a time. Unlike the output/feedForward methods, the rnnTimeStep method keeps track of the internal state of the RNN layers when it is called. It is important to note that output for the rnnTimeStep and the output/feedForward methods should be identical (for each time step), whether we make these predictions all at once (output/feedForward) or whether these predictions are generated one or more steps at a time (rnnTimeStep). Thus, the only difference should be the computational cost.

In summary, the MultiLayerNetwork.rnnTimeStep method does two things:
1.	Generate output/predictions (forward pass), using the previous stored state (if any)
2.	Update the stored state, storing the activations for the last time step (ready to be used next time rnnTimeStep is called)

For example, suppose we want to use a RNN to predict the weather, one hour in advance (based on the weather at say the previous 100 hours as input).
If we were to use the output method, at each hour we would need to feed in the full 100 hours of data to predict the weather for hour 101. Then to predict the weather for hour 102, we would need to feed in the full 100 (or 101) hours of data; and so on for hours 103+.

Alternatively, we could use the rnnTimeStep method. Of course, if we want to use the full 100 hours of history before we make our first prediction, we still need to do the full forward pass:
![RNN Time Step](../img/rnn_timestep_1.png)

Thus, the only practical difference between the two is that the activations/state of the last time step are stored, when using the rnnTimeStep. This is shown in orange.
However, the next time we use the rnnTimeStep method, this stored state will be used to make the next predictions:
![RNN Time Step](../img/rnn_timestep_2.png)

There are a number of important differences here:
1. The input data when using the rnnTimeStep consists of a single time step, instead of the full history of data
2. The forward pass is thus a single time step (as compared to the hundreds – or more)
3. After the rnnTimeStep method returns, the internal state will automatically be updated. Thus, predictions for time 103 could be made in the same way as for time 102.

However, if you want to start making predictions for a new (entirely separate) time series: it is necessary (and important) to manually clear the stored state, using the MultiLayerNetwork.rnnClearPreviousState() method. This will reset the internal state of the RNN layers.

If you need to store or set the internal state of the RNN for use in predictions, you can use the rnnGetPreviousState and rnnSetPreviousState methods, for each layer individually. This can be useful for example during serialization (network saving/loading), as the internal network state from the rnnTimeStep method is *not* saved by default, and must be saved and loaded separately. Note that these get/set state methods return and accept a map, keyed by the type of activation. For example, in the LSTM model, it is necessary to store both the output activations, and the memory cell state.

Some other points of note:
- We can use the rnnTimeStep method for multiple independent examples/predictions simultaneously. In the weather example above, we might for example want to make predicts for multiple locations using the same neural network. This works in the same way as training and  the forward pass / output methods: multiple rows (dimension 0 in the input data) are used for multiple examples.
- If no history/stored state is set (i.e., initially, or after a call to rnnClearPreviousState), a default initialization (zeros) is used. This is the same approach as during training.
- The rnnTimeStep can be used for an arbitrary number of time steps simultaneously – not just one time step. However, it is important to note:
  - For a single time step: the data is 2 dimensional, with shape [numExamples,nIn]; in this case, the output is also 2 dimensional, with shape [numExamples,nOut]
  - For multiple time steps: the data is 3 dimensional, with shape [numExamples,nIn,numTimeSteps]; the output will have shape [numExamples,nOut,numTimeSteps]. Again, the final time step activations are stored as before.
- It is not possible to change the number of examples between calls of rnnTimeStep (in other words, if the first use of rnnTimeStep is for say 3 examples, all subsequent calls must be with 3 examples). After resetting the internal state (using rnnClearPreviousState()), any number of examples can be used for the next call of rnnTimeStep.
- The rnnTimeStep method makes no changes to the parameters; it is used after training the network has been completed only.
- The rnnTimeStep method works with single and stacked RNN layers, as well as with RNNs that combine other layer types (such as Convolutional or Dense layers).
- The RnnOutputLayer layer type does not have any internal state, as it does not have any recurrent connections.




## <a name="data">Importing Time Series Data</a>

(TODO)


## <a name="examples">Examples</a>

(TODO)
---
title: Deeplearning4j Cheat Sheet
short_title: Cheat Sheet
description: Snippets and links for common functionality in Eclipse Deeplearning4j.
category: Get Started
weight: 2
---

## Quick reference

Deeplearning4j (and related projects) have a lot of functionality. The goal of this page is to summarize this functionality so users know what exists, and where to find more information.

**Contents**

* [Layers](#layers)
    * [Feed-Forward Layers](#layers-ff)
    * [Output Layers](#layers-out)
    * [Convolutional Layers](#layers-conv)
    * [Recurrent Layers](#layers-rnn)
    * [Unsupervised Layers](#layers-unsupervised)
    * [Other Layers](#layers-other)
    * [Graph Vertices](#layers-vertices)
    * [InputPreProcessors](#layers-preproc)
* [Iteration/Training Listeners](#listeners)
* [Evaluation](#evaluation)
* [Network Saving and Loading](#saving)
* [Network Configurations](#config)
    * [Activation Functions](#config-afn)
    * [Weight Initialization](#config-init)
    * [Updaters (Optimizers)](#config-updaters)
    * [Learning Rate Schedules](#config-schedules)
    * [Regularization](#config-regularization)
        * [L1/L2 regularization](#config-l1l2)
        * [Dropout](#config-dropout)
        * [Weight Noise](#config-weightnoise)
        * [Constraints](#config-constraints)
* [Data Classes](#data)
    * [Iterators](#data-iter)
        * [Iterators - Build-In (DL4J-Provided Data)](#data-iter-builtin)
        * [Iterators - User Provided Data](#data-iter-user)
        * [Iterators - Adapter and Utility Iterators](#data-iter-util)
    * [Reading Raw Data: DataVec RecordReaders](#data-datavec)
    * [Data Normalization](#data-norm)
    * [Spark Network Training Data Classes](#data-spark)
* [Transfer Learning](#transfer)
* [Trained Model Library - Model Zoo](#zoo)
* [SKIL - Model Deployment](#skil)
* [Keras Import](#keras)
* [Distributed Training (Spark)](#spark)
* [Hyperparameter Optimization (Arbiter)](#arbiter)

### <a name="layers">Layers</a>

### <a name="layers-ff">Feed-Forward Layers</a>

* **DenseLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/dense/DenseLayer.java)) - A simple/standard fully-connected layer
* **EmbeddingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/embedding/EmbeddingLayer.java)) - Takes positive integer indexes as input, outputs vectors. Only usable as first layer in a model. Mathematically equivalent (when bias is enabled) to DenseLayer with one-hot input, but more efficient.

#### <a name="layers-out">Output Layers</a>

Output layers: usable only as the last layer in a network. Loss functions are set here.

* **OutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/OutputLayer.java)) - Output layer for standard classification/regression in MLPs/CNNs. Has a fully connected DenseLayer built in. 2d input/output (i.e., row vector per example).
* **LossLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LossLayer.java)) - Output layer without parameters - only loss function and activation function. 2d input/output (i.e., row vector per example). Unlike Outputlayer, restricted to nIn = nOut.
* **RnnOutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnOutputLayer.java)) - Output layer for recurrent neural networks. 3d (time series) input and output. Has time distributed fully connected layer built in.
* **RnnLossLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnLossLayer.java)) - The 'no parameter' version of RnnOutputLayer. 3d (time series) input and output.
* **CnnLossLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CnnLossLayer.java)) - Used with CNNs, where a prediction must be made at each spatial location of the output (for example: segmentation or denoising). No parameters, 4d input/output with shape [minibatch, depth, height, width]. When using softmax, this is applied depthwise at each spatial location.
* **Yolo2OutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/objdetect/Yolo2OutputLayer.java)) - Implentation of the YOLO 2 model for object detection in images
* **CenterLossOutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CenterLossOutputLayer.java)) - A version of OutputLayer that also attempts to minimize the intra-class distance of examples' activations - i.e., "If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y"


#### <a name="layers-conv">Convolutional Layers</a>

* **ConvolutionLayer** / Convolution2D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.java)) - Standard 2d convolutional neural network layer. Inputs and outputs have 4 dimensions with shape [minibatch,depthIn,heightIn,widthIn] and [minibatch,depthOut,heightOut,widthOut] respectively.
* **Convolution1DLayer** / Convolution1D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution1DLayer.java)) - Standard 1d convolution layer
* **Deconvolution2DLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/Deconvolution2DLayer.java)) - also known as transpose or fractionally strided convolutions. Can be considered a "reversed" ConvolutionLayer; output size is generally larger than the input, whilst maintaining the spatial connection structure.
* **SeparableConvolution2DLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/SeparableConvolution2DLayer.java)) - depthwise separable convolution layer
* **SubsamplingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/subsampling/SubsamplingLayer.java)) - Implements standard 2d spatial pooling for CNNs - with max, average and p-norm pooling available.
* **Subsampling1DLayer** - ([Source]())
* **Upsampling2D** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling2D.java)) - Upscale CNN activations by repeating the row/column values
* **Upsampling1D** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling1D.java)) - 1D version of the upsampling layer
* **Cropping2D** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/convolutional/Cropping2D.java)) - Cropping layer for 2D convolutional neural networks
* **ZeroPaddingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.java)) - Very simple layer that adds the specified amount of zero padding to edges of the 4d input activations.
* **ZeroPadding1DLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPadding1DLayer.java)) - 1D version of ZeroPaddingLayer
* **SpaceToDepth** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToDepthLayer.java)) - This operation takes 4D array in, and moves data from spatial dimensions (HW) to channels (C) for given blockSize
* **SpaceToBatch** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToBatchLayer.java)) - Transforms data from a tensor from 2 spatial dimensions into batch dimension according to the "blocks" specified


#### <a name="layers-rnn">Recurrent Layers</a>

* **LSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java)) - LSTM RNN without peephole connections. Supports CuDNN.
* **GravesLSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesLSTM.java)) - LSTM RNN with peephole connections. Does *not* support CuDNN (thus for GPUs, LSTM should be used in preference).
* **GravesBidirectionalLSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesBidirectionalLSTM.java)) - A bidirectional LSTM implementation with peephole connections. Equivalent to Bidirectional(ADD, GravesLSTM). Due to addition of Bidirecitonal wrapper (below), has been deprecated on master.
* **Bidirectional** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/Bidirectional.java)) - A 'wrapper' layer - converts any standard uni-directional RNN into a bidirectional RNN (doubles number of params - forward/backward nets have independent parameters). Activations from forward/backward nets may be either added, multiplied, averaged or concatenated.
* **SimpleRnn** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/SimpleRnn.java)) - A standard/'vanilla' RNN layer. Usually not effective in practice with long time series dependencies - LSTM is generally preferred.
* **LastTimeStep** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/LastTimeStep.java)) - A 'wrapper' layer - extracts out the last time step of the (non-bidirectional) RNN layer it wraps. 3d input with shape [minibatch, size, timeSeriesLength], 2d output with shape [minibatch, size].


#### <a name="layers-unsupervised">Unsupervised Layers</a>

* **VariationalAutoencoder** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational/VariationalAutoencoder.java)) - A variational autoencoder implementation with MLP/dense layers for the encoder and decoder. Supports multiple different types of [reconstruction distributions](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational)
* **AutoEncoder** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/AutoEncoder.java)) - Standard denoising autoencoder layer

#### <a name="layers-other">Other Layers</a>

* **GlobalPoolingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GlobalPoolingLayer.java)) - Implements both pooling over time (for RNNs/time series - input size [minibatch, size, timeSeriesLength], out [minibatch, size]) and global spatial pooling (for CNNs - input size [minibatch, depth, h, w], out [minibatch, depth]). Available pooling modes: sum, average, max and p-norm.
* **ActivationLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ActivationLayer.java)) - Applies an activation function (only) to the input activations. Note that most DL4J layers have activation functions built in as a config option.
* **DropoutLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DropoutLayer.java)) - Implements dropout as a separate/single layer. Note that most DL4J layers have a "built-in" dropout configuration option.
* **BatchNormalization** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/BatchNormalization.java)) - Batch normalization for 2d (feedforward), 3d (time series) or 4d (CNN) activations. For time series, parameter sharing across time; for CNNs, parameter sharing across spatial locations (but not depth).
* **LocalResponseNormalization** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocalResponseNormalization.java)) - Local response normalization layer for CNNs. Not frequently used in modern CNN architectures.
* **FrozenLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayer.java)) - Usually not used directly by users - added as part of transfer learning, to freeze a layer's parameters such that they don't change during further training.


#### <a name="layers-vertices">Graph Vertices</a>

Graph vertex: use with ComputationGraph. Similar to layers, vertices usually don't have any parameters, and may support multiple inputs.

* **ElementWiseVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ElementWiseVertex.java)) - Performs an element-wise operation on the inputs - add, subtract, product, average, max
* **L2NormalizeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2NormalizeVertex.java)) - normalizes the input activations by dividing by the L2 norm for each example. i.e., out <- out / l2Norm(out)
* **L2Vertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - calculates the L2 distance between the two input arrays, for each example separately. Output is a single value, for each input value.
* **MergeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - merge the input activations along dimension 1, to make a larger output array. For CNNs, this implements merging along the depth/channels dimension
* **PreprocessorVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/PreprocessorVertex.java)) - a simple GraphVertex that contains an InputPreProcessor only
*  **ReshapeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ReshapeVertex.java)) - Performs arbitrary activation array reshaping. The preprocessors in the next section should usually be preferred.
* **ScaleVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ScaleVertex.java)) - implements simple multiplicative scaling of the inputs - i.e., out = scalar * input
* **ShiftVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ShiftVertex.java)) - implements simple scalar element-wise addition on the inputs - i.e., out = input + scalar
* **StackVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/StackVertex.java)) - used to stack all inputs along the minibatch dimension. Analogous to MergeVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels)
* **SubsetVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/SubsetVertex.java)) - used to get a contiguous subset of the input activations along dimension 1. For example, two SubsetVertex instances could be used to split the activations from an input array into two separate activations. Essentially the opposite of MergeVertex.
*  **UnstackVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/UnstackVertex.java)) - similar to SubsetVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels). Opposite of StackVertex



### <a name="layers-preproc">Input Pre Processors</a>

An InputPreProcessor is a simple class/interface that operates on the input to a layer. That is, a preprocessor is attached to a layer, and performs some operation on the input, before passing the layer to the output. Preprocessors also handle backpropagation - i.e., the preprocessing operations are generally differentiable.

Note that in many cases (such as the XtoYPreProcessor classes), users won't need to (and shouldn't) add these manually, and can instead just use ```.setInputType(InputType.feedForward(10))``` or similar, which whill infer and add the preprocessors as required.

* **CnnToFeedForwardPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToFeedForwardPreProcessor.java)) - handles the activation reshaping necessary to transition from a CNN layer (ConvolutionLayer, SubsamplingLayer, etc) to DenseLayer/OutputLayer etc.
* **CnnToRnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.java)) - handles reshaping necessary to transition from a (effectively, time distributed) CNN layer to a RNN layer.
* **ComposableInputPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/ComposableInputPreProcessor.java)) - simple class that allows multiple preprocessors to be chained + used on a single layer
* **FeedForwardToCnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnnPreProcessor.java)) - handles activation reshaping to transition from a row vector (per example) to a CNN layer. Note that this transition/preprocessor only makes sense if the activations are actually CNN activations, but have been 'flattened' to a row vector.
* **FeedForwardToRnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToRnnPreProcessor.java)) - handles transition from a (time distributed) feed-forward layer to a RNN layer
* **RnnToCnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToCnnPreProcessor.java)) - handles transition from a sequence of CNN activations with shape ```[minibatch, depth*height*width, timeSeriesLength]``` to time-distributed ```[numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight]``` format
* **RnnToFeedForwardPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToFeedForwardPreProcessor.java)) - handles transition from time series activations (shape ```[minibatch,size,timeSeriesLength]```) to time-distributed feed-forward (shape ```[minibatch*tsLength,size]```) activations.


### <a name="listeners">Iteration/Training Listeners</a>

IterationListener: can be attached to a model, and are called during training, once after every iteration (i.e., after each parameter update).
TrainingListener: extends IterationListener. Has a number of additional methods are called at different stages of training - i.e., after forward pass, after gradient calculation, at the start/end of each epoch, etc.

Neither type (iteration/training) are called outside of training (i.e., during output or feed-forward methods)


* **ScoreIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ScoreIterationListener.java), Javadoc) - Logs the loss function score every N training iterations
* **PerformanceListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/PerformanceListener.java), Javadoc) - Logs performance (examples per sec, minibatches per sec, ETL time), and optionally score, every N training iterations.
* **EvaluativeListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/EvaluativeListener.java), Javadoc) - Evaluates network performance on a test set every N iterations or epochs. Also has a system for callbacks, to (for example) save the evaluation results.
* **CheckpointListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/checkpoint/CheckpointListener.java), Javadoc) - Save network checkpoints periodically - based on epochs, iterations or time (or some combination of all three).
* **StatsListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui-model/src/main/java/org/deeplearning4j/ui/stats/StatsListener.java)) - Main listener for DL4J's web-based network training user interface. See [visualization page](https://deeplearning4j.org/visualization) for more details.
* **CollectScoresIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.java), Javadoc) - Similar to ScoreIterationListener, but stores scores internally in a list (for later retrieval) instead of logging scores
* **TimeIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/TimeIterationListener.java), Javadoc) - Attempts to estimate time until training completion, based on current speed and specified total number of iterations

### <a name="evaluation">Evaluation</a>

Link: [Main evaluation page](https://deeplearning4j.org/evaluation)

DL4J has a number of classes for evaluating the performance of a network, against a test set. Different evaluation classes are suitable for different types of networks.

* **Evaluation** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/Evaluation.java)) - Used for the evaluation of multi-class classifiers (assumes standard one-hot labels, and softmax probability distribution over N classes for predictions). Calculates a number of metrics - accuracy, precision, recall, F1, F-beta, Matthews correlation coefficient, confusion matrix. Optionally calculates top N accuracy, custom binary decision thresholds, and cost arrays (for non-binary case). Typically used for softmax + mcxent/negative-log-likelihood networks.
* **EvaluationBinary** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationBinary.java)) - A multi-label binary version of the Evaluation class. Each network output is assumed to be a separate/independent binary class, with probability 0 to 1 independent of all other outputs. Typically used for sigmoid + binary cross entropy networks.
* **EvaluationCalibration** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationCalibration.java)) - Used to evaluation the calibration of a binary or multi-class classifier. Produces reliability diagrams, residual plots, and histograms of probabilities. Export plots to HTML using [EvaluationTools](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java).exportevaluationCalibrationToHtmlFile method
* **ROC** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROC.java)) - Used for single output binary classifiers only - i.e., networks with nOut(1) + sigmoid, or nOut(2) + softmax. Supports 2 modes: thresholded (approximate) or exact (the default). Calculates area under ROC curve, area under precision-recall curve. Plot ROC and P-R curves to HTML using [EvaluationTools](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java)
* **ROCBinary** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROCBinary.java)) - a version of ROC that is used for multi-label binary networks (i.e., sigmoid + binary cross entropy), where each network output is assumed to be an independent binary variable.  
* **ROCMultiClass** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROCMultiClass.java)) - a version of ROC that is used for multi-class (non-binary) networks (i.e., softmax + mcxent/negative-log-likelihood networks). As ROC metrics are only defined for binary classification, this treats the multi-class output as a set of 'one-vs-all' binary classification problems.
* **RegressionEvaluation** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/RegressionEvaluation.java)) - An evaluation class used for regression models (including multi-output regression models). Reports metrics such as mean-squared error (MSE), mean-absolute error, etc for each output.


## <a name="saving">Network Saving and Loading</a>

MultiLayerNetwork and ComputationGraph can be saved using the [ModelSerializer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/util/ModelSerializer.java) class - and specifically the ```writeModel```, ```restoreMultiLayerNetwork``` and ```restoreComputationGraph``` methods.

As of current master (but not 0.9.1) ```MultiLayerNetwork.save(File)``` and ```MultiLayerNetwork.load(File)``` have been added. These use ModelSerializer internally. Similar save/load methods have also been added for ComputationGraph.

Examples: [Saving and loading network](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving)

Networks can be trained further after saving and loading: however, be sure to load the 'updater' (i.e., the historical state for updaters like momentum, ). If no futher training is required, the updater state can be ommitted to save disk space and memory.

Most Normalizers (implementing the ND4J ```Normalizer``` interface) can also be added to a model using the ```addNormalizerToModel``` method.

Note that the format used for models in DL4J is .zip: it's possible to open/extract these files using programs supporting the zip format.



## <a name="config">Network Configurations</a>

This section lists the various configuration options that Deeplearning4j supports.

### <a name="config-afn">Activation Functions</a>

Activation functions can be defined in one of two ways:
(a) By passing an [Activation](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/Activation.java) enumeration value to the configuration - for example, ```.activation(Activation.TANH)```
(b) By passing an [IActivation](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/IActivation.java) instance - for example, ```.activation(new ActivationSigmoid())```

Note that Deeplearning4j supports custom activation functions, which can be defined by extending [BaseActivationFunction](https://github.com/deeplearning4j/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl)

List of supported activation functions:
* **CUBE** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationCube.java)) - ```f(x) = x^3```
* **ELU** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationELU.java)) - Exponential linear unit ([Reference](https://arxiv.org/abs/1511.07289))
* **HARDSIGMOID** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardSigmoid.java)) - a piecewise linear version of the standard sigmoid activation function. ```f(x) = min(1, max(0, 0.2*x + 0.5))```
* **HARDTANH** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardTanH.java)) - a piecewise linear version of the standard tanh activation function.
* **IDENTITY** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationIdentity.java)) - a 'no op' activation function: ```f(x) = x```
* **LEAKYRELU** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationLReLU.java)) - leaky rectified linear unit. ```f(x) = max(0, x) + alpha * min(0, x)``` with ```alpha=0.01``` by default.
* **RATIONALTANH** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRationalTanh.java)) - ```tanh(y) ~ sgn(y) * { 1 - 1/(1+|y|+y^2+1.41645*y^4)}``` which approximates ```f(x) = 1.7159 * tanh(2x/3)```, but should be faster to execute. ([Reference](https://arxiv.org/abs/1508.01292))
* **RELU** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationReLU.java)) - standard rectified linear unit: ```f(x) = x``` if ```x>0``` or ```f(x) = 0``` otherwise
* **RRELU** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRReLU.java)) - randomized rectified linear unit. Deterministic during test time. ([Reference](http://arxiv.org/abs/1505.00853))
* **SIGMOID** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSigmoid.java)) - standard sigmoid activation function, ```f(x) = 1 / (1 + exp(-x))```
* **SOFTMAX** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftmax.java)) - standard softmax activation function
* **SOFTPLUS** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftPlus.java)) - ```f(x) = log(1+e^x)``` - shape is similar to a smooth version of the RELU activation function
* **SOFTSIGN** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftSign.java)) - ```f(x) = x / (1+|x|)``` - somewhat similar in shape to the standard tanh activation function (faster to calculate).
* **TANH** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationTanH.java)) - standard tanh (hyperbolic tangent) activation function
* **RECTIFIEDTANH** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRectifiedTanh.java)) - ```f(x) = max(0, tanh(x))```
* **SELU** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSELU.java)) - scaled exponential linear unit - used with [self normalizing neural networks](https://arxiv.org/abs/1706.02515)
* **SWISH** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSwish.java)) - Swish activation function, ```f(x) = x * sigmoid(x)``` ([Reference](https://arxiv.org/abs/1710.05941))

### <a name="config-init">Weight Initialization</a>

Weight initialization refers to the method by which the initial parameters for a new network should be set.

Weight initialization are usually defined using the [WeightInit](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInit.java) enumeration.

Custom weight initializations can be specified using ```.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))``` for example. As for master (but not 0.9.1 release) ```.weightInit(new NormalDistribution(0, 1))``` is also possible, which is equivalent to the previous approach.

Available weight initializations. Not again that not all are available in the 0.9.1 release:

* **DISTRIBUTION**: Sample weights from a provided distribution (specified via ```dist``` configuration method
* **ZERO**: Generate weights as zeros
* **ONES**: All weights are set to 1
* **SIGMOID_UNIFORM**: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4*sqrt(6/(fanIn + fanOut))
* **NORMAL**: Normal/Gaussian distribution, with mean 0 and standard deviation 1/sqrt(fanIn). This is the initialization recommented in [Klambauer et al. 2017, "Self-Normalizing Neural Network"](https://arxiv.org/abs/1706.02515) paper. Equivalent to DL4J's XAVIER_FAN_IN and LECUN_NORMAL (i.e. Keras' "lecun_normal")
* **LECUN_UNIFORM**: Uniform U[-a,a] with a=3/sqrt(fanIn).
* **UNIFORM**: Uniform U[-a,a] with a=1/sqrt(fanIn). "Commonly used heuristic" as per Glorot and Bengio 2010
* **XAVIER**: As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
* **XAVIER_UNIFORM**: As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
* **XAVIER_FAN_IN**: Similar to Xavier, but 1/fanIn -> Caffe originally used this.
* **RELU**: [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Normal distribution with variance 2.0/nIn
* **RELU_UNIFORM**: [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Uniform distribution U(-s,s) with s = sqrt(6/fanIn)
* **IDENTITY**: Weights are set to an identity matrix. Note: can only be used with square weight matrices
* **VAR_SCALING_NORMAL_FAN_IN**: Gaussian distribution with mean 0, variance 1.0/(fanIn)
* **VAR_SCALING_NORMAL_FAN_OUT**: Gaussian distribution with mean 0, variance 1.0/(fanOut)
* **VAR_SCALING_NORMAL_FAN_AVG**: Gaussian distribution with mean 0, variance 1.0/((fanIn + fanOut)/2)
* **VAR_SCALING_UNIFORM_FAN_IN**: Uniform U[-a,a] with a=3.0/(fanIn)
* **VAR_SCALING_UNIFORM_FAN_OUT**: Uniform U[-a,a] with a=3.0/(fanOut)
* **VAR_SCALING_UNIFORM_FAN_AVG**: Uniform U[-a,a] with a=3.0/((fanIn + fanOut)/2)


### <a name="config-updaters">Updaters (Optimizers)</a>

An 'updater' in DL4J is a class that takes raw gradients and modifies them to become updates. These updates will then be applied to the network parameters.
The [CS231n course notes](http://cs231n.github.io/neural-networks-3/#ada) have a good explanation of some of these updaters.

Supported updaters in Deeplearning4j:
* **AdaDelta** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaDelta.java)) - [Reference](https://arxiv.org/abs/1212.5701)
* **AdaGrad** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaGrad.java)) - [Reference](http://jmlr.org/papers/v12/duchi11a.html)
* **AdaMax** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaMax.java)) - A variant of the Adam updater - [Reference](http://arxiv.org/abs/1412.6980)
* **Adam** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Adam.java))
* **Nadam** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nadam.java)) - A variant of the Adam updater, using the Nesterov mementum update rule - [Reference](https://arxiv.org/abs/1609.04747)
* **Nesterovs** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nesterovs.java)) - Nesterov momentum updater
* **NoOp** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/NoOp.java)) - A 'no operation' updater. That is, gradients are not modified at all by this updater. Mathematically equivalent to the SGD updater with a learning rate of 1.0
* **RmsProp** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/RmsProp.java)) - [Reference - slide 29](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
* **Sgd** - ([Source](https://github.com/deeplearning4j/nd4j/blob/e92cae4626e2c9cf27d416136f1895b747a32cee/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Sgd.java)) - Standard stochastic gradient descent updater. This updater applies a learning rate only.


### <a name="config-schedules">Learning Rate Schedules</a>

All updaters that support a learning rate also support learning rate schedules (the Nesterov momentum updater also supports a momentum schedule). Learning rate schedules can be specified either based on the number of iterations, or the number of epochs that have elapsed. Dropout (see below) can also make use of the schedules listed here.

Configure using, for example: ```.updater(new Adam(new ExponentialSchedule(ScheduleType.ITERATION, 0.1, 0.99 )))```
You can plot/inspect the learning rate that will be used at any point by calling ```ISchedule.valueAt(int iteration, int epoch)``` on the schedule object you have created.

Available schedules:

* **ExponentialSchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/ExponentialSchedule.java)) - Implements ```value(i) = initialValue * gamma^i```
* **InverseSchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/InverseSchedule.java)) - Implements ```value(i) = initialValue * (1 + gamma * i)^(-power)```
* **MapSchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/MapSchedule.java)) - Learning rate schedule based on a user-provided map. Note that the provided map must have a value for iteration/epoch 0. Has a builder class to conveniently define a schedule.
* **PolySchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/PolySchedule.java)) - Implements ```value(i) = initialValue * (1 + i/maxIter)^(-power)```
* **SigmoidSchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/SigmoidSchedule.java)) - Implements ```value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize)))```
* **StepSchedule** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/StepSchedule.java)) - Implements ```value(i) = initialValue * gamma^( floor(iter/step) )```


Note that custom schedules can be created by implementing the ISchedule interface.


### <a name="config-regularization">Regularization</a>

#### <a name="config-l1l2">L1/L2 Regularization</a>

L1 and L2 regularization can easily be added to a network via the configuration: ```.l1(0.1).l2(0.2)```.
Note that ```.regularization(true)``` must be enabled on 0.9.1 also (this option has been removed after 0.9.1 was released).

L1 and L2 regularization is applied by default on the weight parameters only. That is, .l1 and .l2 will not impact bias parameters - these can be regularized using ```.l1Bias(0.1).l2Bias(0.2)```.


#### <a name="config-dropout">Dropout</a>

All dropout types are applied at training time only. They are not applied at test time.

* **Dropout** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/Dropout.java)) - Each input activation x is independently set to (0, with probability 1-p) or (x/p with probability p)<br>
* **GaussianDropout** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/GaussianDropout.java)) - This is a multiplicative Gaussian noise (mean 1) on the input activations. Each input activation x is independently set to: ```x * y```, where ```y ~ N(1, stdev = sqrt((1-rate)/rate))```
* **GaussianNoise** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/GaussianNoise.java)) - Applies additive, mean-zero Gaussian noise to the input - i.e., ```x = x + N(0,stddev)```
* **AlphaDropout** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/AlphaDropout.java)) - AlphaDropout is a dropout technique proposed by [Klaumbauer et al. 2017 - Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515). Designed for self-normalizing neural networks (SELU activation, NORMAL weight init). Attempts to keep both the mean and variance of the post-dropout activations to the the same (in expectation) as before alpha dropout was applied

Note that (as of current master - but not 0.9.1) the dropout parameters can also be specified according to any of the schedule classes mentioned in the Learning Rate Schedules section.

### <a name="config-weightnoise">Weight Noise</a>

As per dropout, dropconnect / weight noise is applied only at training time

* **DropConnect** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/weightnoise/DropConnect.java)) - DropConnect is similar to dropout, but applied to the parameters of a network (instead of the input activations). [Reference](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)
* **WeightNoise** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/weightnoise/WeightNoise.java)) - Apply noise of the specified distribution to the weights at training time. Both additive and multiplicative modes are supported - when additive, noise should be mean 0, when multiplicative, noise should be mean 1

### <a name="config-constraints">Constraints</a>

Constraints are deterministic limitations that are placed on a model's parameters at the end of each iteration (after the parameter update has occurred). They can be thought of as a type of regularization.

* **MaxNormConstraint** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MaxNormConstraint.java)) - Constrain the maximum L2 norm of the incoming weights for each unit to be less than or equal to the specified value. If the L2 norm exceeds the specified value, the weights will be scaled down to satisfy the constraint.
* **MinMaxNormConstraint** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MinMaxNormConstraint.java)) - Constrain the minimum AND maximum L2 norm of the incoming weights for each unit to be between the specified values. Weights will be scaled up/down if required.
* **NonNegativeConstraint** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/NonNegativeConstraint.java)) - Constrain all parameters to be non-negative. Negative parameters will be replaced with 0.
* **UnitNormConstraint** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/b841c0f549194dbdf88b42836df662d9b8ea8c6d/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/UnitNormConstraint.java)) - Constrain the L2 norm of the incoming weights for each unit to be 1.0.


## <a name="data">Data Classes</a>

### <a name="data-iter">Iterators</a>

DataSetIterator is an abstraction that DL4J uses to iterate over minibatches of data, used for training. DataSetIterator returns DataSet objects, which are minibatches, and support a maximum of 1 input and 1 output array (INDArray).

MultiDataSetIterator is similar to DataSetIterator, but returns MultiDataSet objects, which can have as many input and output arrays arrays as required for the network.

#### <a name="data-iter-builtin">Iterators - Build-In (DL4J-Provided Data)</a>

These iterators download their data as required. The actual datasets they return are not customizable.

* **MnistDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.java)) - DataSetIterator for the well-known MNIST digits dataset. By default, returns a row vector (1x784), with values normalized to 0 to 1 range. Use ```.setInputType(InputType.convolutionalFlat())``` to use with CNNs.
* **EmnistDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/EmnistDataSetIterator.java)) - Similar to the MNIST digits dataset, but with more examples, and also letters. Includes multiple different splits (letters only, digits only, letters + digits, etc). Same 1x784 format as MNIST, hence (other than different number of labels for some splits) can be used as a drop-in replacement for MnistDataSetIterator. [Reference 1](https://www.nist.gov/itl/iad/image-group/emnist-dataset), [Reference 2](https://arxiv.org/abs/1702.05373)
* **IrisDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.java)) - An iterator for the well known Iris dataset. 4 features, 3 output classes.
* **CifarDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/CifarDataSetIterator.java)) - An iterator for the CIFAR images dataset. 10 classes, 4d features/activations format for CNNs in DL4J: ```[minibatch,channels,height,width] = [minibatch,3,32,32]```. Features are *not* normalized - instead, are in the range 0 to 255.
* **LFWDataSetIterator** - ([Source]())
* **TinyImageNetDataSetIterator** ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/TinyImageNetDataSetIterator.java)) - A subset of the standard imagenet dataset; 200 classes, 500 images per class
* **UciSequenceDataSetIterator** ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/UciSequenceDataSetIterator.java)) - UCI synthetic control time series dataset

#### <a name="data-iter-user">Iterators - User Provided Data</a>

The iterators in this subsection are used with user-provided data.

* **RecordReaderDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.java)) - an iterator that takes a DataVec record reader (such as CsvRecordReader or ImageRecordReader) and handles conversion to DataSets, batching, masking, etc. One of the most commonly used iterators in DL4J. Handles non-sequence data only, as input (i.e., RecordReader, no SequenceeRecordReader).
* **RecordReaderMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.java)) - the MultiDataSet version of RecordReaderDataSetIterator, that supports multiple readers. Has a builder pattern for creating more complex data pipelines (such as different subsets of a reader's output to different input/output arrays, conversion to one-hot, etc). Handles both sequence and non-sequence data as input.
* **SequenceRecordReaderDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.java)) - The sequence (SequenceRecordReader) version of RecordReaderDataSetIterator. Users may be better off using RecordReaderMultiDataSetIterator, in conjunction with
* **DoublesDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/DoublesDataSetIterator.java))
* **FloatsDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/FloatsDataSetIterator.java))
* **INDArrayDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/INDArrayDataSetIterator.java))


#### <a name="data-iter-util">Iterators - Adapter and Utility Iterators</a>

* **MultiDataSetIteratorAdapter** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/impl/MultiDataSetIteratorAdapter.java)) - Wrap a DataSetIterator to convert it to a MultiDataSetIterator
* **SingletonMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.java)) - Wrap a MultiDataSet into a MultiDataSetIterator that returns one MultiDataSet (i.e., the wrapped MultiDataSet is *not* split up)
* **AsyncDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.java)) - Used automatically by MultiLayerNetwork and ComputationGraph where appropriate. Implements asynchronous prefetching of datasets to improve performance.
* **AsyncMultiDataSetIterator**  - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.java)) - Used automatically by ComputationGraph where appropriate. Implements asynchronous prefetching of MultiDataSets to improve performance.
* **AsyncShieldDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/AsyncShieldDataSetIterator.java)) - Generally used only for debugging. Stops MultiLayerNetwork and ComputationGraph from using an AsyncDataSetIterator.
* **AsyncShieldMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/AsyncShieldMultiDataSetIterator.java)) - The MultiDataSetIterator version of AsyncShieldDataSetIterator
* **EarlyTerminationDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationDataSetIterator.java)) - Wraps another DataSetIterator, ensuring that only a specified (maximum) number of minibatches (DataSet) objects are returned between resets. Can be used to 'cut short' an iterator, returning only the first N DataSets.
* **EarlyTerminationMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationMultiDataSetIterator.java)) - The MultiDataSetIterator version of EarlyTerminationDataSetIterator
* **ExistingDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ExistingDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` or ```Iterable<DataSet>``` to a DataSetIterator. Does not split the underlying DataSet objects
* **FileDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/file/FileDataSetIterator.java)) - An iterator that iterates over DataSet files that have been previously saved with ```DataSet.save(File)```. Supports randomization, filtering, different output batch size vs. saved DataSet batch size, etc.
* **FileMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/file/FileMultiDataSetIterator.java)) - A MultiDataSet version of FileDataSetIterator
* **IteratorDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/IteratorDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` to a DataSetIterator. Unlike ExistingDataSetIterator, the underlying DataSet objects may be split/combined - i.e., the minibatch size may differ for the output, vs. the input iterator.
* **IteratorMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/IteratorMultiDataSetIterator.java)) - The ```Iterator<MultiDataSet>``` version of IteratorDataSetIterator
* **MultiDataSetWrapperIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultiDataSetWrapperIterator.java)) - Convert a MultiDataSetIterator to a DataSetIterator. Note that this is only possible if the number of features and labels arrays is equal to 1.
* **MultipleEpochsIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.java)) - Treat multiple passes (epochs) of the underlying iterator as a single epoch, when training.
* **WorkspaceShieldDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/WorkspacesShieldDataSetIterator.java)) - Generally used only for debugging, and not usually by users. Detaches/migrates DataSets coming out of the underlying DataSetIterator.


### <a name="data-norm">Data Normalization</a>

ND4J provides a number of classes for performing data normalization. These are implemented as DataSetPreProcessors.
The basic pattern for normalization:

1. Create your (unnormalized) DataSetIterator or MultiDataSetIterator: ```DataSetIterator myTrainData = ...```
2. Create the normalizer you want to use: ```NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();```
3. Fit the normalizer: ```normalizer.fit(myTrainData)```
4. Set the normalizer/preprocessor on the iterator: ```myTrainData.setPreProcessor(normalizer);```
End result: the data that comes from your DataSetIterator will now be normalized.

In general, you should fit *only* on the training data, and do ```trainData.setPreProcessor(normalizer)``` and ```testData.setPreProcessor(normalizer)``` with the same/single normalizer that has been fit on the training data only.

Note that where appropriate (NormalizerStandardize, NormalizerMinMaxScaler) statistics such as mean/standard-deviation/min/max are shared across time (for time series) and across image x/y locations (but not depth/channels - for image data).

Data normalization example: [link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/PreprocessNormalizerExample.java)

**Available normalizers: DataSet / DataSetIterator**

* **ImagePreProcessingScaler** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.java)) - Applies min-max scaling to image activations. Default settings do 0-255 input to 0-1 output (but is configurable). Note that unlike the other normalizers here, this one does not rely on statistics (mean/min/max etc) collected from the data, hence the ```normalizer.fit(trainData)``` step is unnecessary (is a no-op).
* **NormalizerStandardize** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/NormalizerStandardize.java)) - normalizes each feature value independently (and optionally label values) to have 0 mean and a standard deviation of 1
* **NormalizerMinMaxScaler** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/NormalizerMinMaxScaler.java)) - normalizes each feature value independently (and optionally label values) to lie between a minimum and maximum value (by default between 0 and 1)
* **VGG16ImagePreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/VGG16ImagePreProcessor.java)) - This is a preprocessor specifically for VGG16. It subtracts the mean RGB value, computed on the training set, from each pixel as reported in [Link](https://arxiv.org/pdf/1409.1556.pdf)


**Available normalizers: MultiDataSet / MultiDataSetIterator**

* **ImageMultiPreProcessingScaler** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/ImageMultiPreProcessingScaler.java)) - A MultiDataSet/MultiDataSetIterator version of ImagePreProcessingScaler
* **MultiNormalizerStandardize** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerStandardize.java)) - MultiDataSet/MultiDataSetIterator version of NormalizerStandardize
* **MultiNormalizerMinMaxScaler** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerMinMaxScaler.java)) - MultiDataSet/MultiDataSetIterator version of NormalizerMinMaxScaler
* **MultiNormalizerHybrid** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerHybrid.java)) - A MultiDataSet normalizer that can combine different normalization types (standardize, min/max etc) for different input/feature and output/label arrays.


### <a name="transfer">Transfer Learning</a>

Deeplearning4j has classes/utilities for performing transfer learning - i.e., taking an existing network, and modifying some of the layers (optionally freezing others so their parameters don't change). For example, an image classifier could be trained on ImageNet, then applied to a new/different dataset. Both MultiLayerNetwork and ComputationGraph can be used with transfer learning - frequently starting from a pre-trained model from the model zoo (see next section), though any MultiLayerNetwork/ComputationGraph can be used.

Link: [Transfer learning examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16)

The main class for transfer learning is [TransferLearning](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/transferlearning/TransferLearning.java). This class has a builder pattern that can be used to add/remove layers, freeze layers, etc.
[FineTuneConfiguration](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.java) can be used here to specify the learning rate and other settings for the non-frozen layers.


### <a name="zoo">Trained Model Library - Model Zoo</a>

Deeplearning4j provides a 'model zoo' - a set of pretrained models that can be downloaded and used either as-is (for image classification, for example) or often for transfer learning.

Link: [Deeplearning4j Model Zoo](https://deeplearning4j.org/model-zoo)

Models available in DL4J's model zoo:

* **AlexNet** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/AlexNet.java))
* **Darknet19** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/Darknet19.java))
* **FaceNetNN4Small2** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/FaceNetNN4Small2.java))
* **InceptionResNetV1** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/InceptionResNetV1.java))
* **LeNet** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/LeNet.java))
* **ResNet50** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/ResNet50.java))
* **SimpleCNN** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/SimpleCNN.java))
* **TextGenerationLSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TextGenerationLSTM.java))
* **TinyYOLO** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TinyYOLO.java))
* **VGG16** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG16.java))
* **VGG19** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG19.java))


**Note*: Trained Keras models (not provided by DL4J) may also be imported, using Deeplearning4j's Keras model import functionality.

## Cheat sheet code snippets

The Eclipse Deeplearning4j libraries come with a lot of functionality, and we've put together this cheat sheet to help users assemble neural networks and use tensors faster.

### Neural networks

Code for configuring common parameters and layers for both `MultiLayerNetwork` and `ComputationGraph`. See [MultiLayerNetwork](/api/{{page.version}}/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html) and [ComputationGraph](/api/{{page.version}}/org/deeplearning4j/nn/graph/ComputationGraph.html) for full API.

**Sequential networks**

Most network configurations can use `MultiLayerNetwork` class if they are sequential and simple.

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(1234)
    // parameters below are copied to every layer in the network
    // for inputs like dropOut() or activation() you should do this per layer
    // only specify the parameters you need
    .updater(new AdaGrad())
    .activation(Activation.RELU)
    .dropOut(0.8)
    .l1(0.001)
    .l2(1e-4)
    .weightInit(WeightInit.XAVIER)
    .weightInit(Distribution.TruncatedNormalDistribution)
    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
    .gradientNormalizationThreshold(1e-3)
    .list()
    // layers in the network, added sequentially
    // parameters set per-layer override the parameters above
    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .build())
    .layer(new ActivationLayer(Activation.RELU))
    .layer(new ConvolutionLayer.Builder(1,1)
            .nIn(1024)
            .nOut(2048)
            .stride(1,1)
            .convolutionMode(ConvolutionMode.Same)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.IDENTITY)
            .build())
    .layer(new GravesLSTM.Builder()
            .activation(Activation.TANH)
            .nIn(inputNum)
            .nOut(100)
            .build())
    .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
            .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true)
    .build();

MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(conf);
```

**Complex networks**

Networks that have complex graphs and "branching" such as *Inception* need to use `ComputationGraph`.

```java
ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
	.seed(seed)
    // parameters below are copied to every layer in the network
    // for inputs like dropOut() or activation() you should do this per layer
    // only specify the parameters you need
    .activation(Activation.IDENTITY)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(updater)
    .weightInit(WeightInit.RELU)
    .l2(5e-5)
    .miniBatch(true)
    .cacheMode(cacheMode)
    .trainingWorkspaceMode(workspaceMode)
    .inferenceWorkspaceMode(workspaceMode)
    .cudnnAlgoMode(cudnnAlgoMode)
    .convolutionMode(ConvolutionMode.Same)
    .graphBuilder()
    // layers in the network, added sequentially
    // parameters set per-layer override the parameters above
    // note that you must name each layer and manually specify its input
    .addInputs("input1")
    .addLayer("stem-cnn1", new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}, new int[] {3, 3})
    	.nIn(inputShape[0])
    	.nOut(64)
	    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
	    .build(),"input1")
    .addLayer("stem-batch1", new BatchNormalization.Builder(false)
    	.nIn(64)
    	.nOut(64)
    	.build(), "stem-cnn1")
    .addLayer("stem-activation1", new ActivationLayer.Builder()
    	.activation(Activation.RELU)
    	.build(), "stem-batch1")
    .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
        .activation(Activation.SOFTMAX).nOut(numClasses).lambda(1e-4).alpha(0.9)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
        "stem-activation1")
    .setOutputs("lossLayer")
    .setInputTypes(InputType.convolutional(224, 224, 3))
    .backprop(true).pretrain(false).build();

ComputationGraph neuralNetwork = new ComputationGraph(graph);
```


### Training

The code snippet below creates a basic pipeline that loads images from disk, applies random transformations, and fits them to a neural network. It also sets up a UI instance so you can visualize progress, and uses early stopping to terminate training early. You can adapt this pipeline for many different use cases.

```java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
int numExamples = Math.toIntExact(fileSplit.length());
int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
InputSplit trainData = inputSplit[0];
InputSplit testData = inputSplit[1];

boolean shuffle = false;
ImageTransform flipTransform1 = new FlipImageTransform(rng);
ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
ImageTransform warpTransform = new WarpImageTransform(rng, 42);
List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
	new Pair<>(flipTransform1,0.9),
    new Pair<>(flipTransform2,0.8),
    new Pair<>(warpTransform,0.5));

ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

// training dataset
ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(trainData, null);
DataSetIterator trainingIterator = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);

// testing dataset
ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(testData, null);
DataSetIterator testingIterator = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numLabels);

// early stopping configuration, model saver, and trainer
EarlyStoppingModelSaver saver = new LocalFileModelSaver(System.getProperty("user.dir"));
EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
    .evaluateEveryNEpochs(1)
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
    .scoreCalculator(new DataSetLossCalculator(testingIterator, true))     //Calculate test set score
    .modelSaver(saver)
    .build();

EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, neuralNetwork, trainingIterator);

// begin training
trainer.fit();
```

### Complex Transformation

DataVec comes with a portable `TransformProcess` class that allows for more complex data wrangling and data conversion. It works well with both 2D and sequence datasets.

```java
Schema schema = new Schema.Builder()
    .addColumnsDouble("Sepal length", "Sepal width", "Petal length", "Petal width")
    .addColumnCategorical("Species", "Iris-setosa", "Iris-versicolor", "Iris-virginica")
    .build();

TransformProcess tp = new TransformProcess.Builder(schema)
    .categoricalToInteger("Species")
    .build();

// do the transformation on spark
JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);
```

We recommend having a look at the [DataVec examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/datavec-examples/src/main/java/org/datavec/transform) before creating more complex transformations.


### Evaluation

Both `MultiLayerNetwork` and `ComputationGraph` come with built-in `.eval()` methods that allow you to pass a dataset iterator and return evaluation results.

```java
// returns evaluation class with accuracy, precision, recall, and other class statistics
Evaluation eval = neuralNetwork.eval(testIterator);
System.out.println(eval.accuracy());
System.out.println(eval.precision());
System.out.println(eval.recall());

// ROC for Area Under Curve on multi-class datasets (not binary classes)
ROCMultiClass roc = neuralNetwork.doEvaluation(testIterator, new ROCMultiClass());
System.out.println(roc.calculateAverageAuc());
System.out.println(roc.calculateAverageAucPR());
```

For advanced evaluation the code snippet below can be adapted into training pipelines. This is when the built-in `neuralNetwork.eval()` method outputs confusing results or if you need to examine raw data.

```java
//Evaluate the model on the test set
Evaluation eval = new Evaluation(numClasses);
INDArray output = neuralNetwork.output(testData.getFeatures());
eval.eval(testData.getLabels(), output, testMetaData); //Note we are passing in the test set metadata here

//Get a list of prediction errors, from the Evaluation object
//Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
List<Prediction> predictionErrors = eval.getPredictionErrors();
System.out.println("\n\n+++++ Prediction Errors +++++");
for(Prediction p : predictionErrors){
    System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
        + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
}

//We can also load the raw data:
List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);
for(int i=0; i<predictionErrors.size(); i++ ){
    Prediction p = predictionErrors.get(i);
    RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
    INDArray features = predictionErrorExamples.getFeatures().getRow(i);
    INDArray labels = predictionErrorExamples.getLabels().getRow(i);
    List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

    INDArray networkPrediction = model.output(features);

    System.out.println(meta.getLocation() + ": "
        + "\tRaw Data: " + rawData
        + "\tNormalized: " + features
        + "\tLabels: " + labels
        + "\tPredictions: " + networkPrediction);
}

//Some other useful evaluation methods:
List<Prediction> list1 = eval.getPredictions(1,2);                  //Predictions: actual class 1, predicted class 2
List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //All predictions for actual class 2
```
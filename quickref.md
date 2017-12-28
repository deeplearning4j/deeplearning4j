# Deeplearning4j Quick Reference: Layers, Functionality and Classes

Deeplearning4j (and related projects) have a lot of functionality. The goal of this page is to summarize this functionality so users know what exists, and where to find more information.

***IMPORTANT: This page contains content for both the last release (version 0.9.1) and the current master.***

* Available in 0.9.1: shown in **bold**
* Available only on master (build from source) / snapshots / after next release: show in *italics* 

# Layers

## Feed-Forward Layers

* **DenseLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/dense/DenseLayer.java)) - A simple/standard fully-connected layer 
* **EmbeddingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/embedding/EmbeddingLayer.java)) - Takes positive integer indexes as input, outputs vectors. Only usable as first layer in a model. Mathematically equivalent (when bias is enabled) to DenseLayer with one-hot input, but more efficient.

## Output Layers

Output layers: usable only as the last layer in a network. Loss functions are set here.

* **OutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/OutputLayer.java)) - Output layer for standard classification/regression in MLPs/CNNs. Has a fully connected DenseLayer built in. 2d input/output (i.e., row vector per example).
* **LossLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LossLayer.java)) - Output layer without parameters - only loss function and activation function. 2d input/output (i.e., row vector per example). Unlike Outputlayer, restricted to nIn = nOut.
* **RnnOutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnOutputLayer.java)) - Output layer for recurrent neural networks. 3d (time series) input and output. Has time distributed fully connected layer built in.
* *RnnLossLayer* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnLossLayer.java)) - The 'no parameter' version of RnnOutputLayer. 3d (time series) input and output.
* *CnnLossLayer* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CnnLossLayer.java)) - Used with CNNs, where a prediction must be made at each spatial location of the output (for example: segmentation or denoising). No parameters, 4d input/output with shape [minibatch, depth, height, width]. When using softmax, this is applied depthwise at each spatial location.
* *Yolo2OutputLayer* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/objdetect/Yolo2OutputLayer.java)) - Implentation of the YOLO 2 model for object detection in images
* **CenterLossOutputLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CenterLossOutputLayer.java)) - A version of OutputLayer that also attempts to minimize the intra-class distance of examples' activations - i.e., "If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y"


## Convolutional Layers

* **ConvolutionLayer** / Convolution2D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.java)) - Standard 2d convolutional neural network layer. Inputs and outputs have 4 dimensions with shape [minibatch,depthIn,heightIn,widthIn] and [minibatch,depthOut,heightOut,widthOut] respectively.
* **Convolution1DLayer** / Convolution1D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution1DLayer.java)) - Standard 1d convolution layer
* *Deconvolution2DLayer* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/Deconvolution2DLayer.java)) - also known as transpose or fractionally strided convolutions. Can be considered a "reversed" ConvolutionLayer; output size is generally larger than the input, whilst maintaining the spatial connection structure.
* *SeparableConvolution2DLayer* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/SeparableConvolution2DLayer.java)) - depthwise separable convolution layer
* **SubsamplingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/subsampling/SubsamplingLayer.java)) - Implements standard 2d spatial pooling for CNNs - with max, average and p-norm pooling available.
* **Subsampling1DLayer** - ([Source]())
* *Upsampling2D* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling2D.java)) - Upscale CNN activations by repeating the row/column values
* *Upsampling1D* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling1D.java)) - 1D version of the upsampling layer
* **ZeroPaddingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.java)) - Very simple layer that adds the specified amount of zero padding to edges of the 4d input activations.
* **ZeroPadding1DLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPadding1DLayer.java)) - 1D version of ZeroPaddingLayer


## Recurrent Layers

* **LSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java)) - LSTM RNN without peephole connections. Supports CuDNN.
* **GravesLSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesLSTM.java)) - LSTM RNN with peephole connections. Does *not* support CuDNN (thus for GPUs, LSTM should be used in preference).
* **GravesBidirectionalLSTM** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesBidirectionalLSTM.java)) - A bidirectional LSTM implementation with peephole connections. Equivalent to Bidirectional(ADD, GravesLSTM). Due to addition of Bidirecitonal wrapper (below), has been deprecated on master.
* *Bidirectional* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/Bidirectional.java)) - A 'wrapper' layer - converts any standard uni-directional RNN into a bidirectional RNN (doubles number of params - forward/backward nets have independent parameters). Activations from forward/backward nets may be either added, multiplied, averaged or concatenated.
* *SimpleRnn* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/SimpleRnn.java)) - A standard/'vanilla' RNN layer. Usually not effective in practice with long time series dependencies - LSTM is generally preferred.
* *LastTimeStep* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/LastTimeStep.java)) - A 'wrapper' layer - extracts out the last time step of the (non-bidirectional) RNN layer it wraps. 3d input with shape [minibatch, size, timeSeriesLength], 2d output with shape [minibatch, size].


## Unsupervised Layers

* **VariationalAutoencoder** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational/VariationalAutoencoder.java)) - A variational autoencoder implementation with MLP/dense layers for the encoder and decoder. Supports multiple different types of [reconstruction distributions](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational)
* **AutoEncoder** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/AutoEncoder.java)) - Standard denoising autoencoder layer

## Other Layers

* **GlobalPoolingLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GlobalPoolingLayer.java)) - Implements both pooling over time (for RNNs/time series - input size [minibatch, size, timeSeriesLength], out [minibatch, size]) and global spatial pooling (for CNNs - input size [minibatch, depth, h, w], out [minibatch, depth]). Available pooling modes: sum, average, max and p-norm.
* **ActivationLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ActivationLayer.java)) - Applies an activation function (only) to the input activations. Note that most DL4J layers have activation functions built in as a config option.
* **DropoutLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DropoutLayer.java)) - Implements dropout as a separate/single layer. Note that most DL4J layers have a "built-in" dropout configuration option.
* **BatchNormalization** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/BatchNormalization.java)) - Batch normalization for 2d (feedforward), 3d (time series) or 4d (CNN) activations. For time series, parameter sharing across time; for CNNs, parameter sharing across spatial locations (but not depth).
* **LocalResponseNormalization** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocalResponseNormalization.java)) - Local response normalization layer for CNNs. Not frequently used in modern CNN architectures.
* **FrozenLayer** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayer.java)) - Usually not used directly by users - added as part of transfer learning, to freeze a layer's parameters such that they don't change during further training.


# Graph Vertices

Graph vertex: use with ComputationGraph. Similar to layers, vertices usually don't have any parameters, and may support multiple inputs.

* **ElementWiseVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ElementWiseVertex.java)) - Performs an element-wise operation on the inputs - add, subtract, product, average, max
* **L2NormalizeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2NormalizeVertex.java)) - normalizes the input activations by dividing by the L2 norm for each example. i.e., out <- out / l2Norm(out)
* **L2Vertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - calculates the L2 distance between the two input arrays, for each example separately. Output is a single value, for each input value.
* **MergeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - merge the input activations along dimension 1, to make a larger output array. For CNNs, this implements merging along the depth/channels dimension
* **PreprocessorVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/PreprocessorVertex.java)) - a simple GraphVertex that contains an InputPreProcessor only
*  **ReshapeVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ReshapeVertex.java)) - Performs arbitrary activation array reshaping. The preprocessors in the next section should usually be preferred.
* **ScaleVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ScaleVertex.java)) - implements simple multiplicative scaling of the inputs - i.e., out = scalar * input
* **ShiftVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ShiftVertex.java)) - implements simple scalar element-wise addition on the inputs - i.e., out = input + scalar
* **StackVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/StackVertex.java)) - used to stack all inputs along the minibatch dimension. Analogous to MergeVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels)
* **SubsetVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/SubsetVertex.java)) - used to get a contiguous subset of the input activations along dimension 1. For example, two SubsetVertex instances could be used to split the activations from an input array into two separate activations. Essentially the opposite of MergeVertex.
*  **UnstackVertex** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/UnstackVertex.java)) - similar to SubsetVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels). Opposite of StackVertex



# Input Pre Processors

An InputPreProcessor is a simple class/interface that operates on the input to a layer. That is, a preprocessor is attached to a layer, and performs some operation on the input, before passing the layer to the output. Preprocessors also handle backpropagation - i.e., the preprocessing operations are generally differentiable.

Note that in many cases (such as the XtoYPreProcessor classes), users won't need to (and shouldn't) add these manually, and can instead just use ```.setInputType(InputType.feedForward(10))``` or similar, which whill infer and add the preprocessors as required.

* **CnnToFeedForwardPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToFeedForwardPreProcessor.java)) - handles the activation reshaping necessary to transition from a CNN layer (ConvolutionLayer, SubsamplingLayer, etc)
* **CnnToRnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.java)) - handles reshaping necessary to transition from a (effectively, time distributed) CNN layer to a RNN layer.
* **ComposableInputPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/ComposableInputPreProcessor.java)) - simple class that allows multiple preprocessors to be chained + used on a single layer
* **FeedForwardToCnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnnPreProcessor.java)) - handles activation reshaping to transition from a row vector (per example) to a CNN layer. Note that this transition/preprocessor only makes sense if the activations are actually CNN activations, but have been 'flattened' to a row vector.
* **FeedForwardToRnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToRnnPreProcessor.java)) - handles transition from a (time distributed) feed-forward layer to a RNN layer
* **RnnToCnnPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToCnnPreProcessor.java)) - handles transition from a sequence of CNN activations with shape ```[minibatch, depth*height*width, timeSeriesLength]``` to time-distributed ```[numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight]``` format
* **RnnToFeedForwardPreProcessor** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToFeedForwardPreProcessor.java)) - handles transition from time series activations (shape ```[minibatch,size,timeSeriesLength]```) to time-distributed feed-forward (shape ```[minibatch*tsLength,size]```) activations.  


# Iteration/Training Listeners

IterationListener: can be attached to a model, and are called during training, once after every iteration (i.e., after each parameter update).
TrainingListener: extends IterationListener. Has a number of additional methods are called at different stages of training - i.e., after forward pass, after gradient calculation, at the start/end of each epoch, etc.

Neither type (iteration/training) are called outside of training (i.e., during output or feed-forward methods)


* **ScoreIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ScoreIterationListener.java), Javadoc) - Logs the loss function score every N training iterations
* **PerformanceListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/PerformanceListener.java), Javadoc) - Logs performance (examples per sec, minibatches per sec, ETL time), and optionally score, every N training iterations.
* **EvaluativeListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/EvaluativeListener.java), Javadoc) - Evaluates network performance on a test set every N iterations or epochs. Also has a system for callbacks, to (for example) save the evaluation results.
* *CheckpointListener* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/checkpoint/CheckpointListener.java), Javadoc) - Save network checkpoints periodically - based on epochs, iterations or time (or some combination of all three).
* **StatsListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-ui-parent/deeplearning4j-ui-model/src/main/java/org/deeplearning4j/ui/stats/StatsListener.java)) - Main listener for DL4J's web-based network training user interface. See [visualization page](https://deeplearning4j.org/visualization) for more details.
* **CollectScoresIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.java), Javadoc) - Similar to ScoreIterationListener, but stores scores internally in a list (for later retrieval) instead of logging scores
* **TimeIterationListener** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/TimeIterationListener.java), Javadoc) - Attempts to estimate time until training completion, based on current speed

# Evaluation

Link: [Main evaluation page](https://deeplearning4j.org/evaluation)

DL4J has a number of classes for evaluating the performance of a network, against a test set. Different evaluation classes are suitable for different types of networks.

* **Evaluation** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/Evaluation.java)) - Used for the evaluation of multi-class classifiers (assumes standard one-hot labels, and softmax probability distribution over N classes for predictions). Calculates a number of metrics - accuracy, precision, recall, F1, F-beta, Matthews correlation coefficient, confusion matrix. Optionally calculates top N accuracy, custom binary decision thresholds, and cost arrays (for non-binary case). Typically used for softmax + mcxent/negative-log-likelihood networks.
* **EvaluationBinary** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationBinary.java)) - A multi-label binary version of the Evaluation class. Each network output is assumed to be a separate/independent binary class, with probability 0 to 1 independent of all other outputs. Typically used for sigmoid + binary cross entropy networks.
* **EvaluationCalibration** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationCalibration.java)) - Used to evaluation the calibration of a binary or multi-class classifier. Produces reliability diagrams, residual plots, and histograms of probabilities. Export plots to HTML using [EvaluationTools](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java).exportevaluationCalibrationToHtmlFile method
* **ROC** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROC.java)) - Used for single output binary classifiers only - i.e., networks with nOut(1) + sigmoid, or nOut(2) + softmax. Supports 2 modes: thresholded (approximate) or exact (the default). Calculates area under ROC curve, area under precision-recall curve. Plot ROC and P-R curves to HTML using [EvaluationTools](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java)
* **ROCBinary** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROCBinary.java)) - a version of ROC that is used for multi-label binary networks (i.e., sigmoid + binary cross entropy), where each network output is assumed to be an independent binary variable.  
* **ROCMultiClass** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROCMultiClass.java)) - a version of ROC that is used for multi-class (non-binary) networks (i.e., softmax + mcxent/negative-log-likelihood networks). As ROC metrics are only defined for binary classification, this treats the multi-class output as a set of 'one-vs-all' binary classification problems.
* **RegressionEvaluation** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/RegressionEvaluation.java)) - An evaluation class used for regression models (including multi-output regression models). Reports metrics such as mean-squared error (MSE), mean-absolute error, etc for each output.


# Network Configurations

This section lists the various configuration options that Deeplearning4j supports.

## Activation Functions

Activation functions can be defined in one of two ways:
(a) By passing an [Activation](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/Activation.java) enumeration value to the configuration - for example, ```.activation(Activation.TANH)```
(b) By passing an [IActivation](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/IActivation.java) instance - for example, ```.activation(new ActivationSigmoid())```

Note that Deeplearning4j supports custom activation functions, which can be defined by extending [BaseActivationFunction](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl)

List of supported activation functions:
* **CUBE** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationCube.java)) - ```f(x) = x^3```
* **ELU** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationELU.java)) - Exponential linear unit ([Reference](https://arxiv.org/abs/1511.07289))
* **HARDSIGMOID** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardSigmoid.java)) - a piecewise linear version of the standard sigmoid activation function. ```f(x) = min(1, max(0, 0.2*x + 0.5))```
* **HARDTANH** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardTanH.java)) - a piecewise linear version of the standard tanh activation function.
* **IDENTITY** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationIdentity.java)) - a 'no op' activation function: ```f(x) = x```
* **LEAKYRELU** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationLReLU.java)) - leaky rectified linear unit. ```f(x) = max(0, x) + alpha * min(0, x)``` with ```alpha=0.01``` by default.
* **RATIONALTANH** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRationalTanh.java)) - ```tanh(y) ~ sgn(y) * { 1 - 1/(1+|y|+y^2+1.41645*y^4)}``` which approximates ```f(x) = 1.7159 * tanh(2x/3)```, but should be faster to execute. ([Reference](https://arxiv.org/abs/1508.01292))
* **RELU** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationReLU.java)) - standard rectified linear unit: ```f(x) = x``` if ```x>0``` or ```f(x) = 0``` otherwise
* **RRELU** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRReLU.java)) - randomized rectified linear unit. Deterministic during test time. ([Reference](http://arxiv.org/abs/1505.00853))
* **SIGMOID** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSigmoid.java)) - standard sigmoid activation function, ```f(x) = 1 / (1 + exp(-x))```
* **SOFTMAX** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftmax.java)) - standard softmax activation function
* **SOFTPLUS** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftPlus.java)) - ```f(x) = log(1+e^x)``` - shape is similar to a smooth version of the RELU activation function
* **SOFTSIGN** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftSign.java)) - ```f(x) = x / (1+|x|)``` - somewhat similar in shape to the standard tanh activation function (faster to calculate).
* **TANH** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationTanH.java)) - standard tanh (hyperbolic tangent) activation function
* RECTIFIEDTANH - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRectifiedTanh.java)) - ```f(x) = max(0, tanh(x))```
* **SELU** - ([Source](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSELU.java)) - scaled exponential linear unit - used with [self normalizing neural networks](https://arxiv.org/abs/1706.02515)
* *SWISH* - ([Source]()) - Swish activation function, ```f(x) = x * sigmoid(x)``` ([Reference](https://arxiv.org/abs/1710.05941))

## Weight Initialization

Weight initialization refers to the method by which the initial parameters for a new network should be set.

[This section: remains to be written]


## Updaters (Optimizers)

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


### Learning Rate Schedules

**NOTE**: The information in this learning rate schedules section pertains to the schedules as they are *after* the 0.9.1 release (snapshots + current master, as of December 2017). As of 0.9.1 learning rate schedules were configured via a number of methods on NeuralNetConfiguration builder class. 

All updaters that support a learning rate also support learning rate schedules (the Nesterov momentum updater also supports a momentum schedule). Learning rate schedules can be specified either based on the number of iterations, or the number of epochs that have elapsed. Dropout (see below) can also make use of the schedules listed here.


[TODO]

## Regularization

### Dropout

* **Dropout**
* *GaussianDropout*
* *GaussianNoise*
* *AlphaDropout*

Note that (as of current master - but not 0.9.1) the dropout parameter can also be specified according to any of the schedule classes mentioned in the Learning Rate Schedules section.

### Weight Noise

* **DropConnect**
* *WeightNoise*

## Constraints

**Note**: Constraints were added after version 0.9.1 was released - and consequently are only available by building from source (or using snapshots).

Constraints are limitations that are placed on a model's parameters at the end of each iteration (after the parameter update has occurred). They can be thought of as a type of regularization.

* *MaxNormConstraint* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MaxNormConstraint.java)) - Constrain the maximum L2 norm of the incoming weights for each unit to be less than or equal to the specified value. If the L2 norm exceeds the specified value, the weights will be scaled down to satisfy the constraint.
* *MinMaxNormConstraint* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MinMaxNormConstraint.java)) - Constrain the minimum AND maximum L2 norm of the incoming weights for each unit to be between the specified values. Weights will be scaled up/down if required.
* *NonNegativeConstraint* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/NonNegativeConstraint.java)) - Constrain all parameters to be non-negative. Negative parameters will be replaced with 0.
* *UnitNormConstraint* - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/b841c0f549194dbdf88b42836df662d9b8ea8c6d/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/UnitNormConstraint.java)) - Constrain the L2 norm of the incoming weights for each unit to be 1.0. 


# Data Classes

## Iterators

DataSetIterator is an abstraction that DL4J uses to iterate over minibatches of data, used for training. DataSetIterator returns DataSet objects, which are minibatches, and support a maximum of 1 input and 1 output array (INDArray).

MultiDataSetIterator is similar to DataSetIterator, but returns MultiDataSet objects, which can have as many input and output arrays arrays as required for the network.

### Iterators - DL4J-Provided Data

These iterators download their data as required. The actual datasets they return are not customizable.

* **MnistDataSetIterator**
* **EmnistDataSetIterator**
* **IrisDataSetIterator**
* **CifarDataSetIterator**
* **LFWDataSetIterator**
* *TinyImageNetDataSetIterator*
* *UciSequenceDataSetIterator*

### Iterators - User Provided Data

The iterators in this subsection are used with user-provided data.

* **RecordReaderDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.java)) - an iterator that takes a DataVec record reader (such as CsvRecordReader or ImageRecordReader) and handles conversion to DataSets, batching, masking, etc. One of the most commonly used iterators in DL4J. Handles non-sequence data only, as input (i.e., RecordReader, no SequenceeRecordReader).
* **RecordReaderMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.java)) - the MultiDataSet version of RecordReaderDataSetIterator, that supports multiple readers. Has a builder pattern for creating more complex data pipelines (such as different subsets of a reader's output to different input/output arrays, conversion to one-hot, etc). Handles both sequence and non-sequence data as input.
* **SequenceRecordReaderDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.java)) - The sequence (SequenceRecordReader) version of RecordReaderDataSetIterator. Users may be better off using RecordReaderMultiDataSetIterator, in conjunction with 
* **DoublesDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/DoublesDataSetIterator.java))
* **FloatsDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/FloatsDataSetIterator.java))
* **INDArrayDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/INDArrayDataSetIterator.java))


### Iterators - Adapter and Utility Classes

* **MultiDataSetIteratorAdapter** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/impl/MultiDataSetIteratorAdapter.java)) - Wrap a DataSetIterator to convert it to a MultiDataSetIterator
* **SingletonMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.java)) - Wrap a MultiDataSet into a MultiDataSetIterator that returns one MultiDataSet (i.e., the wrapped MultiDataSet is *not* split up)
* **AsyncDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.java)) - Used automatically by MultiLayerNetwork and ComputationGraph where appropriate. Implements asynchronous prefetching of datasets to improve performance.
* **AsyncMultiDataSetIterator**  - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.java)) - Used automatically by ComputationGraph where appropriate. Implements asynchronous prefetching of MultiDataSets to improve performance.
* **AsyncShieldDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/AsyncShieldDataSetIterator.java)) - Generally used only for debugging. Stops MultiLayerNetwork and ComputationGraph from using an AsyncDataSetIterator.
* **AsyncShieldMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/AsyncShieldMultiDataSetIterator.java)) - The MultiDataSetIterator version of AsyncShieldDataSetIterator
* **EarlyTerminationDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationDataSetIterator.java)) - Wraps another DataSetIterator, ensuring that only a specified (maximum) number of minibatches (DataSet) objects are returned between resets. Can be used to 'cut short' an iterator, returning only the first N DataSets.
* **EarlyTerminationMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationMultiDataSetIterator.java)) - The MultiDataSetIterator version of EarlyTerminationDataSetIterator
* **ExistingDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/ExistingDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` or ```Iterable<DataSet>``` to a DataSetIterator. Does not split the underlying DataSet objects
* **IteratorDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/IteratorDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` to a DataSetIterator. Unlike ExistingDataSetIterator, the underlying DataSet objects may be split/combined - i.e., the minibatch size may differ for the output, vs. the input iterator.
* **IteratorMultiDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/IteratorMultiDataSetIterator.java)) - The ```Iterator<MultiDataSet>``` version of IteratorDataSetIterator
* **MultiDataSetWrapperIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/MultiDataSetWrapperIterator.java)) - Convert a MultiDataSetIterator to a DataSetIterator. Note that this is only possible if the number of features and labels arrays is equal to 1.
* **MultipleEpochsIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.java)) - Treat multiple passes (epochs) of the underlying iterator as a single epoch, when training.
* **WorkspaceShieldDataSetIterator** - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/WorkspacesShieldDataSetIterator.java)) - Generally used only for debugging, and not usually by users. Detaches/migrates DataSets coming out of the underlying DataSetIterator. 

## Spark Data Classes

[TODO]

# Transfer Learning

[TODO]

# SKIL - Model Deployment

[TODO]

# Keras Import

[TODO]

# Distributed Training (Spark)

[TODO]

# Hyperparameter Optimization (Arbiter)

[TODO]

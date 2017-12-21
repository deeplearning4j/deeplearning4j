# Deeplearning4j Quick Reference: Layers, Functionality and Classes

Deeplearning4j (and related projects) have a lot of functionality. The goal of this page is to summarize this functionality so users know what exists, and where to find more information.

# Layers

## Feed-Forward Layers

* DenseLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/dense/DenseLayer.java)) - A simple/standard fully-connected layer 
* EmbeddingLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/embedding/EmbeddingLayer.java)) - Takes positive integer indexes as input, outputs vectors. Only usable as first layer in a model. Mathematically equivalent (when bias is enabled) to DenseLayer with one-hot input, but more efficient.

## Output Layers

Output layers: usable only as the last layer in a network. Loss functions are set here.

* OutputLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/OutputLayer.java)) - Output layer for standard classification/regression in MLPs/CNNs. Has a fully connected DenseLayer built in. 2d input/output (i.e., row vector per example).
* LossLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LossLayer.java)) - Output layer without parameters - only loss function and activation function. 2d input/output (i.e., row vector per example). Unlike Outputlayer, restricted to nIn = nOut.
* RnnOutputLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnOutputLayer.java)) - Output layer for recurrent neural networks. 3d (time series) input and output. Has time distributed fully connected layer built in.
* RnnLossLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnLossLayer.java)) - The 'no parameter' version of RnnOutputLayer. 3d (time series) input and output.
* CnnLossLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CnnLossLayer.java)) - Used with CNNs, where a prediction must be made at each spatial location of the output (for example: segmentation or denoising). No parameters, 4d input/output with shape [minibatch, depth, height, width]. When using softmax, this is applied depthwise at each spatial location.
* Yolo2OutputLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/objdetect/Yolo2OutputLayer.java)) - Implentation of the YOLO 2 model for object detection in images
* CenterLossOutputLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CenterLossOutputLayer.java)) - A version of OutputLayer that also attempts to minimize the intra-class distance of examples' activations - i.e., "If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y"


## Convolutional Layers

* ConvolutionLayer / Convolution2D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.java)) - Standard 2d convolutional neural network layer. Inputs and outputs have 4 dimensions with shape [minibatch,depthIn,heightIn,widthIn] and [minibatch,depthOut,heightOut,widthOut] respectively.
* Convolution1DLayer / Convolution1D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution1DLayer.java)) - Standard 1d convolution layer
* Deconvolution2DLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/Deconvolution2DLayer.java)) - also known as transpose or fractionally strided convolutions. Can be considered a "reversed" ConvolutionLayer; output size is generally larger than the input, whilst maintaining the spatial connection structure.
* SeparableConvolution2DLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/SeparableConvolution2DLayer.java)) - depthwise separable convolution layer
* SubsamplingLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/subsampling/SubsamplingLayer.java)) - Implements standard 2d spatial pooling for CNNs - with max, average and p-norm pooling available.
* Subsampling1DLayer - ([Source]())
* Upsampling2D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling2D.java)) - Upscale CNN activations by repeating the row/column values
* Upsampling1D - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling1D.java)) - 1D version of the upsampling layer
* ZeroPaddingLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.java)) - Very simple layer that adds the specified amount of zero padding to edges of the 4d input activations.
* ZeroPadding1DLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPadding1DLayer.java)) - 1D version of ZeroPaddingLayer


## Recurrent Layers

* LSTM - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java)) - LSTM RNN without peephole connections. Supports CuDNN.
* GravesLSTM - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesLSTM.java)) - LSTM RNN with peephole connections. Does *not* support CuDNN (thus for GPUs, LSTM should be used in preference).
* Bidirectional - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/Bidirectional.java)) - A 'wrapper' layer - converts any standard uni-directional RNN into a bidirectional RNN (doubles number of params - forward/backward nets have independent parameters). Activations from forward/backward nets may be either added, multiplied, averaged or concatenated.
* SimpleRnn - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/SimpleRnn.java)) - A standard/'vanilla' RNN layer. Usually not effective in practice with long time series dependencies - LSTM is generally preferred.
* LastTimeStep - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/LastTimeStep.java)) - A 'wrapper' layer - extracts out the last time step of the (non-bidirectional) RNN layer it wraps. 3d input with shape [minibatch, size, timeSeriesLength], 2d output with shape [minibatch, size].


## Unsupervised Layers

* VariationalAutoencoder - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational/VariationalAutoencoder.java)) - A variational autoencoder implementation with MLP/dense layers for the encoder and decoder. Supports multiple different types of [reconstruction distributions](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational)
* AutoEncoder - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/AutoEncoder.java)) - Standard denoising autoencoder layer

## Other Layers

* GlobalPoolingLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GlobalPoolingLayer.java)) - Implements both pooling over time (for RNNs/time series - input size [minibatch, size, timeSeriesLength], out [minibatch, size]) and global spatial pooling (for CNNs - input size [minibatch, depth, h, w], out [minibatch, depth]). Available pooling modes: sum, average, max and p-norm.
* ActivationLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ActivationLayer.java)) - Applies an activation function (only) to the input activations. Note that most DL4J layers have activation functions built in as a config option.
* DropoutLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DropoutLayer.java)) - Implements dropout as a separate/single layer. Note that most DL4J layers have a "built-in" dropout configuration option.
* BatchNormalization - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/BatchNormalization.java)) - Batch normalization for 2d (feedforward), 3d (time series) or 4d (CNN) activations. For time series, parameter sharing across time; for CNNs, parameter sharing across spatial locations (but not depth).
* LocalResponseNormalization - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocalResponseNormalization.java)) - Local response normalization layer for CNNs. Not frequently used in modern CNN architectures.
* FrozenLayer - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayer.java)) - Usually not used directly by users - added as part of transfer learning, to freeze a layer's parameters such that they don't change during further training.


# Graph Vertices

Graph vertex: use with ComputationGraph. Similar to layers, vertices usually don't have any parameters, and may support multiple inputs.

* ElementWiseVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ElementWiseVertex.java)) - Performs an element-wise operation on the inputs - add, subtract, product, average, max
* L2NormalizeVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2NormalizeVertex.java)) - normalizes the input activations by dividing by the L2 norm for each example. i.e., out <- out / l2Norm(out)
* L2Vertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - calculates the L2 distance between the two input arrays, for each example separately. Output is a single value, for each input value.
* MergeVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - merge the input activations along dimension 1, to make a larger output array. For CNNs, this implements merging along the depth/channels dimension
* PreprocessorVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/PreprocessorVertex.java)) - a simple GraphVertex that contains an InputPreProcessor only
*  ReshapeVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ReshapeVertex.java)) - Performs arbitrary activation array reshaping. The preprocessors in the next section should usually be preferred.
* ScaleVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ScaleVertex.java)) - implements simple multiplicative scaling of the inputs - i.e., out = scalar * input
* ShiftVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ShiftVertex.java)) - implements simple scalar element-wise addition on the inputs - i.e., out = input + scalar
* StackVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/StackVertex.java)) - used to stack all inputs along the minibatch dimension. Analogous to MergeVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels)
* SubsetVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/SubsetVertex.java)) - used to get a contiguous subset of the input activations along dimension 1. For example, two SubsetVertex instances could be used to split the activations from an input array into two separate activations. Essentially the opposite of MergeVertex.
*  UnstackVertex - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/UnstackVertex.java)) - similar to SubsetVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels). Opposite of StackVertex



# Input Pre Processors

An InputPreProcessor is a simple class/interface that operates on the input of a layer. That is, a preprocessor is attached to a layer, and performs some operation on the input, before passing the layer to the output.

Note that in many cases (such as the XtoYPreProcessor classes), users won't need to (and shouldn't) add these manually, and can instead just use ```.setInputType(InputType.feedForward(10))``` or similar, which whill infer and add the preprocessors as required.

* CnnToFeedForwardPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToFeedForwardPreProcessor.java)) - handles the activation reshaping necessary to transition from a CNN layer (ConvolutionLayer, SubsamplingLayer, etc)
* CnnToRnnPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.java)) - handles reshaping necessary to transition from a (effectively, time distributed) CNN layer to a RNN layer.
* ComposableInputPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/ComposableInputPreProcessor.java)) - simple class that allows multiple preprocessors to be chained + used on a single layer
* FeedForwardToCnnPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnnPreProcessor.java)) - handles activation reshaping to transition from a row vector (per example) to a CNN layer. Note that this transition/preprocessor only makes sense if the activations are actually CNN activations, but have been 'flattened' to a row vector.
* FeedForwardToRnnPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToRnnPreProcessor.java)) - handles transition from a (time distributed) feed-forward layer to a RNN layer
* RnnToCnnPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToCnnPreProcessor.java)) - handles transition from a sequence of CNN activations with shape ```[minibatch, depth*height*width, timeSeriesLength]``` to time-distributed ```[numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight]``` format
* RnnToFeedForwardPreProcessor - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToFeedForwardPreProcessor.java)) - handles transition from time series activations (shape ```[minibatch,size,timeSeriesLength]```) to time-distributed feed-forward (shape ```[minibatch*tsLength,size]```) activations.  


# Iteration/Training Listeners

IterationListener: can be attached to a model. Called during training, once after every iteration (i.e., after each parameter update).
TrainingListener: extends IterationListener. Has a number of methods are called at different stages of training - i.e., after forward pass, after gradient calculation, at the start/end of each epoch.

Neither type (iteration/training) are called outside of training.


* ScoreIterationListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ScoreIterationListener.java), Javadoc) - Logs the loss function score every N training iterations
* PerformanceListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/PerformanceListener.java), Javadoc) - Logs performance (examples per sec, minibatches per sec, ETL time), and optionally score, every N training iterations.
* EvaluativeListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/EvaluativeListener.java), Javadoc) - Evaluates network performance on a test set every N iterations or epochs
* CheckpointListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/checkpoint/CheckpointListener.java), Javadoc) - Save network checkpoints periodically - based on epochs, iterations or time.
* StatsListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-ui-parent/deeplearning4j-ui-model/src/main/java/org/deeplearning4j/ui/stats/StatsListener.java)) - Main listener for DL4J's web-based network training user interface. See [visualization page](https://deeplearning4j.org/visualization) for more details.
* CollectScoresIterationListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.java), Javadoc) - Similar to ScoreIterationListener, but stores scores internally in a list (for later retrieval) instead of logging scores
* TimeIterationListener - ([Source](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/TimeIterationListener.java), Javadoc) - Attempts to estimate time until training completion, based on current speed

# Evaluation

Link: [Main evaluation page](https://deeplearning4j.org/evaluation)

* Evaluation
* EvaluationBinary
* EvaluationCalibration
* ROC
* ROCBinary
* ROCMultiClass
* RegressionEvaluation


# Network Configurations

## Activation Functions

## Weight Initialization

## Updaters (Optimizers)

## Regularization

### Dropout

* Dropout
* GaussianDropout
* GaussianNoise
* AlphaDropout


### Weight Noise

* DropConnect
* WeightNoise

# Data Classes

## Iterators

## Spark Data Classes


# Transfer Learning

# SKIL - Model Deployment

# Keras Import

# Distributed Training (Spark)

# Hyperparameter Optimization (Arbiter)

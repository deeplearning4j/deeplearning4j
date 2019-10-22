---
title: Deeplearning4j Cheat Sheet
short_title: Cheat Sheet
description: Snippets and links for common functionality in Eclipse Deeplearning4j.
category: Get Started
weight: 2
---

# Quick reference

Deeplearning4j (and related projects) have a lot of functionality. We've put together this cheat sheet to help you assemble neural networks and use tensors faster.

The goal of this page is to summarize Deeplearning4j's capabilities so that you know what exists, and where to find more information.

**Contents**

- [Layers](#layers)
  - [Feed-Forward Layers](#layers-ff)
    - [Output Layers](#layers-out)
    - [Convolutional Layers](#layers-conv)
    - [Recurrent Layers](#layers-rnn)
    - [Unsupervised Layers](#layers-unsupervised)
    - [Other Layers](#layers-other)
    - [Capsule Layers](#layers-capsule)
    - [Attention Layers](#layers-attention)
    - [Graph Vertices](#layers-vertices)
  - [Input Preprocessors](#layers-preproc)
- [Iteration/Training Listeners](#listeners)
- [Evaluation](#evaluation)
- [Network Saving and Loading](#saving)
- [Network Configurations](#config)
  - [Activation Functions](#config-afn)
  - [Weight Initialization](#config-init)
  - [Updaters (Optimizers)](#config-updaters)
  - [Learning Rate Schedules](#config-schedules)
  - [Regularization](#config-regularization)
    - [L1/L2 Regularization](#config-l1l2)
    - [Dropout](#config-dropout)
    - [Weight Noise](#config-weightnoise)
    - [Constraints](#config-constraints)
- [Data Classes](#data)
  - [Iterators](#data-iter)
    - [Iterators - Built-In (Deeplearning4j-Provided Data)](#data-iter-builtin)
    - [Iterators - User Provided Data](#data-iter-user)
    - [Iterators - Adapter and Utility Iterators](#data-iter-util)
  - [Reading Raw Data: DataVec RecordReaders](#data-datavec)
  - [Data Normalization](#data-norm)
  - [Spark Network Training Data Classes](#data-spark)
- [Transfer Learning](#transfer)
- [Trained Model Library - Model Zoo](#zoo)
- [Keras Import](#keras)
- [Distributed Training (Spark)](#spark)

## <a name="layers">Layers</a> 

### <a name="layers-ff">Feed-Forward Layers</a> 

`org.deeplearning4j.nn.conf.layers` 
`org.deeplearning4j.nn.conf.graph`

- **DenseLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DenseLayer.java)) - A simple/standard fully-connected layer.
- **EmbeddingLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/EmbeddingLayer.java)) - Takes positive integer indexes as input, outputs vectors. Only usable as first layer in a model. Mathematically equivalent (when bias is enabled) to **DenseLayer** with one-hot input, but more efficient. See also: **EmbeddingSequenceLayer**.
- **ElementWiseMultiplicationLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/ElementWiseMultiplicationLayer.java)) - Elementwise multiplication layer with weights: implements ```out = activationFn(input .* w + b)```. Note that the input and output sizes of the element-wise layer are the same for this layer.

#### <a name="layers-out">Output Layers</a>

Output layers are usable only as the last layer in a network. Loss functions are set here.

- **OutputLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/OutputLayer.java)) - Output layer for standard classification/regression in multilayer perceptrons (MLP)/convolutional neural networks (CNN). Has a fully connected **DenseLayer** built in. 2D input/output (i.e. row vector per example).
- **LossLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LossLayer.java)) - Output layer without parameters - only loss function and activation function. 2D input/output (i.e. row vector per example). Unlike **OutputLayer**, restricted to `nIn = nOut`.
- **RnnOutputLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnOutputLayer.java)) - Output layer for recurrent neural networks (RNN). 3D (time series) input and output. Has time distributed fully connected layer built in.
- **RnnLossLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnLossLayer.java)) - The 'no parameter' version of **RnnOutputLayer**. 3D (time series) input and output.
- **CnnLossLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CnnLossLayer.java)) - Used with CNNs, where a prediction must be made at each spatial location of the output (for example: segmentation or denoising). No parameters, 4D input/output with shape `[minibatch, depth, height, width]`. When using softmax, this is applied depthwise at each spatial location.
- **Cnn3DLossLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Cnn3DLossLayer.java)) - Used with 3D CNNs, where a preduction must be made at each spatial location (x/y/z) of the output. Layer has no parameters, 5D data in either "channels first" (NCDHW) or "channels last"(NDHWC) format (configurable). Supports masking. When using Softmax, this is applied along channels at each spatial location.
- **Yolo2OutputLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/objdetect/Yolo2OutputLayer.java)) - Implementation of the YOLOv2 model for object detection in images.
- **CenterLossOutputLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CenterLossOutputLayer.java)) - A version of **OutputLayer** that also attempts to minimize the intra-class distance of examples' activations, i.e. "If example x is in class Y, ensure that `embedding(x)` is close to `average(embedding(y))` for all examples y in Y".
- **OCNNOutputLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/ocnn/OCNNOutputLayer.java)) - An implementation of one class neural networks from [Anomaly Detection Using One-class Neural Networks](https://arxiv.org/pdf/1802.06360.pdf). 

#### <a name="layers-conv">Convolutional Layers</a>

🔗 [Main CNN page](/docs/{{page.version}}/deeplearning4j-nn-convolutional)

- **Convolution2D** / ConvolutionLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.java)) - Standard 2D convolutional layer. Inputs and outputs have 4 dimensions with shape `[minibatch, depthIn, heightIn, widthIn]` and `[minibatch, depthOut, heightOut, widthOut]` respectively.
- **Convolution1D** / Convolution1DLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution1DLayer.java)) - Standard 1D convolutional layer.
- **Convolution3D** / Convolution3DLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution3D.java)) - Standard 3D convolutional layer. Supports both "channels first" (NCDHW) or "channels last"(NDHWC) activations format.
- **Deconvolution2D** / Deconvolution2DLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Deconvolution2D.java)) - Also known as transpose or fractionally strided convolutions. Can be considered a "reversed" **ConvolutionLayer**; output size is generally larger than the input, whilst maintaining the spatial connection structure.
- **SeparableConvolution2D** / SeparableConvolution2DLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SeparableConvolution2D.java)) - Depthwise separable convolutional layer.
- **SubsamplingLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SubsamplingLayer.java)) - Implements standard 2D spatial pooling for CNNs—with max, average and p-norm pooling available.
- **Subsampling1DLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Subsampling1DLayer.java)) - 1D version of the subsampling layer.
- **Subsampling3DLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Subsampling3DLayer.java)) - 3D subsampling / pooling layer for convolutional neural networks.
- **Upsampling2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Upsampling2D.java)) - Upscale CNN activations by repeating the row/column values.
- **Upsampling1D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Upsampling1D.java)) - 1D version of the upsampling layer.
- **Upsampling3D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Upsampling3D.java)) - Upsampling 3D layer.
- **Cropping2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/convolutional/Cropping2D.java)) - Cropping layer for 2D CNNs.
- **DepthwiseConvolution2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DepthwiseConvolution2D.java))- 2D depthwise convolutional layer.
- **ZeroPaddingLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ZeroPaddingLayer.java)) - Very simple layer that adds the specified amount of zero padding to edges of the 4D input activations.
- **ZeroPadding1DLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ZeroPadding1DLayer.java)) - 1D version of **ZeroPaddingLayer**.
- **ZeroPadding3DLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ZeroPadding3DLayer.java)) - Zero padding 3D layer for CNNs.
- **SpaceToDepth** /  SpaceToDepthLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToDepthLayer.java)) - This operation takes 4D array as input, and moves data from spatial dimensions (HW) to channels (C) for given `blockSize`.
- **SpaceToBatch** / SpaceToBatchLayer ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToBatchLayer.java)) - Transforms data from a tensor from 2 spatial dimensions into batch dimension according to the "blocks" specified.
- **Pooling1D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Pooling1D.java)) - 1D Pooling (subsampling) layer. Equivalent to **Subsampling1DLayer**.
- **Pooling2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Pooling2D.java)) - 2D Pooling (subsampling) layer. Equivalent to **SubsamplingLayer**.

#### <a name="layers-rnn">Recurrent Layers</a>

🔗 [Main RNN page](/docs/{{page.version}}/deeplearning4j-nn-recurrent)

- **LSTM** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java)) - Long Short-term Memory (LSTM) RNN without peephole connections. Supports CuDNN.
- **Bidirectional** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/Bidirectional.java)) - A 'wrapper' layer—converts any standard uni-directional RNN into a bidirectional RNN (doubles number of params—forward/backward nets have independent parameters). Activations from forward/backward nets may be either added, multiplied, averaged or concatenated.
- **SimpleRnn** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/SimpleRnn.java)) - A standard/'vanilla' RNN layer. Usually not effective in practice with long time series dependencies - LSTM is generally preferred.
- **LastTimeStep** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/LastTimeStep.java)) - A 'wrapper' layer—extracts out the last time step of the (non-bidirectional) RNN layer it wraps. 3D input with shape `[minibatch, size, timeSeriesLength]`, 2D output with shape `[minibatch, size]`.
- **EmbeddingSequenceLayer**: ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/EmbeddingSequenceLayer.java)) - A version of EmbeddingLayer that expects fixed-length number (inputLength) of integers/indices per example as input, ranged from `0` to `numClasses - 1`. This input thus has shape `[numExamples, inputLength]` or shape `[numExamples, 1, inputLength`]. The output of this layer is 3D (sequence/time series), namely of shape `[numExamples, nOut, inputLength]`. Can only be used as the first layer for a network.
- **RecurrentAttentionLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RecurrentAttentionLayer.java)) - Implements Recurrent Dot Product Attention. Takes in RNN style input in the shape of `[batchSize, features, timesteps]` and applies dot product attention using the hidden state as the query and **all** time steps as keys/values. `a_i = σ(W * x_i + R * attention(a_i, x, x) + b)` The output will be in the shape of `[batchSize, nOut, timesteps]`. 

> 📝 **Note**: At the moment, this is limited to equal-length mini-batch input. Mixing mini-batches of differing timestep counts will not work.

#### <a name="layers-unsupervised">Unsupervised Layers</a>

- **VariationalAutoencoder** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational/VariationalAutoencoder.java)) - A variational autoencoder implementation with MLP/dense layers for the encoder and decoder. Supports multiple different types of [reconstruction distributions](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational).
- **AutoEncoder** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/AutoEncoder.java)) - Standard denoising autoencoder layer.

#### <a name="layers-capsule">Capsule Layers</a>

CapsNet is from [Dynamic Routing Between Capsules](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf).

- **PrimaryCapsules** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/PrimaryCapsules.java)) -  A reshaped 2D convolution, and the input should be 2D convolutional with shape `[mb, c, h, w]`.
- **CapsuleLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CapsuleLayer.java)) - An implementation of the DigiCaps layer. Input should come from a **PrimaryCapsules** layer and be of shape `[mb, inputCaps, inputCapDims]`.
- **CapsuleStrengthLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CapsuleStrengthLayer.java)) - A layer to get the "strength" of each capsule, that is, the probability of it being in the input. This is the vector length or L2 norm of each capsule's output. The lengths will not exceed one because of the squash function. Input should come from a Capsule Layer and be of shape `[mb, capsules, capsuleDims]`. 

#### <a name="layers-attention">Attention Layers</a>

Attention implemented as in [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf), pp. 4,5

- **LearnedSelfAttentionLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LearnedSelfAttentionLayer.java)) - Implements Dot Product Self Attention with learned queries. Takes in RNN style input in the shape of `[batchSize, features, timesteps]` and applies dot product attention using learned queries. The output will be in the shape of `[batchSize, nOut, nQueries]`. If input masks are used, they are applied to the input here and not propagated any further as now the time steps are given by the configured query count. While not an exact implementation of the paper, this is inspired by [A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf). 
- **SelfAttentionLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CapsuleStrengthLayer.java)) - Implements Dot Product Self Attention. Takes in RNN-style input in the shape of `[batchSize, features, timesteps]` and applies dot product attention using each timestep as the query. The output will be in the shape of `[batchSize, nOut, timesteps]`. 
- **AttentionVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/AttentionVertex.java)) - Implements Dot Product Attention using the given inputs. For Timestep-wise Self-Attention, use the same value for all three inputs.

#### <a name="layers-other">Other Layers</a>

- **GlobalPoolingLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GlobalPoolingLayer.java)) - Implements both pooling over time (for RNNs/time series: input size `[minibatch, size, timeSeriesLength]`, out `[minibatch, size]`) and global spatial pooling (for CNNs - input size `[minibatch, depth, h, w]`, out `[minibatch, depth]`). Available pooling modes: sum, average, max and p-norm.
- **ActivationLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ActivationLayer.java)) - Applies an activation function (only) to the input activations. Note that most Deeplearning4j layers have activation functions built in as a config option.
- **DropoutLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DropoutLayer.java)) - Implements dropout as a separate/single layer. Note that most Deeplearning4j layers have a "built-in" dropout configuration option.
- **BatchNormalization** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/BatchNormalization.java)) - Batch normalization for 2D (feedforward), 3D (time series) or 4D (CNN) activations. For time series, parameter sharing across time; for CNNs, parameter sharing across spatial locations (but not depth).
- **LocalResponseNormalization** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocalResponseNormalization.java)) - Local response normalization layer for CNNs. Not frequently used in modern CNN architectures.
- **FrozenLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayer.java)) - Usually not used directly by users. Added as part of transfer learning, to freeze a layer's parameters such that they don't change during further training.
- **FrozenLayerWithBackprop** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayerWithBackprop.java)) -  Frozen layer freezes parameters of the layer it wraps, but allows the backpropagation to continue.
- **LocallyConnected2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocallyConnected2D.java)) - A 2D locally connected layer, assumes input is 4D data in NCHW ("channels first") format.
- **LocallyConnected1D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocallyConnected1D.java)) - A 1D locally connected layer, assumes input is 3D data in NCW (`[minibatch, size, sequenceLength]`) format.
- **MaskLayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/cd710314dbd87bb6c769a4862433f4eaac31277c/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/util/MaskLayer.java)) - Applies the mask array to the forward pass activations, and backward pass gradients, passing through this layer. It can be used with 2D (feed-forward), 3D (time series) or 4D (CNN) activations.

#### <a name="layers-vertices">Graph Vertices</a>

🔗 [Main vertices page](/docs/{{page.version}}/deeplearning4j-nn-vertices)

Graph vertex: use with ComputationGraph. Similar to layers, vertices usually don't have any parameters, and may support multiple inputs.

- **ElementWiseVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ElementWiseVertex.java)) - Performs an element-wise operation on the inputs: add, subtract, product, average, max. 
- **FrozenVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/FrozenVertex.java)) - Used for the purposes of transfer learning. A **FrozenVertex** wraps another Deeplearning4j GraphVertex within it. During backpropagation, the **FrozenVertex** is skipped, and any parameters are not be updated. Usually users will not create **FrozenVertex** instances directly - they are usually used in the process of performing transfer learning.
- **L2NormalizeVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2NormalizeVertex.java)) - Normalizes the input activations by dividing by the L2 norm for each example, that is, `out <- out / l2Norm(out)`.
- **L2Vertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - Calculates the L2 distance between the two input arrays, for each example separately. Output is a single value, for each input value.
- **MergeVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - Merge the input activations along dimension 1, to make a larger output array. For CNNs, this implements merging along the depth/channels dimension.
- **PreprocessorVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/PreprocessorVertex.java)) - A simple GraphVertex that contains an InputPreProcessor only.
- **ReshapeVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ReshapeVertex.java)) - Performs arbitrary activation array reshaping. The preprocessors in the next section should usually be preferred.
- **ScaleVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ScaleVertex.java)) - Implements simple multiplicative scaling of the inputs, that is, `out = scalar * input`.
- **ShiftVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ShiftVertex.java)) - Implements simple scalar element-wise addition on the inputs, that is, `out = input + scalar`.
- **StackVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/StackVertex.java)) - Used to stack all inputs along the minibatch dimension. Analogous to **MergeVertex**, but along dimension 0 (minibatch) instead of dimension 1 (`nOut / channels`).
- **SubsetVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/SubsetVertex.java)) - Used to get a contiguous subset of the input activations along dimension 1. For example, two **SubsetVertex** instances could be used to split the activations from an input array into two separate activations. Essentially the opposite of **MergeVertex**.
- **UnstackVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/UnstackVertex.java)) - Similar to **SubsetVertex**, but along dimension 0 (minibatch) instead of dimension 1 (`nOut / channels`). Opposite of **StackVertex**.
- **ReverseTimeSeriesVertex** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/rnn/ReverseTimeSeriesVertex.java)) - Used in recurrent neural networks to revert the order of time series. As a result, the last time step is moved to the beginning of the time series and the first time step is moved to the end. This allows recurrent layers to backward process time series.



### <a name="layers-preproc">Input Preprocessors</a>

An InputPreProcessor is a simple class/interface that operates on the input to a layer. That is, a preprocessor is attached to a layer, and performs some operation on the input, before passing the layer to the output. Preprocessors also handle backpropagation, i.e. the preprocessing operations are generally differentiable.

Note that in many cases (such as the XtoYPreProcessor classes), users won't need to (and shouldn't) add these manually, and can instead just use ```.setInputType(InputType.feedForward(10))``` or similar, which will infer and add the preprocessors as required.

- **Cnn3DToFeedForwardPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/Cnn3DToFeedForwardPreProcessor.java)) - A preprocessor to allow CNN and standard feed-forward network layers to be used together, for example, **Convolution3DLayer** -> **DenseLayer**.
- **CnnToFeedForwardPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToFeedForwardPreProcessor.java)) - Handles the activation reshaping necessary to transition from a CNN layer (**ConvolutionLayer**, **SubsamplingLayer**, etc.) to **DenseLayer**/**OutputLayer** etc.
- **CnnToRnnPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.java)) - Handles reshaping necessary to transition from a (effectively, time distributed) CNN layer to a RNN layer.
- **ComposableInputPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/ComposableInputPreProcessor.java)) - Simple class that allows multiple preprocessors to be chained and used on a single layer.
- **FeedForwardToCnnPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnnPreProcessor.java)) - Handles activation reshaping to transition from a row vector (per example) to a CNN layer. Note that this transition/preprocessor only makes sense if the activations are actually CNN activations, but have been 'flattened' to a row vector.
- **FeedForwardToCnn3DPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnn3DPreProcessor.java)) - A preprocessor to allow 3D CNN and standard feed-forward network layers to be used together. For example, **DenseLayer** -> **Convolution3DLayer**.
- **FeedForwardToRnnPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToRnnPreProcessor.java)) - Handles transition from a (time distributed) feed-forward layer to a RNN layer.
- **RnnToCnnPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToCnnPreProcessor.java)) - Handles transition from a sequence of CNN activations with shape ```[minibatch, depth * height * width, timeSeriesLength]``` to time-distributed ```[numExamples * timeSeriesLength, numChannels, inputWidth, inputHeight]``` format.
- **RnnToFeedForwardPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToFeedForwardPreProcessor.java)) - Handles transition from time series activations (shape ```[minibatch, size, timeSeriesLength]```) to time-distributed feed-forward (shape ```[minibatch * tsLength, size]```) activations.
- **CombinedPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/CombinedPreProcessor.java)) - This is a special preProcessor that allows to combine multiple preprocessors, and apply them to data sequentially.
- **CombinedMultiDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/CombinedMultiDataSetPreProcessor.java)) - Combines various multidataset preprocessors. Applied in the order they are specified to in the builder. 

## <a name="listeners">Iteration/Training Listeners</a>

🔗 [Main listeners page](/docs/{{page.version}}/deeplearning4j-nn-listeners)

- IterationListener: can be attached to a model, and are called during training, once after every iteration (i.e. after each parameter update).
- TrainingListener: extends IterationListener. Has a number of additional methods which are called at different stages of training, such as after forward pass, after gradient calculation and at the start/end of each epoch.

Neither type (iteration/training) are called outside of training (i.e. during output or feed-forward methods)

- **CheckpointListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CheckpointListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/checkpoint/CheckpointListener.html)) - Save network checkpoints periodically based on epochs, iterations or time (or some combination of all three).
- **CollectScoresListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/CollectScoresListener.html)) - A simple listener that collects scores to a list every N iterations. Can also log the score, optionally.
- **CollectScoresIterationListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.html)) - Similar to **ScoreIterationListener**, but stores scores internally in a list (for later retrieval) instead of logging scores
- **EvaluativeListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/EvaluativeListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/EvaluativeListener.html)) - Evaluates network performance on a test set every N iterations or epochs. Also has a system for callbacks, to save the evaluation results, for example.
- **ParamAndGradientIterationListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ParamAndGradientIterationListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/ParamAndGradientIterationListener.html)) - An iteration listener that provides details on parameters and gradients at each iteration during traning. Attempts to provide much of the same information as the UI histogram iteration listener, but in a text-based format (for example, when learning on a system accessed via SSH etc). It is intended to aid network tuning and debugging. This iteration listener is set up to calculate mean, min, max, and mean absolute value of each type of parameter and gradient in the network at each iteration.
- **PerformanceListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/PerformanceListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/PerformanceListener.html)) - Logs performance (examples per sec, minibatches per sec, ETL time), and optionally score, every N training iterations.
- **ScoreIterationListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ScoreIterationListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/ScoreIterationListener.html)) - Logs the loss function score every N training iterations.
- **StatsListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui-model/src/main/java/org/deeplearning4j/ui/stats/StatsListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/ui/stats/StatsListener.html)) - Main listener for Deeplearning4j's web-based network training user interface. See [visualization page](https://deeplearning4j.org/visualization) for more details.
- **TimeIterationListener** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/TimeIterationListener.java), [Javadoc](/api/{{page.version}}/org/deeplearning4j/optimize/listeners/TimeIterationListener.html)) - Attempts to estimate time until training completion based on current speed and specified total number of iterations.

## <a name="evaluation">Evaluation</a>

🔗 [Main evaluation page](/docs/{{page.version}}/deeplearning4j-nn-evaluation)

ND4J has a number of classes for evaluating the performance of a network, against a test set. Deeplearning4j (and SameDiff) use these ND4J evaluation classes. Different evaluation classes are suitable for different types of networks.

> 📝 **Note**: in 1.0.0-beta3 (November 2018), all evaluation classes were moved from Deeplearning4j to ND4J; previously they were in Deeplearning4j.

- **Evaluation** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/Evaluation.java)) - Used for the evaluation of multi-class classifiers (assumes standard one-hot labels, and softmax probability distribution over N classes for predictions). Calculates a number of metrics: accuracy, precision, recall, F1, F-beta, Matthews correlation coefficient, confusion matrix. Optionally calculates top N accuracy, custom binary decision thresholds, and cost arrays (for non-binary case). Typically used for softmax + mcxent/negative-log-likelihood networks.
- **EvaluationBinary** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/EvaluationBinary.java)) - A multi-label binary version of the Evaluation class. Each network output is assumed to be a separate/independent binary class, with probability 0 to 1 independent of all other outputs. Typically used for sigmoid + binary cross entropy networks.
- **EvaluationCalibration** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/EvaluationCalibration.java)) - Used to evaluation the calibration of a binary or multi-class classifier. Produces reliability diagrams, residual plots, and histograms of probabilities. Export plots to HTML using [EvaluationTools](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java).exportevaluationCalibrationToHtmlFile method.
- **ROC** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROC.java)) - Used for single output binary classifiers only, i.e. networks with `nOut(1) + sigmoid` or `nOut(2) + softmax`. Supports 2 modes: thresholded (approximate) or exact (the default). Calculates area under receiver operating characteristic (ROC) curve, area under precision-recall curve. Plot ROC and P-R curves to HTML using [EvaluationTools](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java).
- **ROCBinary** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROCBinary.java)) - A version of ROC that is used for multi-label binary networks (i.e. sigmoid + binary cross entropy), where each network output is assumed to be an independent binary variable.  
- **ROCMultiClass** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROCMultiClass.java)) - A version of ROC that is used for multi-class (non-binary) networks (i.e. softmax + mcxent/negative-log-likelihood networks). As ROC metrics are only defined for binary classification, this treats the multi-class output as a set of 'one-vs-all' binary classification problems.
- **RegressionEvaluation** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/regression/RegressionEvaluation.java)) - An evaluation class used for regression models (including multi-output regression models). Reports metrics such as mean-squared error (MSE) and mean-absolute error (MAE) for each output/column.

## <a name="saving">Network Saving and Loading</a>

```MultiLayerNetwork.save(File)``` and ```MultiLayerNetwork.load(File)``` methods can be used to save and load models. These use ModelSerializer internally. Similar save/load methods are also available for ComputationGraph.

MultiLayerNetwork and ComputationGraph can be saved using the [ModelSerializer](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/util/ModelSerializer.java) class - and specifically the ```writeModel```, ```restoreMultiLayerNetwork``` and ```restoreComputationGraph``` methods.

🔗 [deeplearning4j-examples: Saving and loading network](https://github.com/eclipse/deeplearning4j-examples/tree/master/Deeplearning4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving)

Networks can be trained further after saving and loading: however, be sure to load the 'updater' (i.e. the historical state for updaters like momentum). If no futher training is required, the updater state can be omitted to save disk space and memory.

Most Normalizers (implementing the ND4J ```Normalizer``` interface) can also be added to a model using the ```addNormalizerToModel``` method.

> 📝 **Note**: The format used for models in Deeplearning4j is ZIP. It's possible to open/extract these files using programs supporting the ZIP format.



## <a name="config">Network Configurations</a>

This section lists the various configuration options that Deeplearning4j supports.

### <a name="config-afn">Activation Functions</a>

`org.nd4j.linalg.activations.impl`

🔗 [Main activations page](/docs/{{page.version}}/nd4j-nn-activations)

Activation functions can be defined in one of two ways:

- By passing an [Activation](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/Activation.java) enumeration value to the configuration, for example, ```.activation(Activation.TANH)```.
- By passing an [IActivation](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/IActivation.java) instance, for example, ```.activation(new ActivationSigmoid())```.

> 📝 **Note**: Deeplearning4j supports custom activation functions, which can be defined by extending [BaseActivationFunction](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl).

List of supported activation functions:

- CUBE / **ActivationCube** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationCube.java)) - ```f(x) = x^3```
- ELU / **ActivationELU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationELU.java)) - Exponential linear unit. ([Reference](https://arxiv.org/abs/1511.07289))
- HARDSIGMOID / **ActivationHardSigmoid** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardSigmoid.java)) - A piecewise linear version of the standard sigmoid activation function. ```f(x) = min(1, max(0, 0.2 * x + 0.5))```
- HARDTANH / **ActivationHardTanH** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardTanH.java)) - A piecewise linear version of the standard tanh activation function.
- IDENTITY / **ActivationIdentity** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationIdentity.java)) - A 'no op' activation function. ```f(x) = x```
- LEAKYRELU / **ActivationLReLU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationLReLU.java)) - Leaky rectified linear unit. ```f(x) = max(0, x) + alpha * min(0, x)``` with ```alpha = 0.01``` by default.
- RATIONALTANH / **ActivationRationalTanh** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRationalTanh.java)) - ```tanh(y) ~ sgn(y) * { 1 - 1 / (1 + |y| + y^2 +  1.41645 * y^4)}``` which approximates ```f(x) = 1.7159 * tanh(2x / 3)```, but should be faster to execute. ([Reference](https://arxiv.org/abs/1508.01292))
- RELU / **ActivationReLU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationReLU.java)) - Standard rectified linear unit. ```f(x) = x``` if ```x > 0``` or ```f(x) = 0``` otherwise
- RELU6 / **ActivationReLU6** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationReLU6.java)) - ```f(x) = min(max(input, cutoff), 6)```
- RRELU / **ActivationRReLU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRReLU.java)) - Randomized rectified linear unit. Deterministic during test time. ([Reference](http://arxiv.org/abs/1505.00853))
- SIGMOID / **ActivationSigmoid** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSigmoid.java)) - Standard sigmoid activation function. ```f(x) = 1 / (1 + exp(-x))```
- SOFTMAX / **ActivationSoftmax** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftmax.java)) - Standard softmax activation function.
- SOFTPLUS / **ActivationSoftPlus** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftPlus.java)) - ```f(x) = log(1 + e^x)``` - shape is similar to a smooth version of the RELU activation function.
- SOFTSIGN / **ActivationSoftSign** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftSign.java)) - ```f(x) = x / (1 + |x|)``` - somewhat similar in shape to **ActivationTanH** (faster to calculate).
- TANH / **ActivationTanH** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationTanH.java)) - Standard tanh (hyperbolic tangent) activation function.
- RECTIFIEDTANH / **ActivationRectifiedTanh** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRectifiedTanh.java)) - ```f(x) = max(0, tanh(x))```
- SELU / **ActivationSELU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSELU.java)) - Scaled exponential linear unit. Used with [self-normalizing neural networks](https://arxiv.org/abs/1706.02515).
- SWISH / **ActivationSwish** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSwish.java)) - Swish activation function. ```f(x) = x * sigmoid(x)``` ([Reference](https://arxiv.org/abs/1710.05941))
- THRESHOLDEDRELU / **ActivationThresholdedReLU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationThresholdedReLU.java)) - Thresholded RELU. ```f(x) = x``` for ```x > theta```, ```f(x) = 0``` otherwise. theta defaults to ```1.0```.
- GELU / **ActivationGELU** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationGELU.java)) - Gaussian Error Linear Units (GELU) activation function.

> 📝 **Note**: You should import the following activation function as a layer from `org.deeplearning4j.nn.conf.layers`.

- **PReLULayer** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/PReLULayer.java)) - Parametrized Rectified Linear Unit (PReLU).

### <a name="config-init">Weight Initialization</a>

`org.deeplearning4j.nn.weights`

Weight initialization refers to the method by which the initial parameters for a new network should be set.

> 📝 **Note**: Beginning from 1.0.0-beta4, weight initialization can be defined as classes using [IWeightInit](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/IWeightInit.java) instances, not just enumerations with [WeightInit](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInit.java).

Weight initialization can be defined in one of two ways:

- By passing a [WeightInit](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInit.java) enumeration value to the configuration, for example, ```.weightInit(WeightInit.UNIFORM)```.
- By passing an [IWeightInit](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/IWeightInit.java) instance, for example, ```.weightInit(new WeightInitUniform())```.

Custom weight initializations can be specified by extending the IWeightInit class.

Available weight initializations:

- DISTRIBUTION / **WeightInitDistribution** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitDistribution.java)) - Sample weights from a provided distribution. 
- **WeightInitConstant** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitConstant.java)) - Initialize to a constant value (default 0). Enumerators ZERO and ONES available for constant values 0 and 1 respectively. 
- IDENTITY / **WeightInitIdentity** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitIdentity.java)) - Weights are set to an identity matrix. `nIn==nOut`.
  - For Dense layers, this means square weight matrix.
  - For convolution layers, an additional constraint is that kernel size must be odd length in all dimensions.
- LECUN_UNIFORM / **WeightInitLecunUniform** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitLecunUniform.java)) - Uniform `U[-a, a]` with `a = 3 / sqrt(fanIn)`.
- SIGMOID_UNIFORM / **WeightInitSigmoidUniform** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitSigmoidUniform.java)) - A version of WeightInitXavierUniform for sigmoid activation functions. `U(-r, r)` with `r = 4 * sqrt(6 / (fanIn + fanOut))`.
- NORMAL / **WeightInitNormal** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitNormal.java)) - Normal/Gaussian distribution, with mean 0 and standard deviation `1 / sqrt(fanIn)`. This is the initialization recommented in [Klambauer et al. 2017, "Self-Normalizing Neural Network"](https://arxiv.org/abs/1706.02515) paper. Equivalent to Deeplearning4j's XAVIER_FAN_IN and LECUN_NORMAL (i.e. Keras' "lecun_normal")
- RELU / **WeightInitRelu** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitRelu.java)) - [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Normal distribution with variance `2.0 / nIn`.
- RELU_UNIFORM / **WeightInitReluUniform** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitReluUniform.java)) - [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Uniform distribution `U(-s, s)` with `s = sqrt(6 / fanIn)`.
- UNIFORM / **WeightInitUniform** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitUniform.java)) Uniform `U[-a, a]` with `a = 1 / sqrt(fanIn)`. "Commonly used heuristic" as per Glorot and Bengio 2010.
- VAR_SCALING_NORMAL_FAN_AVG / **WeightInitVarScalingNormalFanAvg** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingNormalFanAvg.java)) - Gaussian distribution with mean 0, variance `1.0 / ((fanIn + fanOut) / 2)`.
- VAR_SCALING_NORMAL_FAN_IN / **WeightInitVarScalingNormalFanIn** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingNormalFanIn.java)) - Gaussian distribution with mean 0, variance `1.0 / (fanIn)`.
- VAR_SCALING_NORMAL_FAN_OUT / **WeightInitVarScalingNormalFanOut** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingNormalFanOut.java)) Gaussian distribution with mean 0, variance `1.0 / (fanOut)`.
- VAR_SCALING_UNIFORM_FAN_AVG / **WeightInitVarScalingUniformFanAvg** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingUniformFanAvg.java)) - Uniform `U[-a, a]` with `a = 3.0 / ((fanIn + fanOut) / 2)`.
- VAR_SCALING_UNIFORM_FAN_IN / **WeightInitVarScalingUniformFanIn** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingUniformFanIn.java)) - Uniform `U[-a, a]` with `a = 3.0 / (fanIn)`.
- VAR_SCALING_UNIFORM_FAN_OUT / **WeightInitVarScalingUniformFanOut** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitVarScalingUniformFanOut.java)) - Uniform `U[-a, a]` with `a = 3.0 / (fanOut)`.
- XAVIER / **WeightInitXavier** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitXavier.java)) - As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Gaussian distribution with mean 0, variance `2.0 / (fanIn + fanOut)`.
- XAVIER_LEGACY / **WeightInitXavierLegacy** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitXavierLegacy.java)) - Xavier weight init in Deeplearning4j up to 0.6.0. XAVIER should be preferred.
- XAVIER_UNIFORM / **WeightInitXavierUniform** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitXavierUniform.java)) - As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Uniform distribution `U(-s, s)` with `s = sqrt(6 / (fanIn + fanOut))`.

### <a name="config-updaters">Updaters (Optimizers)</a>

`org.nd4j.linalg.learning.config` 

🔗 [Main updaters page](/docs/{{page.version}}/nd4j-nn-updaters)

An 'updater' in ND4J is a class that takes raw gradients and modifies them to become updates. These updates will then be applied to the network parameters.
The [CS231n course notes](http://cs231n.github.io/neural-networks-3/#ada) have a good explanation of some of these updaters.

Supported updaters in ND4J:

- **AdaDelta** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaDelta.java)) - [Reference](https://arxiv.org/abs/1212.5701)
- **AdaGrad** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaGrad.java)) - [Reference](http://jmlr.org/papers/v12/duchi11a.html)
- **AdaMax** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaMax.java)) - A variant of the Adam updater. [Reference](http://arxiv.org/abs/1412.6980)
- **Adam** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Adam.java))
- **AMSGrad** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AMSGrad.java)) - The AMSGrad updater. [Reference](https://openreview.net/forum?id=ryQu7f-RZ)
- **Nadam** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nadam.java)) - A variant of the Adam updater, using the Nesterov mementum update rule. [Reference](https://arxiv.org/abs/1609.04747)
- **Nesterovs** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nesterovs.java)) - Nesterov momentum updater.
- **NoOp** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/NoOp.java)) - A 'no operation' updater. That is, gradients are not modified at all by this updater. Mathematically equivalent to the **Sgd** updater with a learning rate of 1.0.
- **RmsProp** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/RmsProp.java)) - [Reference - slide 29](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- **Sgd** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Sgd.java)) - Standard stochastic gradient descent updater. This updater applies a learning rate only.

### <a name="config-schedules">Learning Rate Schedules</a>

`org.nd4j.linalg.schedule`

All updaters that support a learning rate also support learning rate schedules (the Nesterov momentum updater also supports a momentum schedule). Learning rate schedules can be specified either based on the number of iterations, or the number of epochs that have elapsed. Dropout (see below) can also make use of the schedules listed here.

Configure using, for example, ```.updater(new Adam(new ExponentialSchedule(ScheduleType.ITERATION, 0.1, 0.99)))```
You can plot/inspect the learning rate that will be used at any point by calling ```ISchedule.valueAt(int iteration, int epoch)``` on the schedule object you have created.

Available schedules:

- **CycleSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/CycleSchedule.java)) - Based on 1-cycle schedule as proposed in https://arxiv.org/abs/1803.09820. Starts at initial learning rate, then linearly increases learning rate until max learning rate is reached, at that point the learning rate is decreased back to initial learning rate.  When `cycleLength - annealingLength` is reached, the annealing period starts, and the learning rate starts decaying below the initial learning rate.
- **ExponentialSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/ExponentialSchedule.java)) - Implements ```value(i) = initialValue * gamma^i```
- **FixedSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/FixedSchedule.java))
- **InverseSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/InverseSchedule.java)) - Implements ```value(i) = initialValue * (1 + gamma * i)^(-power)```
- **MapSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/MapSchedule.java)) - Learning rate schedule based on a user-provided map. Note that the provided map must have a value for iteration/epoch 0. Has a builder class to conveniently define a schedule.
- **PolySchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/PolySchedule.java)) - Implements ```value(i) = initialValue * (1 + i / maxIter)^(-power)```
- **RampSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/RampSchedule.java)) - A "Wrapper" schedule that ramps up from `1 / numIter * baseLR` to `baseLR` over numIter iterations. The base learning rate is determined by the underlying ISchedule, as a function of time. This can be used to provide a slow start for use cases such as transfer learning.
- **SigmoidSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/SigmoidSchedule.java)) - Implements ```value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize)))```
- **StepSchedule** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/StepSchedule.java)) - Implements ```value(i) = initialValue * gamma^(floor(iter / step))```

Note that custom schedules can be created by implementing the ISchedule interface.

### <a name="config-regularization">Regularization</a>

🔗 [Main regularization page](/docs/{{page.version}}/deeplearning4j-troubleshooting-training#regularization)

#### <a name="config-l1l2">L1/L2 Regularization</a>

L1 and L2 regularization can easily be added to a network via the configuration: ```.l1(0.1).l2(0.2)```.

> 📝 **Note**: In v0.9.1, ```.regularization(true)``` must also be enabled. This is not required in 1.0.0-alpha onwards.

L1 and L2 regularization is applied by default on the weight parameters only. That is, `.l1` and `.l2` will not impact bias parameters - these can be regularized using ```.l1Bias(0.1).l2Bias(0.2)```.

#### <a name="config-dropout">Dropout</a>

`org.deeplearning4j.nn.conf.dropout`

All dropout types are applied at training time only. They are not applied at test time.

- **Dropout** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/Dropout.java)) - Each input activation `x` is independently set to (0, with probability `1 - p`) or (`x / p` with probability `p`).
- **GaussianDropout** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/GaussianDropout.java)) - This is a multiplicative Gaussian noise (mean 1) on the input activations. Each input activation `x` is independently set to: ```x * y```, where ```y ~ N(1, stdev = sqrt((1 - rate) / rate))```.
- **GaussianNoise** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/GaussianNoise.java)) - Applies additive, mean-zero Gaussian noise to the input - i.e. ```x = x + N(0, stddev)```.
- **AlphaDropout** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/AlphaDropout.java)) - A dropout technique proposed by [Klaumbauer et al. 2017 - Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515). Designed for self-normalizing neural networks (SELU activation, NORMAL weight init). Attempts to keep both the mean and variance of the post-dropout activations to the same (in expectation) as before alpha dropout was applied.
- **SpatialDropout** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/dropout/SpatialDropout.java)) - Can only be applied to 3D (time series), 4D (convolutional 2D) or 5D (convolutional 3D) activations. Dropout mask is generated along the depth dimension, and is applied to:
  - For 3D/time series/sequence input: each step in the sequence.
  - For 4D (CNN 2D) input: each `x`/`y` location in an image.
  - For 5D (CNN 3D) input: each `x`/`y`/`z` location in a volume.
    Note that the dropout mask is generated independently for each example: i.e. a dropout mask of shape `[minibatch, channels]` is generated and applied to activations of shape `[minibatch, channels, height, width]`. [Reference](https://arxiv.org/abs/1411.4280)

> 📝 **Note**: the dropout parameters can also be specified according to any of the schedule classes mentioned in the Learning Rate Schedules section.

#### <a name="config-weightnoise">Weight Noise</a>

`org.deeplearning4j.nn.conf.weightnoise`

As per dropout, **DropConnect** / **WeightNoise** is applied only at training time.

- **DropConnect** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/weightnoise/DropConnect.java)) - Similar to dropout, but applied to the parameters of a network (instead of the input activations). [Reference](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)
- **WeightNoise** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/weightnoise/WeightNoise.java)) - Apply noise of the specified distribution to the weights at training time. Both additive and multiplicative modes are supported - when additive, noise should be mean 0, when multiplicative, noise should be mean 1.

#### <a name="config-constraints">Constraints</a>

`org.deeplearning4j.nn.conf.constraint`

Constraints are deterministic limitations that are placed on a model's parameters at the end of each iteration (after the parameter update has occurred). They can be thought of as a type of regularization.

- **MaxNormConstraint** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MaxNormConstraint.java)) - Constrains the maximum L2 norm of the incoming weights for each unit to be less than or equal to the specified value. If the L2 norm exceeds the specified value, the weights will be scaled down to satisfy the constraint.
- **MinMaxNormConstraint** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/MinMaxNormConstraint.java)) - Constrains the minimum AND maximum L2 norm of the incoming weights for each unit to be between the specified values. Weights will be scaled up/down if required.
- **NonNegativeConstraint** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/NonNegativeConstraint.java)) - Constrains all parameters to be non-negative. Negative parameters will be replaced with 0.
- **UnitNormConstraint** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/constraint/UnitNormConstraint.java)) - Constrains the L2 norm of the incoming weights for each unit to be 1.0.

## <a name="data">Data Classes</a>

### <a name="data-iter">Iterators</a>

🔗 [Main iterators page](/docs/{{page.version}}/deeplearning4j-nn-iterators)

DataSetIterator is an abstraction that Deeplearning4j uses to iterate over minibatches of data, used for training. DataSetIterator returns DataSet objects, which are minibatches, and support a maximum of 1 input and 1 output array (INDArray).

MultiDataSetIterator is similar to DataSetIterator, but returns MultiDataSet objects, which can have as many input and output arrays as required for the network.

#### <a name="data-iter-builtin">Iterators - Built-In (Deeplearning4j-Provided Data)</a>

`org.deeplearning4j.datasets.iterator.impl`

These iterators download their data as required. The actual datasets they return are not customizable.

- **MnistDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.java)) - DataSetIterator for the well-known MNIST digits dataset. By default, returns a row vector (1x784), with values normalized to 0 to 1 range. Use ```.setInputType(InputType.convolutionalFlat())``` to use with CNNs.
- **EmnistDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/EmnistDataSetIterator.java)) - Similar to the MNIST digits dataset, but with more examples, and also letters. Includes multiple different splits (letters only, digits only, letters + digits, etc). Same 1x784 format as MNIST, hence (other than different number of labels for some splits) can be used as a drop-in replacement for MnistDataSetIterator. [Reference 1](https://www.nist.gov/itl/iad/image-group/emnist-dataset), [Reference 2](https://arxiv.org/abs/1702.05373)
- **IrisDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.java)) - An iterator for the well known Iris dataset. 4 features, 3 output classes.
- **Cifar10DataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/Cifar10DataSetIterator.java)) (**CifarDataSetIterator** before 1.0.0-beta4) - An iterator for the CIFAR images dataset. 10 classes, 4D features/activations format for CNNs in Deeplearning4j: ```[minibatch, channels, height, width] = [minibatch, 3, 32, 32]```. Features are *not* normalized, that is, they are in the range of 0 to 255.
- **LFWDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.java))
- **TinyImageNetDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/TinyImageNetDataSetIterator.java)) - A subset of the standard ImageNet dataset. 200 classes, 500 images per class.
- **UciSequenceDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/UciSequenceDataSetIterator.java)) - University of California, Irvine (UCI) synthetic control time series dataset.

#### <a name="data-iter-user">Iterators - User Provided Data</a>

`org.deeplearning4j.datasets.datavec`
`org.deeplearning4j.datasets.iterator`

The iterators in this subsection are used with user-provided data.

- **RecordReaderDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.java)) - An iterator that takes a DataVec record reader (such as CsvRecordReader or ImageRecordReader) and handles conversion to DataSets, batching, masking, etc. One of the most commonly used iterators in Deeplearning4j. Handles non-sequence data only, as input (i.e. RecordReader, no SequenceRecordReader).
- **RecordReaderMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.java)) - The MultiDataSet version of RecordReaderDataSetIterator, that supports multiple readers. Has a builder pattern for creating more complex data pipelines, such as different subsets of a reader's output to different input/output arrays and conversion to one-hot. Handles both sequence and non-sequence data as input.
- **SequenceRecordReaderDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-datavec-iterators/src/main/java/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.java)) - The sequence (SequenceRecordReader) version of **RecordReaderDataSetIterator**. Users may be better off using **RecordReaderMultiDataSetIterator**, in conjunction with
- **AbstractDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/AbstractDataSetIterator.java)) -  This is simple DataSetIterator implementation, that builds DataSetIterator out of INDArray/float[]/double[] pairs. Suitable for model feeding with externally originated data.
- **DoublesDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/DoublesDataSetIterator.java))
- **FloatsDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/FloatsDataSetIterator.java))
- **INDArrayDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/INDArrayDataSetIterator.java))

#### <a name="data-iter-util">Iterators - Adapter and Utility Iterators</a>

`org.deeplearning4j.datasets.iterator`

- **MultiDataSetIteratorAdapter** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/impl/MultiDataSetIteratorAdapter.java)) - Wrap a DataSetIterator to convert it to a MultiDataSetIterator.
- **SingletonMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.java)) - Wrap a MultiDataSet into a MultiDataSetIterator that returns one MultiDataSet (i.e. the wrapped MultiDataSet is *not* split up).
- **BertIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/iterator/BertIterator.java)) - MultiDataSetIterator for training BERT (Transformer) models in the following ways:
  - Unsupervised - Masked language model task (no sentence matching task is implemented thus far)
  - Supervised - For sequence classification (i.e., 1 label per sequence, typically used for fine tuning)
- **DataSetIteratorSplitter** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/DataSetIteratorSplitter.java)) -  This iterator virtually splits given MultiDataSetIterator into Train and Test parts, for example, you have 100000 examples. Your batch size is 32. That means you have 3125 total batches. With split ratio of 0.7 that will give you 2187 training batches, and 938 test batches.
- **EarlyTerminationDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationDataSetIterator.java)) - Wraps another DataSetIterator, ensuring that only a specified (maximum) number of minibatches (DataSet) objects are returned between resets. Can be used to 'cut short' an iterator, returning only the first N DataSets.
- **EarlyTerminationMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/EarlyTerminationMultiDataSetIterator.java)) - The MultiDataSetIterator version of **EarlyTerminationDataSetIterator**.
- **ExistingDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ExistingDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` or ```Iterable<DataSet>``` to a DataSetIterator. Does not split the underlying DataSet objects.
- **FileDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/file/FileDataSetIterator.java)) - An iterator that iterates over DataSet files that have been previously saved with ```DataSet.save(File)```. Supports randomization, filtering, different output batch size vs. saved DataSet batch size etc.
- **FileMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/file/FileMultiDataSetIterator.java)) - A MultiDataSet version of **FileDataSetIterator**.
- **FileSplitDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/FileSplitDataSetIterator.java)) - Simple iterator working with list of files. File to DataSet conversion will be handled via provided FileCallback implementation.
- **IteratorDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/IteratorDataSetIterator.java)) - Convert an ```Iterator<DataSet>``` to a DataSetIterator. Unlike **ExistingDataSetIterator**, the underlying DataSet objects may be split/combined, i.e. the minibatch size may differ for the output, vs. the input iterator.
- **IteratorMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/IteratorMultiDataSetIterator.java)) - The ```Iterator<MultiDataSet>``` version of **IteratorDataSetIterator**.
- **JointMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/JointMultiDataSetIterator.java)) - This dataset iterator combines multiple DataSetIterators into 1 MultiDataSetIterator. Values from each iterator are joined on a per-example basis, i.e. the values from each DataSet are combined as different feature arrays for a multi-input neural network. Labels can come from either one of the underlying DataSetIterators only (if 'outcome' is >= 0) or from all iterators (if outcome is < 0).
- **MultiDataSetIteratorSplitter** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultiDataSetIteratorSplitter.java)) -  This iterator virtually splits given MultiDataSetIterator into Train and Test parts. For example, you have 100000 examples. Your batch size is 32. That means you have 3125 total batches. With split ratio of 0.7 that will give you 2187 training batches, and 938 test batches.
- **MultiDataSetWrapperIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultiDataSetWrapperIterator.java)) - Convert a MultiDataSetIterator to a DataSetIterator. Note that this is only possible if the number of features and labels arrays is equal to 1.
- **MultipleEpochsIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.java)) - Treat multiple passes (epochs) of the underlying iterator as a single epoch, when training.
- **ReconstructionDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ReconstructionDataSetIterator.java)) - Wraps a data set iterator setting the first (feature matrix) as the labels.
- **SamplingDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/SamplingDataSetIterator.java)) - A wrapper for a dataset to sample from. This will randomly sample from the given dataset.
- **ScrollableDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ScrollableDataSetIterator.java)) 
- **ScrollableMultiDataSetIterator** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ScrollableMultiDataSetIterator.java)) 

## <a name="data-datavec">Reading Raw Data: DataVec RecordReaders</a>

Implementing classes for RecordReader: 

- **ArrowRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-arrow/src/main/java/org/datavec/arrow/recordreader/ArrowRecordReader.java)) - Implements a record reader using Arrow.
- **ComposableRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/ComposableRecordReader.java)) - Create a RecordReader that takes RecordReaders and iterates over them and concatenates them.
- **ConcatenatingRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/ConcatenatingRecordReader.java)) - Combine multiple readers into a single reader.
- **CSVRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVRecordReader.java)) - Simple CSV RecordReader.
- **CSVRegexRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVRegexRecordReader.java)) - A **CSVRecordReader** that can split each column into additional columns using regular expressions.
- **ExcelRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-excel/src/main/java/org/datavec/poi/excel/ExcelRecordReader.java)) -  RecordReader for loading rows of an Excel spreadsheet from multiple spreadsheets very similar to the **CSVRecordReader**.
- **FileRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/FileRecordReader.java)) - File reader/writer. 
- **ImageRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/ImageRecordReader.java)) - Image RecordReader. Reads a local file system and parses images of a given height and width.
- **JacksonLineRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/jackson/JacksonLineRecordReader.java)) - Will read a single file line-by-line when .next() is called. It uses Jackson ObjectMapper and FieldSelection to read the fields in each line.
- **JacksonRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/jackson/JacksonRecordReader.java)) - RecordReader using Jackson.
- **JDBCRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-jdbc/src/main/java/org/datavec/api/records/reader/impl/jdbc/JDBCRecordReader.java)) - Iterate on rows from a JDBC datasource and return corresponding records.
- **LibSvmRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/misc/LibSvmRecordReader.java)) - RecordReader for libsvm format, which is closely related to SVMLight format.
- **LineRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/LineRecordReader.java)) - Reads files line by line.
- **ListStringRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/collection/ListStringRecordReader.java)) - Iterates through a list of strings return a record. Only accepts a ListStringInputSplit as input.
- **LocalTransformProcessRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-local/src/main/java/org/datavec/local/transforms/LocalTransformProcessRecordReader.java)) - A wrapper around the **TransformProcessRecordReader** that uses the LocalTransformExecutor instead of the TransformProcess methods.
- **MatlabRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/misc/MatlabRecordReader.java)) - MATLAB RecordReader.
- **NativeAudioRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-audio/src/main/java/org/datavec/audio/recordreader/NativeAudioRecordReader.java)) - Native audio file loader using FFmpeg.
- **ObjectDetectionRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/objdetect/ObjectDetectionRecordReader.java)) - An image RecordReader for object detection.
- **RegexLineRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/regex/RegexLineRecordReader.java)) - Read a file, one line at a time, and split it into fields using a regular expression.
- **SVMLightRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/misc/SVMLightRecordReader.java)) - Record reader for SVMLight format, which can generally be described as `LABEL INDEX:VALUE INDEX:VALUE ...`.
- **TfidfRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-nlp/src/main/java/org/datavec/nlp/reader/TfidfRecordReader.java)) -  Term frequency–inverse document frequency (TFIDF) RecordReader (wraps a TFIDF vectorizer for delivering labels and conforming to the RecordReader interface).
- **TransformProcessRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/transform/TransformProcessRecordReader.java)) - This wraps a RecordReader with a TransformProcess and allows every Record that is returned by the RecordReader to have a transform process applied before being returned.
- **WavFileRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-audio/src/main/java/org/datavec/audio/recordreader/WavFileRecordReader.java)) - WAV file loader.
- **ImageRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/ImageRecordReader.java)) - Image RecordReader. Reads a local file system and parses images of a given height and width. All images are rescaled and converted to the given height, width, and number of channels.

Implementing classes for **SequenceRecordReader**: 

- **CodecRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-codec/src/main/java/org/datavec/codec/reader/CodecRecordReader.java)) - Codec RecordReader for parsing: H.264 (AVC) Main profile, MP3 decoder/encoder, Apple ProRes decoder and encoder, AAC encoder, H264 Baseline profile encoder, Matroska (MKV) demuxer and muxer, MP4 (ISO BMF, QuickTime) demuxer/muxer and tools, MPEG 1/2 decoder (supports interlace), MPEG PS/TS demuxer, Java player applet, VP8 encoder, MXF demuxer.
- **CSVLineSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVLineSequenceRecordReader.java)) - Used for loading univariance (single valued) sequences from a CSV, where each line in a CSV represents an independent sequence, and each sequence has exactly 1 value per time step.
- **CSVMultiSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVMultiSequenceRecordReader.java)) - Used to read CSV-format time series (sequence) data where there are multiple independent sequences in each file.
- **CSVNLinesSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVNLinesSequenceRecordReader.java)) - A CSV Sequence record reader where:
  - all time series are in a single file,
  - each time series is of the same length (specified in constructor), and
  - no delimiter is used between time series.
- **CSVSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVSequenceRecordReader.java)) - This reader is intended to read sequences of data in CSV format, where each sequence is defined in its own file (and there are multiple files). Each line in the file represents one time step.
- **CSVVariableSlidingWindowRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVVariableSlidingWindowRecordReader.java)) - A sliding window of variable size across an entire CSV file.
- **JacksonLineSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/jackson/JacksonLineSequenceRecordReader.java)) - The sequence record reader version of **JacksonLineRecordReader**.
- **LocalTransformProcessSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-local/src/main/java/org/datavec/local/transforms/LocalTransformProcessSequenceRecordReader.java)) 
- **NativeCodecRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-codec/src/main/java/org/datavec/codec/reader/NativeCodecRecordReader.java)) - An implementation of the CodecRecordReader that uses JavaCV and FFmpeg.
- **RegexSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-api/src/main/java/org/datavec/api/records/reader/impl/regex/RegexSequenceRecordReader.java)) - Read an entire file (as a sequence), one line at a time and split each line into fields using a regular expression.
- **TransformProcessSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-local/src/main/java/org/datavec/local/transforms/LocalTransformProcessSequenceRecordReader.java))

### <a name="data-norm">Data Normalization</a>

ND4J provides a number of classes for performing data normalization. These are implemented as DataSetPreProcessors.
The basic pattern for normalization is as follows:

1. Create your (unnormalized) DataSetIterator or MultiDataSetIterator: ```DataSetIterator myTrainData = ...```
2. Create the normalizer you want to use: ```NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();```
3. Fit the normalizer: ```normalizer.fit(myTrainData)```
4. Set the normalizer/preprocessor on the iterator: ```myTrainData.setPreProcessor(normalizer);```
   End result: the data that comes from your DataSetIterator will now be normalized.

In general, you should fit *only* on the training data, and do ```trainData.setPreProcessor(normalizer)``` and ```testData.setPreProcessor(normalizer)``` with the same/single normalizer that has been fit on the training data only.

Note that where appropriate (NormalizerStandardize, NormalizerMinMaxScaler) statistics such as mean, standard deviation, min and max are shared across time (for time series) and across image `x`/`y` locations. In the case of image data, statistics are not shared across depth/channels.

🔗 [Data normalization example](https://github.com/eclipse/deeplearning4j-examples/blob/master/Deeplearning4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/PreprocessNormalizerExample.java)



**Available normalizers: DataSet / DataSetIterator**

- **ImagePreProcessingScaler** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.java)) - Applies min-max scaling to image activations. Default settings do 0-255 input to 0-1 output (but is configurable). Note that unlike the other normalizers here, this one does not rely on statistics (mean/min/max etc) collected from the data, hence the ```normalizer.fit(trainData)``` step is unnecessary (is a no-op).
- **NormalizerStandardize** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/NormalizerStandardize.java)) - Normalizes each feature value independently (and optionally label values) to have 0 mean and a standard deviation of 1.
- **NormalizerMinMaxScaler** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/NormalizerMinMaxScaler.java)) - Normalizes each feature value independently (and optionally label values) to lie between a minimum and maximum value (by default between 0 and 1).
- **VGG16ImagePreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/VGG16ImagePreProcessor.java)) - This is a preprocessor specifically for VGG16. It subtracts the mean RGB value—computed on the training set—from each pixel, as reported in [Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
- **RGBtoGrayscaleDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/RGBtoGrayscaleDataSetPreProcessor.java)) - The RGBtoGrayscaleDataSetPreProcessor will turn a DataSet of a RGB image into a grayscale one. NOTE: Expects data format to be NCHW. After processing, the channel dimension is eliminated. (NCHW -> NHW)
- **PermuteDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/PermuteDataSetPreProcessor.java)) - The PermuteDataSetPreProcessor will rearrange the dimensions. There are two predefined permutation types: from NCHW to NHWC and from NHWC to NCHW. Or, pass the new order to the ctor. 
- **LabelLastTimeStepPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/LabelLastTimeStepPreProcessor.java)) - Used to extract the labels from a 3D format (shape: [minibatch, nOut, sequenceLength]) to a 2D format (shape: `[minibatch, nOut]`) where the values are the last time step of the labels.
- **ImageFlatteningDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/ImageFlatteningDataSetPreProcessor.java)) - Used to flatten a 4D CNN features array to a flattened 2D format (for use in networks such as a DenseLayer/multi-layer perceptron).
- **CropAndResizeDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/CropAndResizeDataSetPreProcessor.java)) - Crop and resize the processed dataset. 
  📝 **Note**: The data format must be NHWC.
- **CompositeDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/CompositeDataSetPreProcessor.java)) - A simple Composite DataSetPreProcessor - allows you to apply multiple DataSetPreProcessors sequentially on the one DataSet, in the order they are passed to the constructor.

**Available normalizers: MultiDataSet / MultiDataSetIterator**

- **ImageMultiPreProcessingScaler** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/ImageMultiPreProcessingScaler.java)) - A MultiDataSet/MultiDataSetIterator version of **ImagePreProcessingScaler**.
- **MultiNormalizerStandardize** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerStandardize.java)) - MultiDataSet/MultiDataSetIterator version of **NormalizerStandardize**.
- **MultiNormalizerMinMaxScaler** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerMinMaxScaler.java)) - MultiDataSet/MultiDataSetIterator version of **NormalizerMinMaxScaler**.
- **MultiNormalizerHybrid** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/MultiNormalizerHybrid.java)) - A MultiDataSet normalizer that can combine different normalization types such as standardize and min/max for different input/feature and output/label arrays.
- **CompositeMultiDataSetPreProcessor** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/CompositeMultiDataSetPreProcessor.java)) - Allows you to apply multiple MultiDataSetPreProcessors sequentially on the one MultiDataSet, in the order they are passed to the constructor.

### <a name="data-spark">Spark Network Training Data Classes</a>

🔗 [Deeplearning4j on Spark: How To Build Data Pipelines](/docs/{{page.version}}/deeplearning4j-scaleout-data-howto)

- **MapFileRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-hadoop/src/main/java/org/datavec/hadoop/records/reader/mapfile/MapFileRecordReader.java)) - A RecordReader implementation for reading from a Hadoop `org.apache.hadoop.io.MapFile`.
- **MapFileSequenceRecordReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-hadoop/src/main/java/org/datavec/hadoop/records/reader/mapfile/MapFileSequenceRecordReader.java)) - A SequenceRecordReader implementation for reading from a Hadoop `org.apache.hadoop.io.MapFile`.
- **SparkSourceDummyReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-scaleout/spark/dl4j-spark/src/main/java/org/deeplearning4j/spark/datavec/iterator/SparkSourceDummyReader.java)) - Dummy reader for use in IteratorUtils.
- **SparkSourceDummySeqReader** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-scaleout/spark/dl4j-spark/src/main/java/org/deeplearning4j/spark/datavec/iterator/SparkSourceDummySeqReader.java))

## <a name="transfer">Transfer Learning</a>

🔗 [Main transfer learning page](/docs/{{page.version}}/deeplearning4j-nn-transfer-learning)

Deeplearning4j has classes/utilities for performing transfer learning, that is, taking an existing network, and modifying some of the layers, optionally freezing other layers so that their parameters don't change. For example, an image classifier could be trained on ImageNet, then applied to a new/different dataset. Both MultiLayerNetwork and ComputationGraph can be used with transfer learning, frequently starting from a pre-trained model from the model zoo (see next section), though any MultiLayerNetwork/ComputationGraph can be used.

🔗 [Transfer learning examples](https://github.com/eclipse/deeplearning4j-examples/tree/master/Deeplearning4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16)

The main class for transfer learning is [TransferLearning](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/transferlearning/TransferLearning.java). This class has a builder pattern that can be used to add/remove layers, freeze layers, etc.
[FineTuneConfiguration](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.java) can be used here to specify the learning rate and other settings for the non-frozen layers.

## <a name="zoo">Trained Model Library - Model Zoo</a>

🔗 [Deeplearning4j Model Zoo](/docs/{{page.version}}/deeplearning4j-zoo-models)

Deeplearning4j provides a 'model zoo'—a set of pretrained models that can be downloaded and used either as-is (for image classification, for example) or often for transfer learning.

Models available in Deeplearning4j's model zoo:

- **AlexNet** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/AlexNet.java))
- **Darknet19** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/Darknet19.java))
- **FaceNetNN4Small2** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/FaceNetNN4Small2.java))
- **InceptionResNetV1** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/InceptionResNetV1.java))
- **LeNet** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/LeNet.java))
- **NASNet** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/NASNet.java))
- **ResNet50** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/ResNet50.java))
- **SimpleCNN** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/SimpleCNN.java))
- **SqueezeNet** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/SqueezeNet.java))
- **TextGenerationLSTM** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TextGenerationLSTM.java))
- **TinyYOLO** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TinyYOLO.java))
- **U-Net** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/UNet.java))
- **VGG16** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG16.java))
- **VGG19** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG19.java))
- **Xception** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/Xception.java))
- **YOLOv2** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/YOLO2.java))

> 📝 **Note**: Trained Keras models (not provided by Deeplearning4j) may also be imported using Deeplearning4j's Keras model import functionality.

## <a name="keras">Keras Import</a>

🔗 [Deeplearning4j: Keras model import](/docs/{{page.version}}/keras-import-overview)

## <a name="spark">Distributed Training (Spark)</a>

🔗 [Distributed Deep Learning with DL4J and Spark](/docs/{{page.version}}/deeplearning4j-scaleout-intro)

## Cheat sheet code snippets

### Neural networks

Code for configuring common parameters and layers for both `MultiLayerNetwork` and `ComputationGraph`. See [MultiLayerNetwork](/api/{{page.version}}/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html) and [ComputationGraph](/api/{{page.version}}/org/deeplearning4j/nn/graph/ComputationGraph.html) for the full API.

**Sequential networks**

Most network configurations can use the `MultiLayerNetwork` class if they are sequential and simple.

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
    .layer(new ConvolutionLayer.Builder(1, 1)
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
      .build(), "input1")
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
int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; // This only works if your root is clean: only label subdirs.
BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
InputSplit trainData = inputSplit[0];
InputSplit testData = inputSplit[1];

boolean shuffle = false;
ImageTransform flipTransform1 = new FlipImageTransform(rng);
ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
ImageTransform warpTransform = new WarpImageTransform(rng, 42);
List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
  new Pair<>(flipTransform1, 0.9),
    new Pair<>(flipTransform2, 0.8),
    new Pair<>(warpTransform, 0.5));

ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);
DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

// training dataset
ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(trainData);
DataSetIterator trainingIterator = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);

// testing dataset
ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(testData);
DataSetIterator testingIterator = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numLabels);

// early stopping configuration, model saver, and trainer
EarlyStoppingModelSaver saver = new LocalFileModelSaver(System.getProperty("user.dir"));
EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) // Max of 50 epochs
    .evaluateEveryNEpochs(1)
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) // Max of 20 minutes
    .scoreCalculator(new DataSetLossCalculator(testingIterator, true))     // Calculate test set score
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

We recommend having a look at the [DataVec examples](https://github.com/eclipse/deeplearning4j-examples/tree/master/datavec-examples/src/main/java/org/datavec/transform) before creating more complex transformations.

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

For advanced evaluation, the code snippet below can be adapted into training pipelines. Use this when the built-in `neuralNetwork.eval()` method outputs confusing results, or if you need to examine raw data.

```java
// Evaluate the model on the test set
Evaluation eval = new Evaluation(numClasses);
INDArray output = neuralNetwork.output(testData.getFeatures());
eval.eval(testData.getLabels(), output, testMetaData); // Note we are passing in the test set metadata here

// Get a list of prediction errors, from the Evaluation object
// Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
List<Prediction> predictionErrors = eval.getPredictionErrors();
System.out.println("\n\n+++++ Prediction Errors +++++");
for (Prediction p : predictionErrors) {
    System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
        + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
}

// We can also load the raw data:
List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);
for (int i = 0; i < predictionErrors.size(); i++) {
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
List<Prediction> list1 = eval.getPredictions(1, 2);                 // Predictions: actual class 1, predicted class 2
List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     // All predictions for predicted class 2
List<Prediction> list3 = eval.getPredictionsByActualClass(2);       // All predictions for actual class 2
```

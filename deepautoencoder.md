---
title: 
layout: default
---

# Deep autoencoders

*to cut straight to the code, [click here](../deepautoencoder.html#initiate)*

Deep autoencoders are a special form of deep-belief net that typically have four or five layers to represent the encoder itself, which is just one half of the net. The layers are a series of RBMs, with several particularities that we'll discuss below. 

Processing Mnist, a deep autoencoder will use binary transformations after each layer. They can also be used for other types of datasets, with real-valued data which you would use Gaussian rectified RBMs. 

Let’s define an example encoder:
    
     784 (input) ----> 1000 ----> 500 ----> 250 ----> 30

If, say, the input fed to the network is 784 pixels (the square of the 28x28 pixel images in the Mnist dataset), then the first layer of the deep autoencoder should have 1000 parameters -- slightly larger. 

This may seem counterintuitive, because having more parameters than input is a good way to overfit a network. In this case, expanding the parameters, and in a sense expanding the features of the input itself, will make the eventual decoding of the autoencoded data possible. 

This is due to the representational capacity of sigmoid-belief units, a form of transformation used with each layer. Sigmoid belief units can’t represent as much as information and variance as real-valued data. The expanded first layer is a way of compensating for that.

The layers will be 1000, 500, 250, 100 nodes wide, respectively, until at the end, the net produces a vector 30 numbers long. This 30-number vector is the last layer of the first, pretraining, half of the deep autoencoder, and it is the product of a normal RBM, rather than an output layer such as Softmax or logistic regression. 

### the decoding half

Those 30 numbers are an encoded version of the 28x28 pixel image. The second half of a deep autoencoder actually learns how to decode the vector, which becomes the input.

The decoding half of a deep autoencoder is a feed-forward net with layers 100, 250, 500 and 1000 nodes wide, respectively. Those layers initially have the same weights as their counterparts in the pretraining net, except that the weights are transposed. (They are not initialized randomly.) 

		784 (output) <---- 1000 <---- 500 <---- 250 <---- 30

The decoding half of a deep autoencoder is the part that learns to reconstruct the image. It does so with a second feed-forward net which also conducts back propagation. The back propagation happens through reconstruction entropy.

In other words, to train a deep autoencoder, pretrain a net a your choice. The pretraining can leverage DL4J’s distributed architecture. 

You then feed this pretrained net into a DeepAutoEncoder object that will produce the associated decoder, and from there you call "finetune." This type of fine-tuning doesn't use labels for classification. It's using the pretrained net's output as its input, in order to reconstruct. 

If you don’t have a prebuilt net, just call "train" on a net that's been passed in, which will then have the decoder copied from the encoder’s architecture.

### training nuances

At the stage of the decoder's backpropagation, the learning rate should be lowered, or made slower. It should be somewhere between 1e-3 and 1e-6, depending on whether you're handling binary or continuous data, respectively.

### image search

As we mentioned above, deep autoencoders are capable of compressing images into 30-number vectors. So searching for images is as simple as uploading an image, which the search engine will then compress to 30 numbers, and compare that vector to all the others in its index. 

Vectors containing similar numbers will be returned for the search query, and translated into their appropriate image. 

### topic modeling & information retrieval (ir)

Deep autoencoders are highly useful in topic modeling, or statistically modeling abstract topics that are distributed across a collection of documents. 

This, in turn, is an important step in cognitive computing, since question answering computing systems such as Watson match questions to answers when they share a topic. 

That is, autoencoders can be used for topic modeling, and employed for classification and prediction to advance cognitive computing. 

In brief, each document in a collection is converted to a Bag-of-Words (i.e. a set of word counts) and those word counts are scaled to decimals between 0 and 1, which may be thought of as the probability of a word occurring in the doc. 

The scaled word counts are then fed into a deep-belief network, a stack of restricted Boltzmann machines, which themselves are just a subset of feedforward-backprop autoencoders. Those deep-belief networks, or DBNs, compress each document to a set of 10 numbers through a series of sigmoid transforms that map it onto the feature space. 

Each document’s number set, or vector, is then introduced to the same vector space, and its distance from every other document-vector measured. Roughly speaking, nearby document-vectors fall under the same topic. 

For example, one document could be the “question” and others could be the “answers,” a match the software would make using vector-space measurements. 

###<a name="initiate">initiating a deep autoencoder</a> 

You set up a deep autoencoder like this:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/deepautoencoder/DeepAutoEncoderSDA.java?slice=52:62"></script>

Deep autoencoders, when they employ Hessian Free, require no pre-training phase. They only finetune. 

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/deepautoencoder/DeepAutoEncoderSDA.java?slice=69:74"></script>

Rather than producing an f1 score, as other nets do, deep autoencoders' performance is gauged by their reconstructions. 

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/deepautoencoder/DeepAutoEncoderSDA.java?slice=79:86"></script>

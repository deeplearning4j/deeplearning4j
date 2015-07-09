---
title: 
layout: default
---

# Recurrent Nets

Recurrent nets are a powerful set of artificial neural nets especially useful for processing sequential data such as sound, time series (sensor) data or written natural language. A version of recurrent networks was used by DeepMind in their work playing video games with autonomous agents.* 

Recurrent nets differ from feedforward nets because they include a feedback loop, whereby output from step n-1 is fed back to the net to affect the outcome of step n, and so forth for each subsequent step. For example, if a net is exposed to a word letter by letter, and it is asked to guess each following letter, the first letter of a word will help determine what a recurrent net thinks the second letter will be, etc. 

This differs from a feedforward network, which learns to classify each handwritten numeral of MNIST independently according to the pixels it is exposed to from a single example, without referring to the preceding example to adjust its predictions. Feedforward networks accept one input at a time, and produce one output. Recurrent nets don't face the same one-to-one constraint.

While some forms of data, like images, do not seem to be sequential, they can be understood as sequences when fed into a recurrent net. Consider an image of a handwritten word. Just as recurrent nets process handwriting, converting each cursive image into a letter, and using the beginning of a word to guess how that word will end, so nets can treat part of any image like letters in a sequence. A neural net roving over a large picture may learn from each region what the neighboring regions are more likely to be.  

## Neural Net Models and Memory

Recurrent nets and feedforward nets both "remember" something about the world, in a loose sense, by modeling the data they are exposed to. But they remember in very different ways. After training, feedforward net produces a static model of the data it has  been shown, and that model can then accept new examples and accurately classify or cluster them. 

In contrast, recurrent nets produce dynamic models -- i.e. models that change over time -- in ways that yield accurate classifications dependent of the context of the examples they're exposed to. 

To be precise, recurrent models include the hidden state that determined the previous classification in a series. In each subsequent step, that hidden state is combined with the new step's input data to produce a) a new hidden state and then b) a new classification. Each hidden state is recycled to produce its modified successor. 

Human memories are also context aware, recycling an awareness of previous states to properly interpret new data. For example, let's take two individuals. One is aware that she is near Jack's house. The other is aware that she has entered an airplane. They will interpret the sounds "Hi Jack!" in two very different ways, precisely because they retain a hidden state affected by their short-term memories and preceding sensations. 

Different short-term memories should be recalled at different times, in order to assign the right meaning to current input. Some of those memories will have been forged recently, and other memories will have been forged many time steps before they are needed. The recurrent net that effectively associates memories and input remote in time is called a Long Short-Term Memory (LSTM), as much as that sounds like an oxymoron.

## Code Example

Recall that Deeplearning4j's multinetwork configuration lets you create a layer in the API simply by naming it. In this case, you create an LSTM. 

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/recurrent/RecurrentLSTMMnistExample.java?slice=27:64"></script>

An explanation of the hyperparameters is given in our [Iris tutorial](http://deeplearning4j.org/iris-flower-dataset-tutorial.html). 

### LSTM Implementations in DL4J

* [LSTM](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/recurrent/LSTM.java)
* [Graves LSTM](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/recurrent/GravesLSTM.java) (Useful for sensor data and time series)

### Resources

* [Recurrent Neural Networks](http://people.idsia.ch/~juergen/rnn.html); Juergen Schmidhuber
* [Modeling Sequences With RNNs and LSTMs](https://class.coursera.org/neuralnets-2012-001/lecture/77); Geoff Hinton
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/); Andrej Karpathy

### Definition

		Recurrent neural networks "allow for both parallel and sequential computation, and in principle can compute anything a traditional computer can compute. Unlike traditional computers, however, RNN are similar to the human brain, which is a large feedback network of connected neurons that somehow can learn to translate a lifelong sensory input stream into a sequence of useful motor outputs. The brain is a remarkable role model as it can solve many problems current machines cannot yet solve." - Juergen Schmidhuber

### Credit Assignment

*Much research in recurrent nets has been led by Juergen Schmidhuber and his students, notably Sepp Hochreiter, who identified the vanishing gradient problem confronted by very deep networks and later invented Long Short-Term Memory (LSTM) recurrent nets, as well as Alex Graves, now at DeepMind.*

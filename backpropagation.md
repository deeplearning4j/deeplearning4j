---
title: Backpropagation & Deep Learning
layout: default
---

# Backpropagation & Deep Learning

*To propagate* is to transmit something (light, sound, motion or information) in a particular direction or through a particular medium. When we discuss backpropagation in deep learning, we are talking about the transmission of information, and that information relates to the error produced by the neural network. 

Neural networks are like new-born babies: They are created ignorant of the world, and it is only through exposure to the world, experiencing it, that their ignorance is slowly revised. Algorithms experience the world through data. So by training a neural network on a relevant dataset, we seek to decrease its ignorance. The way we measure progress is by monitoring the error produced by the network. 

The knowledge of a neural network with regard to the world is captured by its weights, the parameters that alter input data as the signal flows through the neural network towards the final layer that will make a decision about that input. Those decisions are often wrong, because the parameters transforming the signal into a decision are wrong. 

So the parameters of the neural network have a relationship with the error the net produces, and when the parameters change, presumably the error does, too. We change the parameters using an optimization algorithm called gradient descent, which is useful for finding the minimum of a function. We are seeking to minimize the error, which is also known as the *loss function* or the *objective function*.

So a neural propagates the signal of the input data forward through its parameters towards the moment of decision, and then *backprogates* information about the error through the network so that it can alter the parameters one step at a time. 

A *gradient* is a slope whose angle we can measure. Like all slopes, it can be expressed as a relationship between two variables: "y over x", or *rise over run*. In this case, the `y` is the error produced by the neural network, and `x` is the parameter. So the gradient tells us the change we can expect in `y` with regard to `x`. 

To obtain this information, we must use differential calculus, which enables us to measure *instantaneous rates of change*, which in this case is the tangent of a changing slope expressed the relationship of the parameter to the net's error. 

Obviously, a neural network has many parameters, so what we're really measuring are the *[partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative)* of each parameter's contribution to the total change in error. 

What's more, neural networks have parameters that process the input data sequentially, one after another. Therefore, backpropagation establishes the relationship between the neural network's error and the parameters of the net's last layer; then it establishes the relationship between the parameters of the neural net's last layer those the parameters of the second-to-last layer, and so forth, in an application of the *[chain rule of calculus](https://en.wikipedia.org/wiki/Chain_rule)*. 

### Further Reading About Backpropagation

* [Backpropagation](https://brilliant.org/wiki/backpropagation/)

## <a name="intro">More Machine Learning Tutorials</a>

For people just getting started with deep learning, the following tutorials and videos provide an easy entrance to the fundamental ideas of deep neural networks:

* [Deep Reinforcement Learning](./deepreinforcementlearning.html)
* [Deep Convolutional Networks](./convolutionalnets.html)
* [Recurrent Networks and LSTMs](./lstm.html)
* [Multilayer Perceptron (MLPs) for Classification](./multilayerperceptron.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network.html)
* [Symbolic Reasoning & Deep Learning](./symbolicreasoning.html)
* [Using Graph Data with Deep Learning](./graphdata.html)
* [AI vs. Machine Learning vs. Deep Learning](./ai-machinelearning-deeplearning.html)
* [Markov Chain Monte Carlo & Machine Learning](/markovchainmontecarlo.html)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine.html)
* [Eigenvectors, PCA, Covariance and Entropy](./eigenvector.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary.html)
* [Word2vec and Natural-Language Processing](./word2vec.html)
* [Deeplearning4j Examples via Quickstart](./quickstart.html)
* [Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs) (A seven-video series)
* [Inference: Machine Learning Model Server](./modelserver.html)

---
layout: default
---

# training tricks

There is an art to training neural networks, just as there is an art to training tigers to jump through a ring of fire. A little to the left or a little to the right and disaster strikes.

Many of these tips have already been discussed in the academic literature. Our purpose is to consolidate them in one site and express them as clearly as possible. 

1. When constructing distributed neural networks, for example, it’s important to lower the learning rate; that is, a small step size as you make your gradient descent is required. Otherwise the weights diverge, and when weights diverge, your net has ceased to learn. Why is a lower learning rate required? Distributed neural networks use parameter averaging, which speeds up learning, so you need to correct for that acceleration by slowing the algorithm down elsewhere.

  A default value for your learning rate is 0.01. Higher than that makes the weights diverge. Your aim is to minimize reconstruction entropy, but that can’t occur if weights can no longer learn features and classify. Each weight represents a neuron’s on-off function, the likelihood that it will be activated. If it gets too large, it becomes meaningless. 

  The learning rate represents a series of calculus operations that measure the derivative of each step.

2. When creating your hidden layers, give them fewer neurons than your input data. If the hidden-layer nodes are too close to the number of input nodes, you risk reconstructing the identity function. Too many hidden-layer neurons increase the likelihood of noise and overfitting.

  For an input layer of 784, you might choose an initial hidden layer of 500, and a second hidden layer of 250. No hidden layer should be less than a quarter of the input layer’s nodes. And the output layer will simply be the number of labels. 

  Most deep-learning nets have three to four hidden layers, as accuracy tends to decrease beyond that depth. Typical machine learning, of course, has one hidden layer, and those nets are called Perceptrons. 

3. Training neural networks has two steps: pre-training and fine-tuning. You’ll want to pre-train your net first on the raw data, and then save a backup, because fine-tuning will destroy the feature blend of the pretrained net. That is, a good pre-trained net is a product in itself, on which you can iteratively fine tune. In addition, pre-training can occur in several stages. Feed your net one set of raw data, pause, and then resume when you have more to feed it. 

  Two aspects of this process are notable. Pretraining and finetuning have different goals, and different directions. Pretraining is a form of forward propagation. The data moves through the net in one direction. Finetuning is a form of back propagation in which the direction is reversed. Pretraining learns features. Finetuning teaches the network to perform classification. It becomes specialized. 

  It is also interesting to note that Hinton’s network made 30 passes in pretraining to achieve its 99 percent accuracy. 
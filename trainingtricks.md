---
layout: default
---

# Training tricks

There is an art to training neural networks, just as there is an art to training tigers to jump through a ring of fire. A little to the left or the right and disaster strikes.

Many of these tips have already been discussed in the academic literature. Our purpose is to consolidate them in one site and express them as clearly as possible. 

First, know the problem you are trying to solve. This [list of questions](http://deeplearning4j.org/questions.html) will help you clarify your task if you are new to deep learning. 

* Researchers experienced with deep neural networks tend to try three different learning rates -- 1e^-1, 1^e-3, and 1e^-6 -- to get a rough idea of what it should be. 

* They use Nesterov's Momentum.

* They pair activations and loss functions. That is, for classification problems, they tend to pair the softmax activation function with the negative log likelihood loss function. For regression problems, they tend to pair linear (identity) activation functions with a loss function of root mean squared error (RMSE) cross entropy.

* When constructing distributed neural networks, for example, it’s important to lower the learning rate; that is, a smaller step size as you make your gradient descent is required. Otherwise the weights diverge, and when weights diverge, your net has ceased to learn. Why is a lower learning rate required? Distributed neural networks use parameter averaging, which speeds up learning, so you need to correct for that acceleration by slowing the algorithm down elsewhere.

* When creating your hidden layers, give them fewer neurons than your input data. If the hidden-layer nodes are too close to the number of input nodes, you risk reconstructing the identity function. Too many hidden-layer neurons increase the likelihood of noise and overfitting. For an input layer of 784, you might choose an initial hidden layer of 500, and a second hidden layer of 250. No hidden layer should be less than a quarter of the input layer’s nodes. And the output layer will simply be the number of labels. 

  Larger datasets require more hidden layers. Facebook's Deep Face uses nine hidden layers on what we can only presume to be an immense corpus. Many smaller datasets might only require three or four hidden layers, with their accuracy decreasing beyond that depth. As a rule: larger data sets contain more variation, which require more features/neurons for the net to obtain accurate results. Typical machine learning, of course, has one hidden layer, and those shallow nets are called Perceptrons. 

* Training neural networks has two steps: pre-training and fine-tuning. You’ll want to pre-train your net first on the raw data, and then save a backup, because fine-tuning will destroy the feature blend of the pretrained net. That is, a good pre-trained net is a product in itself, on which you can iteratively fine tune. In addition, pre-training can occur in several stages. Feed your net one set of raw data, pause, and then resume when you have more to feed it. 

  Two aspects of this process are notable. Pretraining and finetuning have different goals, and different directions. Pretraining is a form of forward propagation. The data moves through the net in one direction. Finetuning is a form of back propagation in which the direction is reversed. Pretraining learns features. Finetuning teaches the network to perform classification. It becomes specialized. 

  It is also interesting to note that Hinton’s network made 30 passes in pretraining to achieve its 99 percent accuracy. This is only feasible with a speedy, massively parallel network that lowers the cost of iteration.

* Minibatch size should in the ballpark of 10-100. You'll want more than one example per batch, but you also want to ensure that training, on average, gets done faster. Ten is the recommended number here. This goes both for parallel training, as well as single threaded processing.

  In parallel training, when the batch size is too large, any data batches that contain outliers will take longer to train in pretraining or finetuning. On average, smaller batch sizes even out these minibatches, which may have more error attached to them.

* Large datasets require that you pretrain your neural net several times. Only with multiple pretrainings will the algorithm learn to correctly weight features in the context of the dataset. That said, you can run the data in parallel or through a cluster to speed up the pretraining. 

---
title: Performance and practical considerations
layout: default
---

# Performance and practical considerations

A number of other deep-learning frameworks exist, differing from DL4J largely in their use of Python and their audience, which is the research community. They include [Theano](http://deeplearning.net/software/theano/), [PyBrain](http://pybrain.org/), [PyLearn2](http://deeplearning.net/software/pylearn2/) and the [neural network toolbox](http://www.mathworks.com/products/neural-network/) for MatLab.

Another difference between DL4J and other frameworks is performance. 

With the default parameters, our code runs for 1000 pre-training epochs with mini-batches of size 10. This corresponds to performing 5,000,000 unsupervised parameter updates. We use an unsupervised learning rate of 0.01, with a supervised learning rate of 0.1. The DBN itself consists of three hidden layers with 600, 400 and 250 units in each layer, respectively. With conjugate gradient, this configuration achieved a minimal validation error of 0.5 or less with corresponding test error of TK after TK supervised epochs.

On an [[UPDATE THIS -->Intel(R) Xeon(R) CPU X5560 running at 2.80GHz]], using DL4J’s built-in parallelism running on 8 cores, pretraining and fine-tuning took 12 hours.
---
title: Generative Adversarial Networks
layout: default
redirect_from: gan
---

# Generative Adversarial Networks (GANs)

Generative adversarial networks (GANs) are two-part neural networks capable of unsupervised learning that pit one net against the other, thus the "adversarial." 

One neural network called the generator generates data instances and the other, called the discriminator, evaluates them for authenticity; i.e. the discriminator decides whether they belong to the unlabeled training dataset or not. 

For example, the generative network might create images and the evaluating network would accept or reject them, in what is essentially an [actor-critic model](https://arxiv.org/abs/1610.01945). Both nets are trying to optimize a different and opposing objective, or loss, function. 

[GANs were introduced in a paper](https://arxiv.org/abs/1406.2661) by Ian Goodfellow and other researchers at the University of Montreal, including Yoshua Bengio, in 2014. Goodfellow, Bengio and Aaron Courville co-authored [a well-known deep learning textbook](http://www.deeplearningbook.org/). 

While difficult to tune and therefore to use, GANs have stimulated a lot of [interesting research and writing](https://blog.openai.com/generative-models/). 


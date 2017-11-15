---
title: Generative Adversarial Networks
layout: default
redirect_from: gan
---

# Generative Adversarial Networks

Generative adversarial networks (GANs) are two-part neural networks capable of unsupervised learning that pits one net against the other, thus adversarial. 

One neural network generates data instances and the other evaluates them for authenticity; i.e. whether they are real or not, based on the underlying dataset. 

For example, the generative network might create images and the evaluating network would accept or reject them, in what is known as the actor-critic model. Each net is trying to optimize a different and opposing objective, or loss, function. 

[GANs were introduced in a paper](https://arxiv.org/abs/1406.2661) by Ian Goodfellow and other researchers at the University of Montreal, including Yoshua Bengio, in 2014. Goodfellow, Bengio and Aaron Courville co-authored [a well-known deep learning textbook](http://www.deeplearningbook.org/). 

While difficult to tune, GANs have stimulated a lot of [interesting research and writing](https://blog.openai.com/generative-models/). 


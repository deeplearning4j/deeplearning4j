---
layout: default
title: Deeplearning4j Updaters Explained
---

# Deeplearning4j Updaters Explained

This page and the explanations that follow assume that readers know how [Stochastic Gradient Descent](../glossary.html#stochasticgradientdescent) works.

The main difference among the updaters described below is how they treat the learning rate. 

## Stochastic Gradient Descent

![Alt text](../img/updater_math1.png)

`Theta` (weights) is getting changed according to the gradient of the loss with respect to theta.

`alpha` is the learning rate. If alpha is very small, convergence on an error minimum will be slow. If it is very large, the model will diverge away from the error minimum, and learning will cease.

Now, the gradient of the loss (L) changes quickly after each iteration due to the diversity among training examples. Have a look at the convergence below. The updater takes small steps, but those steps zig-zag back and forth on their way to reaching to an error minimum.

![Alt text](../img/updater_1.png)

* [SGDUpdater in Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/SgdUpdater.java)

## Momentum

To stop the zig-zagging, we introduce *momentum*. Momentum takes knowledge from previous steps and applies it to where the updater should be heading. We are introducing a new hyperparameter `μ`.

![Alt text](../img/updater_math2.png)

We will use the concept of momentum again later. (Don't confuse it with moment, which is also used later.)

![Alt text](../img/updater_2.png)

The image above represents SGD equipped with momentum.

* [Nesterov's Momentum Updater in Deeplearnign4j](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/NesterovsUpdater.java)

## Adagrad

Adagrad scales alpha for each parameter according to the history of gradients (previous steps) for that parameter. That's basically done by dividing the current gradient in the update rule by the sum of previous gradients. As a result, when the gradient is very large, alpha is reduced, and vice-versa.

![Alt text](../img/updater_math3.png)

* [AdaGradUpdater in Deeplearning4j](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdaGradUpdater.html)

## RMSProp

The only difference between RMSProp and Adagrad is that the gt term is calculated by exponentially decaying the average and not the sum of gradients.

![Alt text](../img/updater_math4.png)

Here `g_t` is called the second order moment of `δL`. Additionally, a first-order moment mtmt can also be introduced.

![Alt text](../img/updater_math5.png)

Adding momentum as in the first case:

![Alt text](../img/updater_math6.png)

And finally collecting a new `theta` as we have done in the first example:

![Alt text](../img/updater_math7.png)

* [RMSPropUpdater in Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/RmsPropUpdater.java)

## AdaDelta

AdaDelta also uses an exponentially decaying average of `g_t`, which was our second moment of gradient. But without using the alpha we were traditionally using as learning rate, it introduces `x_t`, which is the second moment of `v_t`.

![Alt text](../img/updater_math8.png)

* [AdaDeltaUpdater in Deepelearning4j](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdaDeltaUpdater.html)

## ADAM

ADAM uses both first-order moment mt and second-order moment `g_t`, but they both decay over time. Step size is approximately `±α`. Step size will decrease as it approaches the minimum.

![Alt text](../img/updater_math9.png)

* [AdamUpdater in Deeplearning4j](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdamUpdater.html)

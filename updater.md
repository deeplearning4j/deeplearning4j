---
layout: default
title: Deeplearning4j Updaters Explained
---

# Deeplearning4j Updaters Explained

We assume here that readers know how Stochastic Gradient Descent works.

The main difference in updaters is how they treat the learning rate.

## Stochastic Gradient Descent

![Alt text](../img/udpater_math1.png)

`Theta` (weights) is getting changed according to the gradient of the loss with respect to theta.

`alpha` is the learning rate. If it is very small, convergence will be slow. If it is very large, the model will diverge.

Now, the gradient of the loss (L) changes quickly after each iteration due to the diversity of each training example. Have a look at the convergence below. We are taking small steps but they are quite zig-zag (even though we slowly reach to a loss minima).

![Alt text](../img/udpater_1.png)

## Momentum

To overcome this, we introduce `momentum`. Basically taking knowledge from previous steps about where we should be heading. We are introducing a new hyperparameter μμ

![Alt text](../img/udpater_math2.png)

We will use the concept of momentum again later.  (Don't confuse it with moment, which is also used later.)

![Alt text](../img/udpater_2.png)

This is the image of SGD equipped with momentum.

## Adagrad

Adagrad scales alpha for each parameter according to the history of gradients (previous steps) for that parameter which is basically done by dividing current gradient in update rule by the sum of previous gradients. As a result, what happens is that when the gradient is very large, alpha is reduced and vice-versa.

![Alt text](../img/udpater_math3.png)

## RMSProp

The only difference RMSProp has with Adagrad is that the gtgtterm is calculated by exponentially decaying average and not the sum of gradients.

gt+1=γgt+(1−γ)δLgt+1=γgt+(1−γ)δL(θ)2(θ)2
Here gtgt is called the second order moment of δLδL . Additionally, a first order moment mtmt can also be introduced.
mt+1=γmt+(1−γ)δLmt+1=γmt+(1−γ)δL(θ)(θ)
gt+1=γgt+(1−γ)δLgt+1=γgt+(1−γ)δL(θ)2(θ)2
Adding momentum as in the first case,
vt+1=μvt−αδL(θ)gt+1−m2t+1+ϵ‾‾‾‾‾‾‾‾‾‾‾‾‾‾√vt+1=μvt−αδL(θ)gt+1−mt+12+ϵ
And finally collecting new theta as we have done in the first example,
θt+1=θt+vt+1θt+1=θt+vt+1

## AdaDelta

AdaDelta also uses exponentially decaying average of gtgt which was our 2nd moment of gradient. But without using alpha that we were traditionally using as learning rate, it introduces xtxt which is the 2nd moment of vtvt.

gt+1=γgt+(1−γ)▽gt+1=γgt+(1−γ)▽L(θ)2(θ)2
xt+1=γxt+(1−γ)v2t+1xt+1=γxt+(1−γ)vt+12
vt+1=−xt+ϵ‾‾‾‾‾‾√δL(θt)gt+1+ϵ‾‾‾‾‾‾‾‾√vt+1=−xt+ϵδL(θt)gt+1+ϵ
θt+1=θt+vt+1θt+1=θt+vt+1

## Adam

Adam uses both first-order moment mtmt and second-order moment gtgt, but they are both decayed over time. Step size is approximately ±α±α. Step size will decrease as it approaches the minimum.

mt+1=γ1mt+(1−γ1)▽mt+1=γ1mt+(1−γ1)▽L(θt)(θt)
gt+1=γ2gt+(1−γ2)▽gt+1=γ2gt+(1−γ2)▽L(θt)2(θt)2
m̂ t+1=mt+11−γt+11m^t+1=mt+11−γ1t+1
ĝ t+1=gt+11−γt+12g^t+1=gt+11−γ2t+1
θt+1=θt−αm̂ t+1ĝ t+1‾‾‾‾√+ϵ

[From Quora](https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM/answer/Rajarshee-Mitra?srid=Xs23&share=bc33d009)

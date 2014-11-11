---
title: 
layout: default
---

DeepLearning4j uses [nd4j](http://nd4j.org/) as a computational kernel for matrix operations. Maven may also be hard to get up and
running. In order to get started with deep learning4j, you first need to pick an [nd4j backend](http://nd4j.org/downloads.html)

This backend can be GPUs or native computations (if pure java is what you want, we are working on that as well with the vectorz project)


##Why backends?

Many in the deep learning community have standardized on cuda GPUs for much of their parallel processing
and matrix computations. Unfortunately, in industry (the rest of us) we have either legacy hardware
or limited options. However, we may have lots of these machines laying around. If that's the case, throwing CPUs
at a problem should be an option (not the only one). GPUs should also be an option however. I took it upon myself to create
nd4j primarily for the fact that currently no JVM blas based libraries allows you to have a swappable (no code rewrite)
interface to different fast matrix operations. Swappable backends (ala [slf4j](http://slf4j.org/)) was the only clear answer.

A common API for creating machine learning algorithms should also be a real goal. No one wants to rewrite their libraries
if they find that their matrix run time is faster (I did this once already).

##Downloads

Below you will find bundled downloads of deeplearning4j for GPUs, native, among other components.

#GPUs

Jcublas



##Native

Jblas


Netlib blas


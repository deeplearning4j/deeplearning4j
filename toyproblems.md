---
title: Toy problems
layout: default
---

## Toy problems 

People often come to us with toy examples they see in text books with small numbers of examples.

Deeplearning4j's defaults assume lots of data and its often confusing for beginners. Here are a few tuning tips to ensure
your beginning experience with toy problems goes well. We often advocate for beginners to learn python or matlab
due to the fact that deeplearning4j assumes a very different audience.


      1. minibatch(false) in your config - this will prevent the gradient from being normalized minibatch learning
          performs a divide by n which normalizes the learned gradient. When you have the whole problem in memory
          there is no reason to do this.
      
      2. double precision: Neural nets can learn from 32 bit when exposed to lots of data. Use 64 bit (double precision for 
          better results): http://nd4j.org/userguide#miscdatatype
          
      3. Use the sgd updater with the sgd optimization algo. This will prevent complicated things like the line search
         from being activated.

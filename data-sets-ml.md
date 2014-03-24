---
title: 
layout: default
---

*previous* - [mnist for deep-belief networks](../mnist.html)
# data sets and machine learning

Garbage in, garbage out. 

Deep learning, and machine learning more generally, needs a good training set to work properly. Collecting and constructing the training set -- a sizable body of known data -- takes time and domain-specific knowledge of where and how to gather relevant information. The training set acts as the benchmark against which deep-learning nets are trained. That is what they learn to reconstruct when they're eventually unleashed on unstructured data. 

At this stage, human intervention is necessary to find the right raw data and transform it into a numerical representation that the deep-learning algorithm can understand. Building a training set is, in a sense, pre-pre-training. For a text-based training set, you may even have to do some feature creation. 

Training sets that require much time or expertise can serve as a proprietary edge in the competitive world of data science and problem solving. The nature of the expertise is largely in telling your algorithm what matters to you through the training set. 

It involves telling a story -- through the initial data you select -- that will guide your deep-learning nets as they extrapolate the significant features, both in the training set and in the unstructured data they've been created to study.

To create a useful training set, you have to understand the problem you're solving; i.e. what you want your deep-learning nets to pay attention to. 

### three sets

Machine learning typically works with three data sets: training, dev and test. All three should randomly sample a larger body of data.

The first set you use is the training set, the largest of the three. Running a training set through a neural network teaches the net how to weigh different features, assigning them coefficients according to their likelihood of minimizing errors in your results.

Those coefficients, also known as metadata, will be contained in vectors, one for each each layer of your net. They are one of the most important results you will obtain from training a neural network.

The second set, known as dev, is what you optimize against. While the first encounter between your algorithm and the data was unsupervised, the second requires your intervention. This is where you turn the knobs, adjusting coefficients to see which changes will help your network best recognize patterns by focusing on the crucial features.

The third set is your test. It functions as a seal of approval, and you don’t use it until the end. After you’ve trained and optimized your data, you test your neural net against this final random sampling. The results it produces should validate that your net accurately recognizes images, or at least [x] percentage of them.

If you don’t achieve validation, go back to dev, examine the quality of your data and look at your pre-processing techniques. If they do, that’s validation you can publish.

Now that you have the overview, we'll show you how to create [custom datasets](../customdatasets.html).
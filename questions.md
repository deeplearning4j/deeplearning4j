---
title: Questions to Ask When Applying Deep Learning
layout: default
---

# Questions to Ask When Applying Deep Learning

We can't answer these questions for you, because the responses will be specific to the problem you seek to solve. But we hope this will serve as a useful checklist to clarify how you initially approach your choice of algorithms and tools: 

* **Is my problem supervised or unsupervised? If supervised, is it a classification or regression problem?** Supervised learning has a teacher. That teacher takes the form of a training set that establishes correlations between two types of data, your input and your output. You may want to apply labels to images, for example. In this classification problem, your input is raw pixels, and your output is the name of whatever's in the picture. In a regression example, you might teach a neural net how to predict continuous values such as housing price based on an input like square-footage. Unsupervised learning, on the other hand, can help you detect similarities and anomalies simply by analyzing unlabeled data. Unsupervised learning has no teacher; it can be applied to use cases such as image search and fraud detection.
* **If supervised, how many labels am I dealing with?** The more labels you need to apply accurately, the more computationally intensive your problem will be. ImageNet has a training set with about 1000 classes; the Iris dataset has just 3. 
* **What's my batch size?** A batch is a bundle of examples, or instances from your dataset, such as a group of images. In training, all the instances of a batch are passed through the net, and the error resulting from the net's guesses is averaged from all instances in the batch and then used to update the weights of the model. Larger batches mean you wait longer between each update, or learning step. Small batches mean the net learns less about the underlying dataset with each batch. Batch sizes of 1000 can work well on some problems, if you have a lot of data and you're looking for a smart default to start with. 
* **How many features am I dealing with?** The more features you have, the more memory you'll need. With images, the features of the first layer equal the number of pixels in the image. So MNIST's 28*28 pixel images have 784 features. In medical diagnostics, you may be looking at 14 megapixels. 
* **Another way to ask that same question is: What is my architecture?** [Resnet](http://arxiv.org/abs/1512.03385), the Microsoft Research net that won the most recent ImageNet competition, had 150 layers. All other things being equal, the more layers you add, the more features you have to deal with, the more memory you need. A dense layer in a multilayer perceptron (MLP) is a lot more feature intensive than a convolutional layer. People use convolutional nets with subsampling precisely because they get to aggressively prune the features they're computing. 
* **How am I going to tune my neural net?** Tuning neural nets is still something of a dark art for a lot of people. There are a couple of ways to go about it. You can tune empirically, looking at the f1 score of your net and then adjusting the hyperparameters. You can tune with some degree of automation using tools like [hyperparameter optimization](https://github.com/deeplearning4j/Arbiter). And finally, you can rely on heuristics like [a GUI](../visualization.html), which will show you exactly how quickly your error is decreasing, and what your activation distribution looks like. 
* **How much data will be sufficient to train my model? How do I go about finding that data?** 
* **Hardware: Will I be using GPUs, CPUs or both? Am I going to rely on a single-system GPU or a distributed system?** A lot of research is being conducted on 1-4 GPUs. Enterprise solutions usually require more and have to work with large CPU clusters as well. 
* **What's my data pipeline? How do I plan to extract, transform and load the data (ETL)? Is it in an Oracle DB? Is it on a Hadoop cluster? Is it local or in the cloud?** 
* **How will I featurize that data?** Even though deep learning extracts features automatically, you can lighten the computational load and speed training with different forms of feature engineering, especially when the features are sparse. 
* **What kind of non-linearity, loss function and weight initialization will I use?** The non-linearity is the activation function tied to each layer of your deep net. It might be sigmoid, rectified linear, or something else. Specific non-linearities often go hand in hand with specific loss functions. 
* **What is the simplest architecture I can use for this problem?** Not everyone is willing or able to apply Resnet to image classification. 
* **Where will my net be trained and where will the model be deployed? What does it need to integrate with?** Most people don't think about these questions until they have a working prototype, at which point they find themselves forced to rewrite their net with more scalable tools. You should be asking whether you'll eventually need to use Spark, AWS or Hadoop, among other platforms. 

Java developers using Deeplearning4j are welcome to join our [Gitter live chat](https://gitter.im/deeplearning4j/deeplearning4j), where our community helps answer these questions case by case. 

## <a name="beginner">Other Beginner's Guides</a>
* [Restricted Boltzmann Machines](../restrictedboltzmannmachine.html)
* [Eigenvectors, Covariance, PCA and Entropy](../eigenvector.html)
* [Word2vec](../word2vec.html)
* [Neural Networks](../neuralnet-overview.html)
* [Neural Networks and Regression](../linear-regression.html)
* [Convolutional Networks](../convolutionalnets.html)

---
title: Import a Keras Model to Deeplearning4j 
layout: default
---

# Keras for Production: Importing Python Models to Deeplearning4j

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

The code for this tutorial is available on [Github](https://gist.github.com/tomthetrainer/f6e073444286e5d97d976bd77292a064).

[Keras](keras.io) is one of the most widely used open-source deep-learning tools for Python. With an API inspired by Torch, it provides a layer of abstraction on top of Theano and TensorFlow to make them easier to use. Keras allows users to import models from most major deep-learning frameworks, including Theano, TensorFlow, Caffe and Torch. And from Keras, it's possible to import those same models into Deeplearning4j. 

That's important chiefly because different frameworks solve different problems, and different programming languages dominate various phases of the deep learning workflow. While Python dominates the stage of data exploration and prototyping, it is not always the best suited for deployment to production. Deeplearning4j integrates closely with other open-source libraries common to the big data stack, such as Hadoop, Spark, Kafka, ElasticSearch, Hive and Pig. 

It is also certified on Cloudera's CDH and Hortonworks's HDP distributions of the Hadoop ecosystem. For deep-learning practitioners seeking to take their neural-net models and put them to work in the the production stack of large organizations, model import from Keras to Deeplearning4j may help. 

Not every architecture supported by other deep learning frameworks is currently supported, but we're working to expand the number of nets that can be imported from Keras to DL4J. 

Python programmers seeking to interface directly with Deeplearning4j may be interested in [ScalNet, its Scala API](https://github.com/deeplearning4j/scalnet).

For more information, please see this page on [model import](https://deeplearning4j.org/model-import-keras).

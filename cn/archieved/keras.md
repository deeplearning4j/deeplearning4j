---
title: 将Keras模型导入Deeplearning4j 
layout: cn-default
---

# Keras的生产应用：将Python模型导入Deeplearning4j

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

本教程的代码可以在[Github](https://gist.github.com/tomthetrainer/f6e073444286e5d97d976bd77292a064)上找到。

[Keras](keras.io)是目前使用最广泛的为Python编写的开源深度学习工具之一。Keras有一套受Torch启发的API，在Theano和TensorFlow的基础上再提供一个抽象层，让此二者更易于使用。Keras允许用户导入来自Theano、TensorFlow、Caffe和Torch等最先进的深度学习框架的模型。而同样的模型可以从Keras导入到Deeplearning4j中。 

这一点很重要，因为不同的学习框架擅长解决不同的问题，而深度学习工作流的各个阶段也都由不同的编程语言所主导。Python虽然主导了数据探索和原型开发阶段，却并不一定最适合生产部署。Deeplearning4j与Hadoop、Spark、Kafka、ElasticSearch、Hive和Pig等大数据技术栈中常见的开源库高度集成。 

Deeplearning4j也获得了两种Hadoop生态系统发行版——Cloudera CDH和Hortonworks HDP的认证。导入Keras模型的功能让Deeplearning4j可以帮助深度学习用户将自己的神经网络模型转移到大型企业生产栈中运行。 

目前我们还不能支持其他深度学习框架提供的所有网络架构，但我们正在不断增加可以从Keras导入到DL4J的网络类型。 

Python程序员如果希望直接与Deeplearning4j对接，可以考虑使用[它的Scala API：ScalNet](https://github.com/deeplearning4j/scalnet)。

更多详情请参阅[模型导入指南](https://deeplearning4j.org/model-import-keras)。

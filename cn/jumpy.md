---
title: Jumpy：面向JVM的NumPy
layout: cn-default
---

# Jumpy：面向JVM的NumPy

Deeplearning4j和ND4J的Python接口主要由三个部分组成：

* Autodiff（开发中）
* Keras模型导入
* Jumpy

[Jumpy](https://github.com/deeplearning4j/jumpy)是科学计算库[ND4J](http://nd4j.org/)（面向JVM的N维数组）的Python接口，可以通过指针来使用ND4J，而不像其他Python工具那样需要依赖网络通信。 

Jumpy接受NumPy数组，因此我们可以直接使用NumPy数组和张量，无需复制数据。总之，Jumpy对于MLlib或PySpark用户而言是一种更好的接口，因为免除数据复制可以提高工作速度和效率。 

Jumpy是对NumPy和[Pyjnius](https://pyjnius.readthedocs.io/en/latest/)的简单包装。Jumpy让您可以在Python工况下使用JVM的自动完成（只需给出类路径和JAR文件即可），同时还能动态生成Java类。PySpark始终在追赶Scala的最新进展，而Jumpy则允许开发者以动态方式自行拓展绑定。

Jumpy相当于一种把张量导入JVM的方法，而在JVM中使用Spark及其他大数据框架是很容易的。 

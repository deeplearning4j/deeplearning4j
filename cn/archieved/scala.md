---
title: Scala、Spark与Deeplearning4j
layout: cn-default
---

# Scala、Spark与Deeplearning4j

希望构建深度学习解决方案的Scala程序员可以使用Deeplearning4j的Scala API [ScalNet](https://github.com/deeplearning4j/scalnet)，或者借助`Builder`模式来使用该Java框架。Skymind的数值运算库[ND4J](http://nd4j.org/)（面向JVM的N维数组）自带名为[ND4S](https://github.com/deeplearning4j/nd4s)的Scala API。

## Scala

Scala是21世纪发明的最令人激动的编程语言之一。它是一种完全支持函数式、对象导向式、命令式和并发式编程的多范式语言。Scala属于强类型语言，而在我们看来，强类型系统是一种便利的自文档化代码。

Scala与JVM兼容，可以利用Java生态系统中的丰富资源，同时也比Java更为精简。我们在ND4J中采用该语言，其语法与广受众多数据科学家青睐的Python语言惊人地相似。Scala可以像Python那样让程序员皆大欢喜，但同时它的速度也相当快，就和Java一样。 

最后，[Apache Spark](./spark.html)是用Scala编写的，而任何宣称支持分布式运行时的库都至少应该能够与Spark对接。Deeplearning4j和ND4J则更上一层楼，因为它们在Spark集群中运行，并且分别拥有名为ScalNet和ND4S的Scala API。 

我们相信Scala将凭借众多优势在数值运算以及深度学习领域占据主导地位。我们认为这将会在Spark上实现。我们也已在努力开发相关工具，促使这一天早日到来。 

### 另请参见

* [文档：基于Spark的Deeplearning4j](./spark.html)
* [课程：Scala编程思想（Atomic Scala）](http://www.atomicscala.com/)－推荐的入门课程
* [Martin Odersky在Coursera上的Scala课程](https://www.coursera.org/learn/progfun1)
* [书籍：Scala的数据科学应用（Scala for Data Science）](https://www.amazon.com/Scala-Data-Science-Pascal-Bugnion/dp/1785281372)
* [视频课程：用Scala解决问题](https://www.youtube.com/user/DrMarkCLewis)
* [学习：Scala编程语言](http://www.scala-lang.org/documentation/)
* [面向Java程序员的Scala教程](http://www.scala-lang.org/docu/files/ScalaTutorial.pdf)（PDF）
* [Scala示例解析（Scala By Example），Martin Odersky著](http://www.scala-lang.org/docu/files/ScalaByExample.pdf)（PDF） 
* [ND4J的Scala介绍](http://nd4j.org/cn/scala.html)
* [我们的早期Scala API](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-scala-api/src/main/scala/org/nd4j/api/linalg)：（[Github上的一个示例](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-scala-api/src/test/scala/org/nd4j/api/linalg/TestNDArray.scala#L18)）
* SF Spark讲座：[基于Spark的Deeplearning4j和基于JVM的数据科学（借助ND4J实现）](https://www.youtube.com/watch?v=LCsc1hFuNac&feature=youtu.be)
* [Adam Gibson的Spark访谈，由Alexy Khrabrov主持](https://www.youtube.com/watch?v=LJPL8sL0Daw&feature=youtu.be)
* [我们的Spark集成](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark)
* [ND4J：面向JVM的科学计算](http://nd4j.org/cn)
* [面向Python开发人员的Scala基础教程](https://bugra.github.io/work/notes/2014-10-18/scala-basics-for-python-developers/)
* [Coursera为何中意Scala](https://tech.coursera.org/blog/2014/02/18/why-we-love-scala-at-coursera/)

部分[采用Scala的企业](http://alvinalexander.com/scala/whos-using-scala-akka-play-framework)：

* AirBnB
* 亚马逊
* Apple
* Ask.com
* 美国电话电报公司
* Autodesk
* 美国银行
* 彭博
* 瑞信
* eBay
* Foursquare
* 卫报
* IBM
* Klout
* 领英
* NASA
* Netflix
* precog
* 西门子
* 索尼
* Twitter
* Tumblr
* 瑞银
* The Weather Channel
* 施乐
* Yammer

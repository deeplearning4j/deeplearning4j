---
title: 深度学习入门指南
layout: cn-default
---

# 深度学习如何入门？

具体的入门方式取决于您已经掌握的知识。 

要理解并应用深度学习，必须先掌握线性代数、微积分和统计学，还应当具备编程及机器学习的知识。 

就Deeplearning4j而言，您应当熟悉Java语言，并且熟练掌握IDE工具IntelliJ。 

以下是相关学习资源的列表。本页中的段落大致按学习的顺序排列。 

## 免费的机器学习和深度学习网络课程

* [Coursera上的机器学习课程，Andrew Ng主讲](https://www.coursera.org/learn/machine-learning/home/info) 
* [Coursera上的神经网络课程，Geoff Hinton主讲](http://class.coursera.org/neuralnets-2012-001/lecture) 
* [MIT人工智能导论，Patrick Winston主讲](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)（供希望了解人工智能概况的用户参考。）
* [斯坦福卷积神经网络课程，Andrej Karpathy主讲](http://cs231n.github.io)（供希望了解图像识别的用户参考。）

## 数学

* [Andrew Ng的六节线性代数复习课](https://www.youtube.com/playlist?list=PLnnr1O8OWc6boN4WHeuisJWmeQHH9D_Vg)
* [可汗学院的线性代数课程](https://www.khanacademy.org/math/linear-algebra)
* [机器学习中的线性代数](https://www.youtube.com/watch?v=ZumgfOei0Ak)；Patrick van der Smagt主讲
* [CMU的线性代数回顾](http://www.cs.cmu.edu/~zkolter/course/linalg/outline.html)
* [《机器学习中的数学》](https://www.umiacs.umd.edu/~hal/courses/2013S_ML/math4ml.pdf)
* [《沉浸式线性代数教程》](http://immersivemath.com/ila/learnmore.html)
* [《概率论速查手册》](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf)
* [线性代数读物精选](https://begriffs.com/posts/2016-07-24-best-linear-algebra-books.html)
* [马尔可夫链解析](http://setosa.io/ev/markov-chains/)
* [《机器学习中的马尔可夫链蒙特卡洛算法导论》](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7133&rep=rep1&type=pdf)

## 编程

如果您还未掌握如何编程，建议您不要从Java语言开始学习。Python和Ruby的反馈速度快，学习这两种语言更容易掌握编程的基本理念。 

* [《笨办法学Python》](http://learnpythonthehardway.org/)
* [《Learn to Program (Ruby)》](https://pine.fm/LearnToProgram/)
* [命令行入门教程](http://cli.learncodethehardway.org/book/)
* [命令行补充教程](http://www.learnenough.com/command-line)
* [Vim教程与基础知识](https://danielmiessler.com/study/vim/)（Vim是一种基于命令行的编辑器。）
* [计算机科学导论（哈佛edX课程CS50）](https://www.edx.org/course/introduction-computer-science-harvardx-cs50x)
* [《计算机基本原理浅析》](https://marijnhaverbeke.nl/turtle/)

如果您希望跳过Java语言，直接开始使用深度学习，我们推荐[Theano](http://deeplearning.net/)和建立在其基础上的各类Python框架，包括[Keras](https://github.com/fchollet/keras)和[Lasagne](https://github.com/Lasagne/Lasagne)。

## Java

掌握了编程的基础知识后，下一个需要攻克的目标是Java语言——世界上使用最广泛的编程语言，Hadoop也是用Java编写开发的。 

* [Java资源](http://wiht.link/java-resources)
* [Java Ranch：Java语言初学者社区](http://javaranch.com/)
* [普林斯顿Java语言编程导论](http://introcs.cs.princeton.edu/java/home/)
* [《Head First Java》](http://www.amazon.com/gp/product/0596009208)
* [《Java技术手册（Java in a Nutshell）》](http://www.amazon.com/gp/product/1449370829)

## Deeplearning4j

具备上述知识后，我们建议您通过[示例](https://github.com/deeplearning4j/dl4j-examples)来学习Deeplearning4j。 

* [快速入门指南](./quickstart.html)

完成指南中的设置步骤并理解这一API之后，您就可以进行完全安装了。

* [完全安装指南](./gettingstarted)

## 其他资源

目前有关于深度学习的知识大都发表在学术论文中。

[此页](./deeplearningpapers)中列出了一些相关论文。 

任何具体课程所教授的内容都是有限的，但互联网可以提供的知识是无限的。大部分数学和编程问题都可以通过谷歌搜索或搜索[Stackoverflow](http://stackoverflow.com)和[Math Stackexchange](https://math.stackexchange.com/)等网站来解决。

## DL4J入门指南

* [深度神经网络简介](./neuralnet-overview)
* [回归分析与神经网络](./linear-regression)
* [Word2vec：用神经词向量从原始文本中提取关系](./word2vec)
* [受限玻尔兹曼机：构建深度置信网络的基本单元](./restrictedboltzmannmachine)
* [循环网络与长短期记忆单元](./lstm)
* [用于图像处理的卷积网络](./convolutionalnets)
* [人工智能、机器学习与深度学习](./ai-machinelearning-deeplearning)
* [开源深度学习框架比较](./compare-dl4j-torch7-pylearn)
* [本征向量、协方差、PCA和熵](/eigenvector)

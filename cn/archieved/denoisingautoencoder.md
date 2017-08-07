---
title: 降噪自动编码器
layout: cn-default
---

# 降噪自动编码器

自动编码器是一种用于降维的神经网络，这也就是说，它可以用于特征选择和提取。隐藏层数量多于输入的自动编码器有可能会学习[恒等函数](https://en.wikipedia.org/wiki/Identity_function)（输出直接等于输入的函数），进而变得无用。 

降噪自动编码器是基本自动编码器的一种扩展，是加入了随机因素的自动编码器。降噪自动编码器采用对输入进行随机污染（即引入噪声）的方式来减少学习恒等函数的风险，自动编码器必须将污染后的输入重构，或称降噪。 

### 参数和污染率 

对输入加入的噪声以百分比形式计量。一般而言，污染率在30%或0.3是比较合适的，但如果数据量非常少，就有可能要增加噪声量。

### 输入/初始化降噪自动编码器

单线程降噪自动编码器很容易设置。 

要创建自动编码器，只需将一个AutoEncoder类实例化并设定corruptionLevel，即噪声，如下面的例子所示。

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/autoencoder/StackedAutoEncoderMnistExample.java?slice=24:96"></script>

以上就是使用MNIST数据设置有一个可见层和一个隐藏层的降噪自动编码器的方法。该网络的学习速率为0.1，动量为0.9，使用重构叉熵作为损失函数。 

接下来我们将向您介绍[堆叠式降噪自动编码器](./stackeddenoisingautoencoder.html)，也就是许多串在一起的降噪自动编码器。

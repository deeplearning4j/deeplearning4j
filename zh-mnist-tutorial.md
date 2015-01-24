---
title: 
layout: zh-default
---

# 深信度网络的MNIST

如果要探索或研究图像识别,MNIST是一个值得参考的东西。

第一步是从数据集中取出一个图像并将它二值化,意思就是把它的像素从连续灰度转换成一和零。根据有效的经验法则,就是把所有高于35的灰度像素变成1,其余的则设置为0。MNIST数据集迭代器将会这样执行。

[MnistDataSetIterator](http://deeplearning4j.org/doc/org/datasets/iterator/impl/MnistDataSetIterator.html)可以帮您执行。

您可以这样使用DataSetIterator:

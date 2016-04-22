---
title: "深信度网络的MNIST"
layout: zh-default
---

# 深信度网络的MNIST

如果要探索或研究图像识别,MNIST是一个值得参考的东西。

第一步是从数据集中取出一个图像并将它二值化,意思就是把它的像素从连续灰度转换成一和零。根据有效的经验法则,就是把所有高于35的灰度像素变成1,其余的则设置为0。MNIST数据集迭代器将会这样执行。

[MnistDataSetIterator](http://deeplearning4j.org/doc/org/datasets/iterator/impl/MnistDataSetIterator.html)可以帮您执行。

您可以这样使用DataSetIterator:

         DataSetIterator iter = ....;
         while(iter.hasNext()) {
         	DataSet next = iter.next();
         	//do stuff with the data set
         }

一般上,DataSetIterator将处理输入和类似二值化或标准化数据设置的问题。对于MNIST ,下面的伎俩将帮您解决问题:

          //Train on batches of 10 out of 60000
          DataSetIterator mnistData = new MnistDataSetIterator(10,60000);

我们指定批量大小以及指定示例数量的原因是让用户可以选择任何一个示例数量来运行。

Windows用户需注意 ,请参考并执行以下操作方法:

1. 下载预先序列化的mnist数据集[这里](https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing):
2. 使用这数据集的迭代,这一个相当于下列之一:

               DataSet d = new DataSet();
               BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
               d.load(bis);
               bis.close();
          DataSetIterator iter = new ListDataSetIterator(d.asList(),10);
3. 接下来,我们要训练一个深度信念网络来重建MNIST数据集。这将通过以下的代码片段来完成:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierMoon.java?slice=28:95"></script>

当您的网络已被训练后,您会看到一个[F1](https://en.wikipedia.org/wiki/F1_score)分数。在机器学习里,那是一个用来评定分类器性能的指标。F1的分数是一个在零和一之间的数字,它是用来表示在训练过程中您的网络有多好。F1分数类似于一个百分比,1就是表示您的预测结果100%准确。它基本上就是您的网络猜测的正确概率。

[现在,您已经看到MNIST图像的神经网络训练,Iris接下来可以学习如何使用鸢尾花数据集来训练连续数据](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)。

---
title: 堆叠式降噪自动编码器
layout: cn-default
---

# 堆叠式降噪自动编码器

堆叠式降噪自动编码器（SDA）与降噪自动编码器的关系就像是[深度置信网络](/deepbeliefnetwork.html)与[受限玻尔兹曼机](./restrictedboltzmannmachine.html)一样。SDA的关键功能之一是随着输入的传递逐层进行无监督预定型，更广义地来看，这也是深度学习的关键功能之一。当每个层都接受了预定型，学习了如何对来自前一层的输入进行特征选择和提取之后，就可以开始第二阶段的有监督微调。 

这里要对SDA中的随机污染稍作说明：降噪自动编码器会将数据随机化，然后再通过尝试重构这些数据来进行学习。随机化的动作即是引入“噪声”，而网络的任务就是识别出噪声之中的特征，依靠这些特征来将输入的数据分类。网络在定型时会生成一个模型，用一种损失函数来衡量模型与基准之间的距离。网络会尝试将损失函数最小化，方法是重新对随机化的输入采样并再次重构数据，直至找到可以使模型与已设定的实际基准最为接近的输入。 

这种连续的重采样基于一种可以提供随机数据的生成模型，称为马尔可夫链——更具体地说，是马尔可夫蒙特卡洛算法，它会遍历整个数据集，搜寻可以构成越来越复杂的特征的代表性指标样本。

在Deeplearning4j中，构建堆叠式降噪自动编码器的方法是创建一个以自动编码器作为隐藏层的`MultiLayerNetwork`网络。这些自动编码器有`corruptionLevel`（污染率）的设定，指的就是“噪声”；神经网络会学习如何降低这种噪声信号。注意`pretrain`是设定为“真”的。

同理，构建深度置信网络的方法是创建以受限玻尔兹曼机作为隐藏层的`MultiLayerNetwork`网络。总而言之，您可以认为Deeplearning4j是用RBM和自动编码器等“原始神经网络”来构建各种深度神经网络的。

## 直接看代码


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(4)
           .layer(0, new AutoEncoder.Builder().nIn(numRows * numColumns).nOut(500)
                   .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                   .corruptionLevel(0.3)
                   .build())
                .layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)

                        .build())
                .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
                .build();

### <a name="beginner">其他Deeplearning4j教程</a>
* [神经网络简介](./neuralnet-overview)
* [LSTM和循环网络](./cn/lstm)
* [Word2vec](./word2vec)
* [受限玻尔兹曼机](./restrictedboltzmannmachine)
* [本征向量、协方差、PCA和熵](./cn/eigenvector)
* [神经网络与回归分析](./linear-regression)
* [卷积网络](./convolutionalnets)

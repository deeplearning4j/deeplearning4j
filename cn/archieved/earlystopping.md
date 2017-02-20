---
title: 早停法
layout: cn-default
---

# 早停法

为了获得性能良好的神经网络，网络定型过程中需要进行许多关于所用设置（超参数）的决策。超参数之一是定型周期（epoch）的数量：亦即应当完整遍历数据集多少次（一次为一个epoch）？如果epoch数量太少，网络有可能发生欠拟合（即对于定型数据的学习不够充分）；如果epoch数量太多，则有可能发生过拟合（即网络对定型数据中的“噪声”而非信号拟合）。

早停法旨在解决epoch数量需要手动设置的问题。它也可以被视为一种能够避免网络发生过拟合的正则化方法（与L1/L2权重衰减和丢弃法类似）。

早停法背后的原理其实不难理解：

* 将数据分为定型集和测试集
* 每个epoch结束后（或每N个epoch后）：
  * 用测试集评估网络性能
  * 如果网络性能表现优于此前最好的模型：保存当前这一epoch的网络副本
* 将测试性能最优的模型作为最终网络模型


如下图所示：

![Early Stopping](../img/earlystopping.png)

最优模型是在垂直虚线的时间点保存下来的模型，即处理测试集时准确率最高的模型。


使用DL4J的早停功能时需要指明一系列配置选项：

* 一个分值计算器，例如用于多层网络的*DataSetLossCalculator*（[JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculator.html)、 [源代码](https://github.com/deeplearning4j/deeplearning4j/blob/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculator.java)），或者用于计算图的*DataSetLossCalculatorCG*（[JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculatorCG.html)、[源代码](https://github.com/deeplearning4j/deeplearning4j/blob/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculatorCG.java)），用于在每个epoch中进行计算（例如：一个测试集的损失函数值，或者网络处理测试集的准确率）
* 分值函数的计算频率（默认为每个epoch一次）
* 一项或多项终止条件，决定何时停止定型过程。终止条件有两类：
  * epoch终止条件：每N个epoch评估一次
  * 迭代终止条件：每个微批次（minibatch）评估一次
* 一个模型保存器，定义模型的保存方式（参见：[LocalFileModelSaver的JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/LocalFileModelSaver.html)、[LocalFileModelSaver的源代码](https://github.com/deeplearning4j/deeplearning4j/blob/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/saver/LocalFileModelSaver.java)以及[InMemoryModelSaver的JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/InMemoryModelSaver.html)、[InMemoryModelSaver的源代码](https://github.com/deeplearning4j/deeplearning4j/blob/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/saver/InMemoryModelSaver.java)）

示例如下，采用一项epoch终止条件，epoch数量上限30个，定型时间上限20分钟，每个epoch计算一次，将中间结果保存至磁盘：

```

MultiLayerConfiguration myNetworkConfiguration = ...;
DataSetIterator myTrainData = ...;
DataSetIterator myTestData = ...;

EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
		.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
		.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
		.scoreCalculator(new DataSetLossCalculator(myTestData, true))
        .evaluateEveryNEpochs(1)
		.modelSaver(new LocalFileModelSaver(directory))
		.build();

EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,myNetworkConfiguration,myTrainData);

//开始早停定型：
EarlyStoppingResult result = trainer.fit();

//显示结果：
System.out.println("Termination reason: " + result.getTerminationReason());
System.out.println("Termination details: " + result.getTerminationDetails());
System.out.println("Total epochs: " + result.getTotalEpochs());
System.out.println("Best epoch number: " + result.getBestModelEpoch());
System.out.println("Score at best epoch: " + result.getBestModelScore());

//获得最优模型：
MultiLayerNetwork bestModel = result.getBestModel();

```




epoch终止条件示例：

* 如需经过一定数量（上限）的epoch后终止定型，可使用[MaxEpochsTerminationCondition](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/MaxEpochsTerminationCondition.html)
* 如需在测试分值经过M个连续epoch没有改善时终止，可使用[ScoreImprovementEpochTerminationCondition](https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/ScoreImprovementEpochTerminationCondition.html)

迭代终止条件示例：

* 如需在达到一定的时间上限时停止定型（不等待当前epoch完成），可使用[MaxTimeIterationTerminationCondition](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/termination/MaxTimeIterationTerminationCondition.java)
* 不论何时，只要测试分值超过一定数值便停止定型，可使用[MaxScoreIterationTerminationCondition](https://github.com/deeplearning4j/deeplearning4j/blob/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/termination/MaxScoreIterationTerminationCondition.java)。比如可以用于在网络调试效果不佳或定型状况不稳定（例如权重/分值膨胀）时立即终止定型。

内置终止类的源代码参见[此目录](https://github.com/deeplearning4j/deeplearning4j/tree/c152293ef8d1094c281f5375ded61ff5f8eb6587/deeplearning4j-core/src/main/java/org/deeplearning4j/earlystopping/termination)

您当然也可以实现自己的迭代和epoch终止条件。


最后提一下：

* 这个[非常简单的早停法示例](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/earlystopping/EarlyStoppingMNIST.java)可供参考
* [这些单元测试](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/earlystopping/TestEarlyStopping.java)可能也会帮助
* 在Spark上也可以进行早停定型。网络配置方法相同；但使用的不是上文中的EarlyStoppingTrainer，而是[SparkEarlyStoppingTrainer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark/src/main/java/org/deeplearning4j/spark/earlystopping/SparkEarlyStoppingTrainer.java)
  * [这些单元测试](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark/src/test/java/org/deeplearning4j/spark/TestEarlyStoppingSpark.java)可以用于基于Spark的早停定型

---
title: 神经网络的保存和加载
layout: cn-default
---

# 神经网络的保存和加载
ModelSerializer是用于加载和保存模型的类。保存的方法有两种（见以下示例）：
第一个示例保存的是普通的多层网络，第二个示例保存的是一个[计算图](https://deeplearning4j.org/compgraph)。

下面这个[基本示例](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving)给出了用ModelSerializer类保存计算图所需的代码，另一个示例说明了如何用ModelSerializer保存用MultiLayer Configuration构建的神经网络。  

## RNG种子

如果您的模型用到概率（即DropOut/DropConnect），那么最好能将其分开保存，待模型还原后再加以应用。即：

```bash
 Nd4j.getRandom().setSeed(12345);
 ModelSerializer.restoreMultiLayerNetwork(modelFile);
```

这将确保不同会话/JVM能产生一致的结果。

<!---
Verify up to date before re-including
以下是加载多层网络或计算图的示例：
[参见测试示例](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/util/ModelSerializerTest.java)
-->

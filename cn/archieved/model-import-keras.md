---
title: 将Keras模型导入Deeplearning4j
layout: cn-default
---

# 将Keras模型导入Deeplearning4j

*请注意：模型导入是新推出的功能，截至2017年2月，我们建议用户在提出问题或报告bug之前先尝试使用最新的版本，或者用源码进行本地构建。* 

`deeplearning4j-modelimport`模块中包含的例程让用户可以导入用[Keras](https://keras.io/)配置、定型的神经网络模型。Keras是主流的Python深度学习库之一，可以在Deeplearning4j、[Theano](http://deeplearning.net/software/theano/)和[TensorFlow](https://www.tensorflow.org)后端上建立抽象层。Keras模型的保存方法参见Keras的[FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。Deeplearning4j将Keras作为其Python API，详情参见[此处](https://github.com/crockpotveggies/dl4j-examples/tree/keras-examples/dl4j-keras-examples)。

![Model Import Schema](../img/model-import-keras.png)

`IncompatibleKerasConfigurationException`消息表明Deeplearning4j目前还不支持您尝试导入的Keras模型配置（可能是因为导入模块尚不支持该模型配置，或者DL4J中尚未实现这种模型、层或功能）。

导入模型之后，建议您采用我们的modelserializer类来进行模型的保存和加载。 

如有更多问题，请访问[DL4J线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)。您也可以考虑[通过Github提交功能申请](https://github.com/deeplearning4j/deeplearning4j/issues)，让所需的功能进入DL4J的开发路线图，甚至可以自己编写功能分支，然后向我们发出合并请求！

模型导入模块和本页教程都将不断更新，敬请关注！

## 常用模型的支持

VGG16和其他预定型模型被广泛用于演示以及针对特定用例的再定型。我们很高兴宣布DL4J目前已能支持VGG16导入，同时还提供一些辅助功能，包括将所需摄取的数据格式化和标准化，以及将数值输出转换为标签文本类别等。  

## Deeplearning4j模型库

除了可导入预定型的Keras模型，Deeplearning4j还有自己的模型库，我们会不断添加新的模型。 

## 使用modelimport类之前所需的IDE配置

编辑您的Pom.xml文件，加入下列依赖项

```
<dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-modelimport</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
```

## 可用方法

Keras模型导入功能有以下几种使用方式。Keras有两类模型：Sequential（顺序）和Functional（函数式，又译泛型）模型。Keras的Sequential模型相当于DeepLeanring4J的MultiLayerNetwork。Keras的Functional模型相当于DeepLearning4J的计算图。  

## 仅导入模型配置

请注意，目前我们还不能支持全部的模型配置，但我们的目标是让用户能导入最有用且最常用的网络类型。

若要使用本项功能，您需要将Keras模型保存至一个JSON文件，Deeplearning4j支持的选项包括： 

* Sequential模型 
* 包含用于后续定型的更新器的Sequential模型
* Functional模型
* 包含用于后续定型的更新器的Functional模型

### 看代码

* 导入在Keras中用model.to_json()保存的Sequential模型配置

```
MultiLayerNetworkConfiguration modelConfig = KerasModelImport.importKerasSequentialConfiguration("JSON文件的路径)

```

* 导入在Keras中用model.to_json()保存的ComputationGraph模型配置

```
ComputationGraphConfiguration computationGraphConfig = KerasModelImport.importKerasModelConfiguration("JSON文件的路径)

```






## 导入已在Keras中定型的模型的配置和权重

您需要先保存已定型的Keras模型的JSON配置文件以及权重。权重保存在一个H5格式的文件中。您可以在Keras中将权重和模型配置保存到同一个H5文件中，或者可以将模型配置单独保存到另一个文件中。 

### 看代码

* 单文件Sequential模型

```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("H5文件的路径")

```

导入的网络可直接用于推断，只需按原始定型数据的处理方法对新的数据进行格式化、转换和标准化，然后将数据输入网络，再调用network.output即可。

* 配置与权重分别保存在不同文件中的Sequential模型 


```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("JSON文件的路径","H5文件的路径")

```

## 其他选项

modelimport功能包括一项称为enforceTrainingConfig的参数。 

如果您导入预定型的模型**仅仅**是为了用于推断，那么就应当设置enforceTrainingConfig=false。目前尚不支持的仅用于定型的模型配置会触发警告消息，但模型导入功能会继续运行。

如果您导入模型是为了定型，并且希望所得的模型与已定型的Keras模型尽可能接近，那么就应当设置enforceTrainingConfig=true。如果这样设置，那么系统遇到目前尚不支持的仅用于定型的模型配置时会抛出UnsupportedKerasConfigurationException异常并停止导入。



## 导入Keras模型

以下的[视频教程](https://www.youtube.com/embed/bI1aR1Tj2DM)演示了将Keras模型加载到Deeplearning4j并验证网络能否正常运行的代码。主讲Tom Hanlon会介绍如何将一个在Keras中用Theano后端构建的简单的鸢尾花数据分类器导出并加载到Deeplearning4j当中。

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

如果视频无法正常播放，请点击[此处](https://www.youtube.com/embed/bI1aR1Tj2DM)前往YouTube观看。

## 为什么选择Keras？

Keras是在Theano或Tensorflow等Python学习库的基础上建立的一个抽象层，是一种更加便利的深度学习接口。 

如果要在Theano这样的框架中定义一个层，您必须非常精确地定义权重、偏差、激活函数以及将输入数据转换为输出的具体方式。 
此外，您还需要自行处理反向传播以及权重和偏差的更新。Keras将这些都包装起来，为用户提供包含上述各类计算和更新机制的预制层。

使用Keras时，您只需要定义输入输出的形态和误差的计算方式。Keras会确保所有的层大小合适且误差能正常地反向传播，甚至还可以完成分批操作。

更多信息参见[此处](http://deeplearning4j.org/cn/keras)。





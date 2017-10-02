---
title: Deeplearning4j模型库
layout: cn-default
---

# Deeplearning4j模型库

0.9.0版（0.8.1-SNAPSHOT）的Deeplearning4j配有一个全新的原生模型库，可以直接从DL4J中访问和实例化。从Github上复制模型配置的日子已经一去不复返。模型库还提供用自动下载的不同数据集进行预训练后获得的权重，预训练数据集都已经过完整性检查。 

使用新模型库之前需要先将其添加为依赖项。请在Maven POM文件中加入以下代码：

```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-zoo</artifactId>
    <version>${deeplearning4j.version}</version>
</dependency>
```

`${deeplearning4j.version}`应当与您所使用的DL4J版本保持一致，请采用[Maven中央仓库](http://mvnrepository.com/artifact/org.deeplearning4j)提供的最新版本。



## 入门指南

将模型库依赖项添加至项目后，您就可以开始导入模型并加以使用了。每个模型都是`ZooModel`抽象类的扩展，采用`InstantiableModel`接口。这些类提供的方法可以帮助您初始化一个空白的全新网络，或者一个已完成预训练的网络。

### 新神经网络的初始化

您可以用`.init()`方法将模型库中的某一个模型实例化。例如，您可以用以下代码来实例化一个全新的、未经训练的AlexNet网络：

```
import org.deeplearning4j.zoo.model.AlexNet
import org.deeplearning4j.zoo.*;

...

int numberOfClassesInYourData = 1000;
int randomSeed = 123;
int iterations = 1; // 绝大多数情况都是1

ZooModel zooModel = new AlexNet(numberOfClassesInYourData, randomSeed, iterations);
Model net = zooModel.init();
```

如果您希望调试参数或更改优化算法，可引用网络的基础配置：

```
ZooModel zooModel = new AlexNet(numberOfClassesInYourData, randomSeed, iterations);
MultiLayerConfiguration net = zooModel.conf();
```

### 初始化预训练的权重

部分模型有预训练的权重可供使用，有一小部分模型已接受过多个数据集的预训练。枚举器`PretrainedType`界定不同的权重类型，包括`IMAGENET`、`MNIST`、`CIFAR10`和`VGGFACE`。

例如，下列代码可以让您用ImageNet权重来初始化一个VGG-16模型：

```
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.*;

...

ZooModel zooModel = new VGG16();
Model net = zooModel.initPretrained(PretrainedType.IMAGENET);
```

然后再用通过VGG-Face数据集训练所得的权重来初始化另一个VGG-16模型：

```
ZooModel zooModel = new VGG16();
Model net = zooModel.initPretrained(PretrainedType.VGGFACE);
```

如果您不确定一个模型是否包含预训练权重，可以使用返回一个布尔值的`.pretrainedAvailable()`方法。您只需将一个`PretrainedType`枚举类传递给该方法，有预训练的权重可用时，返回结果为真。

请注意，卷积网络模型的输入形状信息遵循NCHW格式。因此，假如一个模型的默认输入形状为`new int[]{3, 224, 224}`，那么表明该模型有3个通道，高/宽为224。



## 模型库里有什么？

模型库中提供深度学习社区所熟知的一系列网络配置，VGG-16、ResNet-50、AlexNet、Inception-ResNet-v1、GoogLeNet、LeNet等ImageNet模型均包括在内。此外，模型库中还包括一个用于文本生成的LSTM网络，以及一个进行通用图像识别的简单卷积神经网络。

您可以通过[deeplearning4j-zoo Github链接](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model)查看完整的模型列表。



## 高级功能

模型库还有一些额外的功能，可帮助您将模型应用于不同的情境。

### 模型选择器

`ModelSelector`让您可以一次选择多个模型。该方法的创建目的是为了同时测试多个模型。

选择器返回的结果是一个`Map<ZooType, ZooModel>`集合。把`ZooType.RESNET50`传递给`ModelSelector.select(ZooType.RESNET50)`后，系统将返回`HashMap<ZooType.RESNET50, new ResNet50()>`类型的集合。但是，如果您想同时为一个特定的数据集初始化多个模型，可以用`ModelSelector.select(ZooType.RESNET50, numLabels, seed, iterations)`的方式向选择器传递恰当的参数。

假设您想要对所有提供ImageNet预训练权重的模型进行基准测试。下列代码可以让您选择所有的卷积模型，检查是否有权重可用，然后执行您自己的代码。

```
import org.deeplearning4j.zoo.*;
...

// 选择所有卷积模型
Map<ZooType, ZooModel> models = ModelSelector.select(ZooType.CNN);

for (Map.Entry<ZooType, ZooModel> entry : models.entrySet()) {
    ZooModel zooModel = entry.getValue();

    if(model.pretrainedAvailable(PretrainedType.IMAGENET)) {
        Model net = zooModel.initPretrained(PretrainedType.IMAGENET);

        // 对预训练模型进行所需的操作
    }

    // 为当前模型进行清理（有助于避免内存不足）
    Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    System.gc();
    Thread.sleep(1000);
}
```

或者，您可以用以下代码选择具体的模型：

```
ModelSelector.select(ZooType.RESNET50, ZooType.VGG16, ZooType,VGG19);
```

### 改变输入

除了将特定配置信息传递给一个模型的构造器之外，您还可以用`.setInputShape()`来改变模型的输入形状。请注意：该方法只适用于全新配置，无法影响已完成预训练的模型。

```
ZooModel zooModel = new ResNet50(10, 123, 1);
zooModel.setInputShape(new int[]{3,28,28});
```

### 迁移学习

预训练模型非常适合迁移学习！您可以在[此页](https://deeplearning4j.org/transfer-learning)进一步了解如何使用DL4J进行迁移学习。

### 工作区

初始化方法通常有一项名为`workspaceMode`的额外参数。大多数用户不需要使用这项参数；但假如您有一台配置超强的大型计算机，那么可以将该参数设为`WorkspaceMode.SINGLE`。如需进一步了解有关工作区的信息，请参阅[此页](https://deeplearning4j.org/cn/workspaces)。

---
title: Deeplearning4j开发者指南
layout: cn-default
---

# 开发者指南

*注：希望为项目做贡献的开发者需要用源码构建系统。请参照[本地构建指南](./buildinglocally.html)进行安装。*

## DeepLearning4J及其相关项目概述

DeepLearning4j本身可能是曝光度最高的项目，您也可以考虑为其他的相关项目做贡献。我们的项目包括：

* [DeepLearning4J](https://github.com/deeplearning4j/deeplearning4j)：包含神经网络所需的全部代码，包括单机学习和分布式学习的代码。
* [ND4J](https://github.com/deeplearning4j/nd4j)：“为Java编写的N维数组”。ND4J是DL4J的数学运算后端和构建基础。DL4J神经网络全部都是用ND4J中的各类运算（矩阵乘法、向量运算等）建立的。也正是因为ND4J，DL4J才能同时支持用CPU和GPU训练网络，而无需对网络自身进行任何修改。没有ND4J，就没有DL4J。
* [DataVec](https://github.com/deeplearning4j/Datavec)：DataVec是数据加工管道的一部分，负责处理数据导入和转换。如果需要导入图像、视频、音频或CSV数据至DL4J，您就很有可能用到DataVec。
* [Arbiter](https://github.com/deeplearning4j/Arbiter)：Arbiter是神经网络的超参数优化包（还具备其他功能）。超参数优化指通过自动选择神经网络超参数（学习速率、层的数量等）来改善性能表现的过程。
* [DL4J示例](https://github.com/deeplearning4j/dl4j-examples)

Deeplearning4j和ND4J采用Apache 2.0许可协议发行。

## 贡献方式

为DeepLearning4J（及其相关项目）做贡献的方式有很多，您可以按个人兴趣和经验来选择。以下建议供您参考：

* 添加新型神经网络层（例如不同类型的RNN、本地连接网络等）
* 添加新的训练功能
* 修正错误
* DL4J示例：哪些应用或网络架构还没有示例？
* 测试性能，发现瓶颈或改善领域
* 改进网站文档（或者撰写教程等）
* 改进JavaDoc

有几种不同的方法来寻找工作内容。具体包括：

* 查看问题跟踪页：
  * [https://github.com/deeplearning4j/deeplearning4j/issues](https://github.com/deeplearning4j/deeplearning4j/issues)
  * [https://github.com/deeplearning4j/nd4j/issues](https://github.com/deeplearning4j/nd4j/issues)
  * [https://github.com/deeplearning4j/DataVec/issues](https://github.com/deeplearning4j/Canova/issues)
  * [https://github.com/deeplearning4j/dl4j-examples/issues](https://github.com/deeplearning4j/dl4j-examples/issues)
* 查阅我们的[开发路线图](http://deeplearning4j.org/roadmap.html)
* 在[Gitter](https://gitter.im/deeplearning4j/deeplearning4j)上与开发者交流，尤其可以利用我们的[早期用户交流群](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters)
* 查阅关于训练功能、网络架构和应用的最新论文与博客文章
* 浏览[网站](http://deeplearning4j.org/documentation.html)和[示例](https://github.com/deeplearning4j/dl4j-examples/)－还缺少哪些内容？有哪些内容还不够完整？还可以添加什么有用的（或者酷炫的）内容？

## DL4J/ND4J及其他项目的开发－基本信息

投入工作之前，您还需要了解一些注意事项。特别是我们使用的工具：

* Maven：依赖项管理和构建工具，用于DL4J的所有项目。Maven的详情参见[此页](http://deeplearning4j.org/maven.html)。
* Git：我们使用的版本控制系统
* [Project Lombok](https://projectlombok.org/)：Project Lombok是一个代码生成/批注工具，用于减少Java中需要的“样板”代码（即标准化的重复代码）。在使用源代码工作时，您需要为IDE安装[Project Lombok插件](https://projectlombok.org/download.html)
* [Travis](https://travis-ci.org/)：Travis是一项持续集成服务，用于实现自动化测试。向代码库发起的任何合并请求都会由Travis自动测试，Travis会尝试构建您的代码，然后运行项目中的*所有*单元测试。这有助于自动发现合并请求中的问题。

还有一些工具可能也会对您有所帮助：

* [VisualVM](https://visualvm.java.net/)：性能分析工具，主要用于发现性能问题和瓶颈。
* [IntelliJ IDEA](https://www.jetbrains.com/idea/)：这是我们选用的IDE，当然您也可以选择Eclipse和NetBeans等其他工具。与开发者选用同样的IDE有助于您解决可能遇到的问题，但最终还是由您决定。

向DL4J及其相关项目做贡献的基本工作流程如下：

1. 选择具体工作内容。如果有相关的[待解决问题](https://github.com/deeplearning4j/deeplearning4j/issues)：您可以要求将该问题分配给自己。这有利于协调工作，避免重复劳动。如果还没有待解决的问题：可以考虑开设一个问题，具体内容可以是某些缺陷或新的功能。
2. 如果您还没有建立过DL4J（或ND4J/Canova/任何其他项目）的派生项目，您可以在该项目的主代码库页面上进行派生操作
3. 每天的第一件事是将本地的（所有）代码库副本同步。
  * 详情参见[派生项目同步](https://help.github.com/articles/syncing-a-fork/)
  * （注：如果您有deeplearning4j代码库的编辑权（大部分人都没有），请对ND4J、DataVec和DL4J进行“git pull”和“mvn clean install”
4. 在您的本地副本中修改代码。添加测试，确保新功能可以正常运行，需要修正的错误确实已经消除。
5. 创建[合并请求](https://help.github.com/articles/using-pull-requests/)
  * 填写说明性的标题和描述。请注明所有与合并请求相关的问题。
6. 等待Travis运行。
  * 如果Travis运行失败：检查运行记录，发现原因。是否有某一项测试运行失败？如果有，请在本地检查这项测试，进行必要的修改。任何推送至您的本地代码库的更改都会自动添加至合并请求。
7. Travis运行成功之后有两种可能的情况：
  * 您的合并请求通过审核，予以合并。太棒了！
  * 或者，我们会在审核之后请您做一些最终的更改，再予以合并。通常需要的都只是细微的调整。


以下是一些需要记住的指导原则和注意事项：

* 代码应当符合Java 7的规范
* 如果您要添加一项新的方法或类：请添加JavaDoc
  * 例如：每项实际参数及其假设状态（可以为空值？始终为正？）分别是什么？
  * 我们欢迎您为重大新增功能添加作者标记。这对于未来的贡献者也有帮助，因为他们可能需要对原始作者提问。假如一个类有多位作者：请详细说明每位作者所做的工作（“原始实现”、“添加了XX功能”等）
  * JavaDoc可包含大量细节，可使用各类格式选项（代码、粗体/斜体文本、链接等）：更多详情参见 [此页](http://docs.oracle.com/javase/7/docs/technotes/tools/windows/javadoc.html)
* 请在您的代码中加入说明性批注，这可以方便所有代码的长期维护。
* 任何新功能都应当包括单元测试（采用[JUnit](http://junit.org/)），来测试您的代码。测试应考虑到极端情况。
* 若要添加一种新的层，您必须添加数值梯度检验，方法参见[这些单元测试](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/GradientCheckTests.java)。这是确保计算所得梯度的正确性的必要检查。
* 若要添加一种重要的新功能，请考虑更新网站中的相关内容并提供一个示例。毕竟，没人知道（或者没人知道怎么使用）的功能是没有意义的。我们鼓励您在情况合适时添加文档，但严格来讲文档不是必需的。
* 如果您有任何疑惑，请随时提问！



## 贡献文档：网站更新

您想要为[deeplearning4j.org](http://deeplearning4j.org/)和[nd4j.org](http://nd4j.org/)等网站做贡献？太好了。我们的网站也是开源的，欢迎您提交合并请求。

DL4J的网站是如何建立起来的呢？其实很简单。以deeplearning4j.org为例：

* 网站本身由GitHub托管，具体位于[gh-pages](https://github.com/deeplearning4j/deeplearning4j/tree/gh-pages)分支下
* 网站代码用[Markdown](https://help.github.com/articles/markdown-basics/)格式/语言编写，自动转换为HTML网页。例如，您现在正在阅读的整个页面都是用Markdown编写的。
  * 有个别网页是例外，例如documentation.html页面采用的是HTML格式。

Markdown语言本身相对比较容易掌握。加入您还不熟悉这种语言，请查看[gh-pages](https://github.com/deeplearning4j/deeplearning4j/tree/gh-pages)分支下的现有页面（.md文件），将其作为入门指导。



## DeepLearning4J：几点注意事项

DL4J是一个很大很复杂的软件。完整地概述DL4J的工作模式是很困难的，本段将努力提供比较全面的简介。

我们从主要的软件包开始：

* deeplearning4j-core：包含所有的层、配置和优化代码。
* deeplearning4j-scaleout：分布式学习（Spark）外加Word2Vec等其他模型
* deeplearning4j-ui：用户界面功能，例如[HistogramIterationListener](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/HistogramIterationListener.java) （柱状图迭代侦听器，[另见此页](http://deeplearning4j.org/visualization.html)）等。DL4J的用户界面功能基于[Dropwizard](http://www.dropwizard.io/)、[FreeMarker](http://freemarker.incubator.apache.org/)和[D3](http://d3js.org/)。简言之，这些组件让UI Javascript代码可以在网络训练过程中使用DL4J的输出结果。




### 为DL4J添加新的层

假设您想要为DL4J添加一种全新的层。以下是一些需要了解的事项。

首先，网络配置和网络实现（即数学运算）是相互分离的。虽然有些容易混淆，但两者都称为层：

* [org.deeplearning4j.nn.api.Layer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Layer.java)用于网络实现
* [org.deeplearning4j.nn.conf.layers.Layer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/layers/Layer.java)用于网络配置

如果要实现一种新的层，您需要实现以下所有项目：

* 一个层的配置类和一个构建器（Builder）类。您可以参考[这些类](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/layers)的设计
* 一个层的实现类。同样，您可以参考[这些类](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers)的设计
* 一个针对您的层的[ParameterInitializer](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params)（参数初始化器，负责按网络配置设定初始参数）
* 一个[LayerFactory](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/factory)（层工厂），对DefaultLayerFactory进行扩展并将您的层添加至DefaultLayerFactory.getInstance()

DL4J目前尚无符号自动微分。这意味着正向传递（预测）和反向传递（反向传播）的代码必须手动实现。

其他一些注意事项：

* DL4J有一项[数值梯度检查工具](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java)，使用[这些单元测试](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck)。
  * 数值梯度检验的目的是确保所有分析梯度（您的层中计算所得）与数值梯度相近。更多信息请参见[这一JavaDoc](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java)
  * 任何新型的层都必须进行梯度检验
* 参数和梯度（见下段说明）会被压缩为一个单行向量。很重要的一点是，参数和梯度的压缩顺序必须相同。在实践中，这通常是指您向[Gradient对象](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/gradient)添加梯度的顺序应当与层参数被压缩为单行向量（即[Model.params()](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Model.java)）的顺序相同。未能做到这一点是梯度检验失败的常见原因之一。

### 反向传播在DL4J中的实现方式

来看反向传播。让我们从基本信息开始－首先概述您需要了解的类：

* MultiLayerNetwork：一个具有多个层的神经网络
* Layer：单个层
* Updater：更新器，例如AdaGrad、动量或RMSProp更新器。
* Optimizer：优化器，这一抽象层让DL4J能够支持随机梯度下降、共轭梯度、L-BFGS等，与其他选项（或更新器）结合使用

接下来我们依次介绍您调用MultiLayerNetwork.fit(DataSet)或MultiLayerNet.fit(DataSetIterator)之后发生的每个步骤。我们假定网络在进行反向传播（而不是无监督的预训练）。

1. 设定MultiLayerNetwork的输入和输出（均为INDArray）
2. 如果[Solver](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/Solver.java)对象不存在，则创建该对象
3. 调用Solver.optimize()。这会调用[ConvexOptimizer.optimize()](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/api/ConvexOptimizer.java)。ConvexOptimizer（凸优化器）是什么呢？我们用这一抽象层来实现多种优化算法，包括[StochasticGradientDescent](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/StochasticGradie)（随机梯度下降）、[LineGradientDescent](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/LineGradientDesc)（线搜索梯度下降）、[ConjugateGradient](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/ConjugateGradient.java)（共轭梯度）和[L-BFGS](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/LBFGS.java)。
  请注意，上述每个ConvexOptimizer类都是[BaseOptimizer](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/optimize/solvers/BaseOptimizer.java)的扩展。下一步我们假设现在使用的是StochasticGradientDescent。
4. StochasticGradientDescent.optimize()：这一步发生两件事：首先：系统调用BaseOptimizer.GradientAndScore()，启动梯度计算。其次：系统对参数进行更新。
5. BaseOptimizer.gradientAndScore()：
  * 调用MultiLayerNetwork.computeGradientAndScore()－计算梯度，然后：
  * 调用BaseOptimizer.updateGradientAccordingToParams()－应用学习速率、adagrad等
6. 回到StochasticGradientDescent：更新值（即调整后的梯度）和网络参数均被压缩为一维单行向量。随后添加梯度和参数，完成网络参数设定
7. 完成！至此网络参数已被更新，我们再回过来细看流程


刚才其实略过了两项重要的组成部分：梯度的计算和梯度的更新/调整。


**梯度计算**
从MultiLayerNetwork.computeGradientAndScore()继续：

* MultiLayerNetwork首先让网络进行一次完整的正向传递，采用先前设定的输入
  * 最终系统将对网络从输入到输出之间的每一层调用[Layer.activate(INDArray,boolean)](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/api/Layer.java#L200-200)方法。
  * 在每一层中，输入的激活值会被保存起来。反向传播时需要这些激活值。
* 随后，MultiLayerNetwork开始对网络进行反向传播，从OutputLayer（输出层）倒回至输入层。
  * 调用MultiLayerNetwork.calcBackpropGradients(INDArray,boolean)
  * 梯度计算从OutputLayer开始，输出层依据网络预测/输出、标签和在配置中设定的损失函数来计算梯度，[见此处](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/BaseOutputLayer.java)
  * 随后依次用上一层的误差来计算每个层的梯度。
  * 最终设定MultiLayerNetwork.gradient字段，实际上是一项包含每个层的梯度的```Map<String,INDArray>```，之后会被优化器检索提取。


**更新梯度**
更新梯度需要将每项参数的梯度变为更新值。“更新值”就是梯度在应用学习速率、动量、L1/L2正则化、梯度修剪、除以微批次大小等操作后的值。

这一功能通过[BaseUpdater](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/BaseUpdater.java)以及[各种更新器类](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater)来实现。

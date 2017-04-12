---
title: MultiLayerConfiguration类
layout: cn-default
---

# MultiLayerConfiguration类：
*DL4J多层神经网络构建器基础*

MultiLayerConfiguration（多层网络配置）类是在Deeplearning4j中创建深度学习网络的基础构建器。以下是多层网络配置的参数及其默认设置。

多层网络可接受与单层网络同样类型的输入，多层网络的参数通常也与对应的单层网络相同。

开始构建多层网络Java类的方法：

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

您可以用以下方式为这个类追加参数：

    new NeuralNetConfiguration.Builder().iterations(100).layer(new RBM())
        .nIn(784).nOut(10).list(4).hiddenLayerSizes(new int[]{500, 250, 200})
        .override(new ClassifierOverride(3))
        .build();

参数：

虽然以下均为多层网络特有的参数，但是单层神经网络的输入同样适用于多层网络。

- **hiddenLayerSizes**：*int[]*，前馈层的节点数
   - 两层格式 = new int[]{50} = int数组初始设置为50个节点
   - 五层格式 = new int[]{32,20,40,32} = 第1层为32个节点，第2层为20个节点，等等
- **list**：*int*，层的数量；该函数会将您的配置复制n次，建立分层的网络结构。
    - 层数中不应包含输入
- **override**：(*int*, *class*) {层数, 数据处理器类}，修改特定层的配置
    - 在构建多层网络时，每个层的配置未必完全相同。
    - 在这种情况下，可以用override方法来对特定层的配置进行必要的调整。
    - 您可以用override()来指定需要修改的层以及用于修改配置的构建器。
    - 将深度网络的第一层设置为卷积层的示例：

            .override(0, new ConfOverride() {
                        public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                            builder.layer(new ConvolutionLayer());
                            builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                            builder.featureMapSize(2, 2);
                        }
                    })

    - 将神经网络第二层/最后一层配置修改为分类器的示例：

            .override(new ClassifierOverride(1))

- **useDropConnect**：*boolean*，由丢弃法（dropout）推广而来的正则化方法；对神经网络的权重进行随机置零。
    - 默认 = false

该类的更多详情参见[JavaDoc](http://deeplearning4j.org/doc/)。

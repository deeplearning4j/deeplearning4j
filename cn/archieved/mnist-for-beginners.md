---
title: MNIST基础教程
layout: cn-default
---

<header>
  <h1>MNIST基础教程</h1>
  <p>本教程将介绍如何对MNIST数据集进行分类，这在机器学习领域是相当于“Hello World”的入门示例。</p>
  <ol class="toc">
    <li><a href="#introduction">序言</a></li>
    <li><a href="#mnist-dataset">MNIST数据集</a></li>
    <li><a href="#configuring">配置MNIST示例</a></li>
    <li><a href="#building">搭建神经网络</a></li>
    <li><a href="#training">模型定型</a></li>
    <li><a href="#evaluating">结果评估</a></li>
    <li><a href="#conclusion">总结</a></li>
  </ol>
</header>
  <p>完成整个过程预计需要30分钟时间。</p>
<section>
  <h2 id="introduction">序言</h2>
  <img src="/img/mnist_render.png"><br><br>
  <p>MNIST是一个手写数字图像的数据集，每幅图像都由一个整数标记。它主要用于机器学习算法的性能对标。深度学习算法处理MNIST的效果相当好，准确率可达到99.7%以上。</p>
  <p>我们将用MNIST来定型一个神经网络，使之能读取每幅图像并预测其中的数字。首先需要安装Deeplearning4j。</p>
  <a href="zh-quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event’, 'quickstart', 'click');">DEEPLEARNING4J快速入门指南</a>
</section>

<section>
  <h2 id="mnist-dataset">MNIST数据集</h2>
  <p>MNIST数据集包含一个有6万个样例的<b>定型集</b>和一个有1万个样例的<b>测试集</b>。定型集用于让算法学习如何准确地预测出图像的整数标签，而测试集则用于检查已定型网络的预测有多准确。</p>
  <p>这在机器学习领域中被称为<a href="https://en.wikipedia.org/wiki/Supervised_learning" target="_blank">有监督学习</a>，因为我们已经知道图像预测所应该得出的正确答案。定型集能起到监督和指导的作用，在神经网络预测错误时予以纠正。</p>
</section>

<section>
  <h2 id="configuring">配置MNIST示例</h2>
  <p>MNIST教程已经打包到Maven当中，所以无需再编写代码。请打开IntelliJ进行准备。（IntelliJ的下载方法请参见<a href="zh-quickstart">快速入门指南</a>）</p>
  <p>打开名为<code>dl4j-examples</code>的文件夹。依次进入以下目录<kbd>src</kbd> → <kbd>main</kbd> → <kbd>java</kbd> → <kbd>feedforward</kbd> → <kbd>mnist</kbd>，打开名为<code>MLPMnistSingleLayerExample.java</code>的文件。</p>
  <p><img src="/img/mlp_mnist_single_layer_example_setup.png"></p>
  <p>我们将在这个文件中配置神经网络，定型模型，评估结果。建议您结合其中的代码来学习本教程。</p>
  <h3>设置变量</h3>
  <pre><code class="language-java">
    final int numRows = 28; // 矩阵的行数。
    final int numColumns = 28; // 矩阵的列数。
    int outputNum = 10; // 潜在结果（比如0到9的整数标签）的数量。
    int batchSize = 128; // 每一步抓取的样例数量。
    int rngSeed = 123; // 这个随机数生成器用一个随机种子来确保定型时使用的初始权重维持一致。下文将会说明这一点的重要性。
    int numEpochs = 15; // 一个epoch指将给定数据集全部处理一遍的周期。
  </code></pre>
  <p>在我们的示例中，每一幅MNIST图像的大小是28x28像素，这意味着输入数据是28 <b>numRows</b> x 28 <b>numColumns</b>的矩阵（矩阵是深度学习的基本数据结构）。其次，MNIST包含10种可能出现的结果（0到9的数字标签），即<b>outputNum</b>。</p>
  <p><b>batchSize</b>和<b>numEpochs</b>必须根据经验选择，而经验则需要通过实验来积累。每批次处理的数据越多，定型速度越快；epoch的数量越多，遍历数据集的次数越多，准确率就越高。</p>
  <p>但是，epoch的数量达到一定的大小之后，增益会开始减少，所以要在准确率与定型速度之间进行权衡。总的来说，需要进行实验才能发现最优的数值。本示例中设定了合理的默认值。</p>
  <h3>抓取MNIST数据</h3>
  <pre><code class="language-java">
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
  </code></pre>
  <p>我们用名为<code>DataSetIterator</code>的类来抓取MNIST中的数据，创建一个用于<b>模型定型</b>的数据集<code>mnistTrain</code>，和另一个用于在定型后<b>评估模型准确率</b>的数据集<code>mnistTest</code>。顺便一提，此处的模型是指神经网络的各种参数。这些参数是用来处理输入数据的系数，神经网络在学习过程中不断调整参数，直至能准确预测出每一幅图像的标签――此时就得到了一个比较准确的模型。</p>
</section>

<section>
  <h2 id="building">搭建神经网络</h2>
  <p>我们将按<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank">Xavier Glorot和Yoshua Bengio的论文</a>中的描述来建立一个前馈神经网络。在这个基础示例中，我们先搭建一个只有单一隐藏层的神经网络。基本的原则是，深度更深（层数更多）的网络能处理更复杂的数据，捕捉更多细节，进而得出更准确的结果。</p>
  <img src="/img/onelayer.png"><br><br>
  <p>请将上图牢记心中，这就是我们将要搭建的单层神经网络。</p>
  <h3>设置超参数</h3>
  <p>不论用Deeplearning4j搭建何种神经网络，其基础都是<a href="http://deeplearning4j.org/neuralnet-configuration.html" target="_blank">NeuralNetConfiguration类</a>。我们用这个类来配置各项超参数，其数值决定了网络的架构和算法的学习方式。直观而言，每一项超参数就如同一道菜里的一种食材：取决于食材好坏，这道菜也许非常可口，也可能十分难吃……所幸在深度学习中，如果结果不正确，超参数还可以进行调整。</p>
  <pre><code class="language-java">
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
  </code></pre>
  <h5>.seed(rngSeed)</h5>
  <p>该参数将一组随机生成的权重确定为初始权重。如果一个示例运行很多次，而每次开始时都生成一组新的随机权重，那么神经网络的表现（准确率和F1值）有可能会出现很大的差异，因为不同的初始权重可能会将算法导向误差曲面上不同的局部最小值。在其他条件不变的情况下，保持相同的随机权重可以使调整其他超参数所产生的效果表现得更加清晰。</p>
  <h5>.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)</h4>
  <p>随机梯度下降（Stochastic Gradient Descent，SGD）是一种用于优化代价函数的常见方法。要了解SGD和其他帮助实现误差最小化的优化算法，可参考<a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng的机器学习课程</a>以及本网站<a href="http://deeplearning4j.org/glossary#stochasticgradientdescent" target="_blank">术语表</a>中对SGD的定义。</p>
  <h5>.iterations(1)</h5>
  <p>对一个神经网络而言，一次迭代（iteration）指的是一个学习步骤，亦即模型权重的一次更新。神经网络读取数据并对其进行预测，然后根据预测的错误程度来修正自己的参数。因此迭代次数越多，网络的学习步骤和学习量也越多，让误差更接近最小值。</p>
  <h5>.learningRate(0.006)</h5>
  <P>本行用于设定学习速率（learning rate），即每次迭代时对于权重的调整幅度，亦称步幅。学习速率越高，神经网络“翻越”整个误差曲面的速度就越快，但也更容易错过误差最小点。学习速率较低时，网络更有可能找到最小值，但速度会变得非常慢，因为每次权重调整的幅度都比较小。</p>
  <h5>.updater(Updater.NESTEROVS).momentum(0.9)</h5>
  <P>动量（momentum）是另一项决定优化算法向最优值靠拢的速度的因素。动量影响权重调整的方向，所以在代码中，我们将其视为一种权重的更新器（<code>updater</code>）。</p>
  <h5>.regularization(true).l2(1e-4)</h5>
  <p>正则化（regularization）是用来防止<b>过拟合</b>的一种方法。过拟合是指模型对定型数据的拟合非常好，然而一旦在实际应用中遇到从未出现过的数据，运行效果就变得很不理想。</p>
  <p>我们用L2正则化来防止个别权重对总体结果产生过大的影响。</p>
  <h5>.list()</h5>
  <p>该函数可指定网络中层的数量；它会将您的配置复制n次，建立分层的网络结构。</p>
  <p>再次提醒：如果对以上任何内容感到困惑，建议您参考<a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng的机器学习课程</a>。</p>
  <h3>搭建层</h3>
  <p>我们不会详细解释每项超参数（如activation、weightlnit等）背后的研究；此处仅对超参数的功能定义作简要介绍。若要了解相关研究的详情，请阅读<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank">Xavier Glorot和Yoshua Bengio的论文</a>。</p>
  <img src="/img/onelayer_labeled.png"><br>
  <pre><code class="language-java">
    .layer(0, new DenseLayer.Builder()
            .nIn(numRows * numColumns) // Number of input datapoints.
            .nOut(1000) // Number of output datapoints.
            .activation("relu") // Activation function.
            .weightInit(WeightInit.XAVIER) // Weight initialization.
            .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(1000)
            .nOut(outputNum)
            .activation("softmax")
            .weightInit(WeightInit.XAVIER)
            .build())
    .pretrain(false).backprop(true)
    .build();
  </code></pre>
  <p>隐藏层究竟是什么？</p>
  <p>隐藏层中的每个节点（上图中的圆圈）表示MNIST数据集中一个手写数字的一项特征。例如，假设现在处理的数字是6，那么一个节点可能表示圆形的边缘，另一个节点可能表示曲线的交叉点，等等。模型的系数按照重要性大小为这些特征赋予权重，随后在每个隐藏层中重新相加，帮助预测当前的手写数字是否确实为6。节点的层数更多，网络就能处理更复杂的因素，捕捉更多细节，进而做出更准确的预测。</p>
  <p>之所以将中间的层称为“隐藏”层，是因为人们可以看到数据输入神经网络、判定结果输出，但网络内部的数据处理方式和原理并非一目了然。神经网络模型的参数其实就是包含许多数字、计算机可以读取的长向量。</p>
</section>

<section>
  <h2 id="training">模型定型</h2>
  <p>模型已经建立完毕，可以开始定型了。点击IntelliJ界面右上方的绿色箭头，运行上文给出的代码。</p>
  <img src="/img/mlp_mnist_single_layer_example_training.png"><br><br>
  <p>运行过程需要几分钟时间，具体时长取决于您的硬件情况。</p>
</section>

<section>
  <h2 id="evaluating">结果评估</h2>
  <img src="/img/mlp_mnist_single_layer_example_results.png"><br><br>
  <p>
  <b>准确率</b> - 模型准确识别出的MNIST图像数量占总数的百分比。<br>
  <b>精确率</b> - 真正例的数量除以真正例与假正例数之和。<br>
  <b>召回率</b> - 真正例的数量除以真正例与假负例数之和。<br>
  <b>F1值</b> - <b>精确率</b>和<b>召回率</b>的加权平均值。<br>
  </p>
  <p><b>准确率</b>衡量的是模型的总体表现。</p>
  <p><b>精确率、召回率和F1值</b>衡量的是模型的<b>相关性</b>。举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，因为病人会不再寻求进一步治疗。所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），尽管总体上的<b>准确率</b>可能会相对较低一些。</p>
</section>

<section>
  <h2 id="conclusion">总结</h2>
  <p>成功了！至此，您已经训练出了一个准确率达到97.1%的神经网络，整个过程不需要任何计算机视觉领域的知识。当前最顶尖的水平比这还要高，您可以借由调整超参数来进一步改善模型的表现。</p>
  <p>与其他深度学习框架相比，Deeplearning4j具有以下优点。</p>
  <ul>
    <li>与Spark、Hadoop、Kafka等主流JVM框架实现大规模集成</li>
    <li>专为基于分布式CPU和/或GPU运行而优化</li>
    <li>服务于Java和Scala用户群</li>
    <li>企业级部署可享商业化支持</li>
  </ul>
  <p>若有任何问题，请加入我们的<a href="https://gitter.im/deeplearning4j/deeplearning4j" target="_blank">Gitter线上支持交流群</a>。</p>
  <ul class="categorized-view view-col-3">
    <li>
      <h5>其他Deeplearning4j教程</h5>
      <a href="https://deeplearning4j.org/cn/zh-neuralnet-overview">深度神经网络简介</a>
      <a href="https://deeplearning4j.org/cn/zh-restrictedboltzmannmachine">受限玻尔兹曼机基础教程</a>
      <a href="http://deeplearning4j.org/cn/zh-eigenvector">本征向量、协方差、PCA和熵</a>
      <a href="http://deeplearning4j.org/cn/zh-lstm">LSTM和循环网络基础教程</a>
      <a href="http://deeplearning4j.org/linear-regression">神经网络与线性回归</a>
      <a href="http://deeplearning4j.org/cn/zh-convolutionalnets">卷积网络</a>
    </li>

    <li>
      <h5>学习资源推荐</h5>
      <a href="https://www.coursera.org/learn/machine-learning/home/week/1">Andrew Ng的机器学习网络课程</a>
      <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java">LeNet示例：用卷积网络处理MNIST</a>
    </li>

  </ul>
</section>

---
title: 深度置信网络教程：鸢尾花数据集
layout: cn-default
---

# 深度置信网络教程：鸢尾花数据集

**鸢尾花数据集的规模比较小，所以用该数据集定型的神经网络的输出可能各不相同。**

深度置信网络（DBN）是一种多类别分类器。在处理许多分属不同类别的输入数据时，DBN可以先用一个较小的定型数据集进行学习，然后再将未标记数据分入相应类别。DBN可以接受一项输入并判断应为数据记录赋予哪种标签。 

DBN会从一组标签中为输入的数据记录选择一个合适的标签，可以处理比布尔型的“是否”标签更复杂的多名分类法。 

DBN网络输出一个向量，其中包含来自每个输出节点的一个数值。输出节点的数量等于标签数量。每个输出值均为0或1，所有的0和1共同组成输出的向量。 

![image of nn multiclassifier here](http://i.imgur.com/qfQWwHB.png)

### 鸢尾花数据集

[鸢尾花数据集](https://archive.ics.uci.edu/ml/datasets/Iris)被广泛用于测试各类机器学习分类算法。我们用它来验证神经网络的有效性。

该数据集包括三种鸢尾花各50个样例的四项测量数据，总共有150朵花、600个数据点。山鸢尾（Iris setosa）、维吉尼亚鸢尾（Iris virginica）、杂色鸢尾（Iris versicolor）三个品种的花瓣和萼片（花瓣底部的绿色叶状薄片）长度各不相同。数据集中的测量数据包括花瓣和萼片的长度与宽度，以厘米为单位，三个不同的种名为标签。 

鸢尾花数据集的连续测量数据很适合用来测试连续深度置信网络。上述四项特征足以实现三个鸢尾花品种的准确分类。成功的定义是：经过学习之后，您的神经网络仅依靠每朵花的尺寸数据就能确定花的种类；如果无法做到这一点，那就表明网络还需要调试。 

鸢尾花数据集规模较小，这有可能会造成一些问题；此外，I. virginica和I. versicolor两个品种十分相似，其尺寸数据是部分重叠的。 

以下是一条数据记录：

| 萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 品种 |
| :------------- | :-------------| :------------- | :-------------| :-------------|
| 5.1 | 3.5 | 1.4 | 0.2 | i.setosa |

上述表格是供人阅读用的，Deeplearning4j算法能够识别的形式更接近于：

     5.1,3.5,1.4,0.2,i.setosa

事实上，我们还需要去除其中的词语，把整条记录变为两个数值形式的数据对象：

数据：5.1,3.5,1.4,0.2    
标签：0,1,0

我们可以采用三个二元判定的输出节点，将三种鸢尾花品种标记为1,0,0、0,1,0、0,0,1。 

### 加载数据

DL4J采用一种称为DataSet的对象来将数据加载到神经网络中。DataSet可以存储需要进行预测的数据（及其相关标签），具体方式很简单。下表中的第一和第二列都是NDArray。其中一个NDArray保存数据的属性，另一个则保存标签。（您可以[用这个文件](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/iris/IrisExample.java)来运行本示例。）

| 第一列（需预测的数据） | 第二列（结果，即标签） |
| :------------- | :-------------|
| 呱 | 蛙 |
| 呱 | 蛙 |
| 汪 | 狗 |
| 喵 | 猫 |

（一个DataSet对象中包含两个NDArray，NDArray即N维数组，是DL4J用于数值计算的基本对象。N维数组是适用于精密数学运算的可扩展多维数组，在科学计算中很常用。） 

大多数程序员对CSV（逗号分隔值）文件类型的数据集都不陌生，鸢尾花数据集也采用这种形式。您可以用以下方法来解析鸢尾花CSV文件，将其转换为DL4J所能识别的对象。 

    File f = new File(“Iris.dat”);
    InputStream fis = new FileInputStream(f);

    List<String> lines = org.apache.commons.io.IOUtils.readLines(fis);
    INDArray data = Nd4j.ones(to, 4);
    List<String> outcomeTypes = new ArrayList<>();
    double[][] outcomes = new double[lines.size()][3];

具体分析一下上述过程：iris.dat这个CSV文件中包含了需要输入网络的数据。

此处我们用一个名为*IOUtils*的Apache库来帮助从文件流中读取数据。请注意，readLines会将所有数据复制到内存中（生产环境中一般不应这么操作），所以应当考虑改用BufferedReader对象。

NDArray的*data*变量将会保存数值形式的原始数据，而*outcomeTypes*列表则会以映射形式保存标签。Dataset的*completedData*对象包含了所有数据，包括二进制化的标签，如以下代码的结尾处所示。 

*outcomes*变量是一个由double元素构成的二维数组，其行数与我们的记录数量相同（即iris.dat中的记录条数），列数则与标签数量相同（即三个鸢尾花品种）。这一数组中会包含已转换为二进制的标签。

请看以下代码片段：

    for(int i = from; i < to; i++) {
        String line = lines.get(i);
        String[] split = line.split(",");

         // 将4个数值转换为double类型并相加
        double[] vector = new double[4];
        for(int i = 0; i < 4; i++)
             vector[i] = Double.parseDouble(line[i]);

        data.putRow(row,Nd4j.create(vector));

        String outcome = split[split.length - 1];
        if(!outcomeTypes.contains(outcome))
            outcomeTypes.add(outcome);

        double[] rowOutcome = new double[3];
        rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
        outcomes[i] = rowOutcome;
    }

    DataSet completedData = new DataSet(data, Nd4j.create(outcomes));

我们来具体说明一下这些代码。

第3行：由于我们处理的是CSV数据，因此可以用*split*来按逗号切分数据，存储到String数组*split*中。

第6～10行：我们的String对象是数字字符串。也就是说，如果数据中有“1.5”，那么它并不是double类型的数值，而是一个包含字符“1.5”的String对象。我们创建一个称为*vector*的临时数组，将字符存储在其中备用。 

第12～14行：抓取String数组中的最后一个元素，获取标签。然后将标签二进制化。为此，我们需要收集outcomeTypes列表中的所有标签，这就引出了下一个步骤。

第16～18行：开始用outcomeTypes列表来将标签转换为二进制。每个标签都有一个特定的位置，亦即索引值，我们用这个索引值来将标签状态映射至我们所创建的标签行。因此，如果标签是*i. setosa*，我们会将其置于outcomeTypes列表的最后，在那里创建一个大小为三个元素的新标签行，把rowOutcome中对应i. setosa的那个位置标记为1，另两个则标记为0。最后，我们将rowOutcome保存至先前创建的二维数组outcomes中。 

整个过程完成后，我们应能得到以数值形式表示标签的一个行。分类为*i. setosa*的数据记录可能如下所示：

| 萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | I. setosa | I. virginica | I. versicolor |
| :------------- | :-------------| :------------- | :-------------| :-------------| :-------------| :-------------|
| 5.1 | 3.5 | 1.4 | 0.2 | 1 | 0 | 0 |

表格第一行的文字仅用于教程说明，提醒我们数字的具体含义。第二行的数字才是数据处理中向量所包含的内容。我们将已完成转换的第二行称为*向量化数据*。

第21行：接下来就可以考虑如何为DL4J来包装数据了。为此，我们要创建一个*DataSet*对象，其中包含所需使用的数据以及相应的二进制标签。

最后返回的completedData列表就是深度置信网络能够处理的数据集了。 

### 创建神经网络（NN）

现在就可以创建深度置信网络（DBN）来对输入数据进行分类了。

在DL4J中，任何神经网络的创建过程都包括以下几个步骤。 

首先需要创建一个网络配置对象：

    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
    .hiddenUnit(RBM.HiddenUnit.RECTIFIED).momentum(5e-1f)
        .visibleUnit(RBM.VisibleUnit.GAUSSIAN).regularization(true)
        .regularizationCoefficient(2e-4f).dist(Distributions.uniform(gen))
        .activationFunction(Activations.tanh()).iterations(10000)
        .weightInit(WeightInit.DISTRIBUTION)
    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
        .learningRate(1e-3f).nIn(4).nOut(3).build();

这样就囊括了DBN分类器所需的全部要素。如您所见，以上的配置包含了许多参数，亦即“调试开关” ，您可以不断学习如何通过调整这些参数来改善网络性能。参数就好比是操纵DL4J深度学习引擎的踏板、排挡和方向盘。 

具体的参数类型包括但不仅限于：动量、正则化（是否采用）及其系数，迭代次数、学习速率、输出节点数量、每个节点层的变换算法（例如高斯或修正变换）等。 

我们还需要一个随机数生成器对象：

        RandomGenerator gen = new MersenneTwister(123);

Finally, we create the DBN itself: dbn

    DBN dbn = new DBN.Builder().configure(conf)
        .hiddenLayerSizes(new int[]{3})
        .build();
      dbn.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
      dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

分析一下上面的代码。在第一行中，我们将称为“conf”的Configuration对象作为参数输入，然后指定隐藏层的大小。我们可以用一个独立的数组来为每个隐藏层指定不同的大小。本示例采用单个隐藏层，包含三个节点。 

接下来可以对先前创建的DataSet对象进行预加工。我们把它放到一个独立的irisLoad()函数中。

    DataSet ourDataSet = loadIris(0, 150);
    ourDataSet.normalizeZeroMeanZeroUnitVariance();
    dbn.fit(ourDataSet);

注意上述代码的第二行。对许多机器学习模型而言，我们都必须先将数据标准化，确保模型不会因离群值而发生扭曲。数据标准化指将不同数量级的值（变化幅度为几十、几百或几百万）缩放至一个常见的范围内，比如0到1之间。只有在同一个范围内才能进行同类之间的比较……

最后，我们调用*fit*方法，用准备好的数据集来定型模型。 

在模型的定型过程中，算法会学习如何从输入数据中提取可以作为分类提示信号的具体特征，亦即将不同品种区分开来的特征。

定型就是不断尝试根据各种提取的特征来对输入数据进行分类。定型包括两个主要步骤：将模型的预测结果与测试数据集中的正确答案进行比较；然后对网络进行相应的奖励或惩罚，使之进一步靠近或远离正确答案。用充足的数据充分定型之后，网络即可用于对鸢尾花数据进行无监督的分类，作出颇为精确的预测。 

如果打开了debug功能，您应当会在最后一行代码运行之后看到一些输出结果。 

### **结果评估**

以下的代码片段出现在调用*fit()*之后。

    Evaluation eval = new Evaluation();
    INDArray output = d.output(next.getFeatureMatrix());
    eval.eval(next.getLabels(),output);
    System.out.printf("Score: %s\n", eval.stats());
    log.info("Score " + eval.stats());

DL4J用一个**Evaluation**对象来收集有关模型性能表现的统计信息。INDArray输出是由一系列*DataSet.getFeatureMatrix()*和**output**的链式调用生成的。调用getFeatureMatrix会返回一个所有数据输入的NDArray，该数组随即进入**output()**。**output()**方法会标记一项输入的概率，此处的输入即是我们的特征矩阵。*eval*的作用就是记录模型预测与真实结果的偏差和吻合情况。 

**Evaluation**对象包括许多有用的调用命令，例如*f1()*。该方法可以估计一个模型的准确率，结果表示为概率（以下的F1值表明模型认为其分类的准确率约为77%）。其他方法包括：*precision()*，该方法可以分析网络在输入数据相同时能否得出同样的预测结果；*recall()*，该方法可以衡量网络召回了多少正确的结果。

本示例的结果如下：

     Actual Class 0 was predicted with Predicted 0 with count 50 times

     Actual Class 1 was predicted with Predicted 1 with count 1 times

     Actual Class 1 was predicted with Predicted 2 with count 49 times

     Actual Class 2 was predicted with Predicted 2 with count 50 times

    ====================F1Scores========================
                     0.767064393939394
    ====================================================

网络定型完毕后，您就会看到这样的F1值。在机器学习中，F1值是用于衡量分类器性能的指标之一。F1值是一个零到一之间的数值，可以说明网络在定型过程中表现如何。它与百分比相类似，F1值为1相当于100%的预测准确率。F1值基本上相当于神经网络作出准确预测的概率。

我们的模型还未调试到最理想的状态（得再回去调试开关！），这也只是第一次遍历，结果还不错。

可以有效处理鸢尾花数据集的神经网络最终应能得出与下图相似的可视化表示：

![Alt text](../img/iris_dataset.png)

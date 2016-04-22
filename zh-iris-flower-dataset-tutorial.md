---
title: "使用深度信念网来运行鸢尾花分类"
layout: zh-default
---

# 使用深度信念网来运行鸢尾花分类

深度信念网:Deep-Belief Net ((DBN)是多级分类的。既然很多输入都属于不同的类别, DBN可以先从小训练集开始学习,然后根据这些不同类别,无标签的数据进行分类。它可以根据数据记录,决定一个输入的标签。

DBN会从一组标签选择一个适合哪个输入记录的标签。这比了Boolean的“对”与“错”输入处理更广泛,更多项分类。

这网络在每个输出节点输出一个拥有一个数字的向量。而输出节点的数量就相等于标签数。每个输出都将会是一个0或1,并且把0和1加在一起构成的矢量。

(要运行这例子,[使用这个文件](https://github.com/SkymindIO/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/iris/IrisExample.java)。)

## 鸢尾花数据集(IRIS Dataset)

的鸢尾花数据集([Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/Iris))被广泛应用于机器学习来测试分类技术。我们将用它来验证我们的神经网络的有效性。

该数据集包括从每个3种鸢尾花的50样品采取的四个测量(总共有150朵花,600个数据)。这些鸢尾花品种具有不同的长度,以厘米测量的花瓣和萼片(绿色,叶状护套在花瓣的基部)。我们采取了Iris-setosa,Iris-Virginica,和Iris-Versicolor的萼片和花瓣的长度和宽度。每个鸢尾花的品种的名字将作为标签。

这些测量的连续性使这鸢尾花数据集成功测试连续深度信念网。仅仅这四个功能就足以准确的分类三种鸢尾花。换句话说,成功就是指教一个神经网络根据记录的个别数据(只知道个别的尺寸)进行鸢尾花分类。如果无法完成分类,这是一个非常强烈的信号告诉您您的神经网络需要修正。

该数据集是很小的,它可以展现出它自己的问题和种类:I. virginica 和 I. versicolor - 它们是如此的相似至到一些部分重叠。

这是一个记录:

![data record table](../img/data_record.png)

虽然我们人类都可以理解上面的表,但是Deeplearning4j的算法需要它的东西如下:

     5.1,3.5,1.4,0.2,i.setosa

事实上,让我们把向前移一步把这些字都拿掉,把这些数值数据重新安排成两个对象:

数据: 5.1,3.5,1.4,0.2

标签: 0,1,0

由于二进制的决定是基于三个输出节点,我们可以把这三个鸢尾花标记为1,0,0, 或0,1,0,或0,0,1。

## 加载这数据

DL4J使用一个称为DataSet的对象将数据加载到一个神经网络。DataSet是用来储存预测数据(及其相关的标签)。以下的第一和第二列都是NDArrays 。其中一个NDArrays将保持数据的属性;其他保持的数据的标签。

![input output table](../img/ribbit_table.png)

( 包含在DataSet对象内的有两个NDArrays ,即是DL4J用于数值计算的基本对象)N二维数组是可伸缩及多维数组的,它适合于复杂的数学运算,并经常用于科学计算。 )

大多数的程序员都熟悉包含在类似的CSV (逗号分隔型)文件中的数据集。而鸢尾花数据集也不例外。这里是教你如何分析一个鸢尾花CSV ,并把它变成一个DL4J可以理解的对象。


    File f = new File(“Iris.dat”);
    InputStream fis = new FileInputStream(f);
    List<String> lines = org.apache.commons.io.IOUtils.readLines(fis);
    INDArray data = Nd4j.ones(to, 4);
    List<String> outcomeTypes = new ArrayList<>();
    double[][] outcomes = new double[lines.size()][3];

让我们来一个一个为您解释: iris.dat是一个CSV文件,里边含有我们需要输入神经网络的数据。

在这里,我们使用IOUtils ,一个Apache库,以帮助从文件中读取数据流中的数据。请注意,readLines方法会将所有的数据复制到内存(一般您在在生产时不应该这样做) 。相反,考虑一个BufferedReader对象。

该NDArray可变数据会保留我们的原始数字数据,而listoutcomeTypes将是一种映射,它包含我们的标签。该数据集objectcompletedData (在以下代码的尾端),它包含了我们所有的数据,也包括了二值化标签。

可变的结果将是一个双二维阵列,这双二维阵列具有我们所有记录的行(即在iris.dat线),也具有我们所有标签的列(即3种种类)。这也将包含我们的二值化标签。

看看这段代码: 

    for(int i = from; i < to; i++) {
        String line = lines.get(i);
        String[] split = line.split(",");

         // turn the 4 numeric values into doubles and add them
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

好,是时候看看我们所写的。

第3行:因为我们正在处理的CSV数据,我们可以只使用拆分来标记每个逗号和存储数据在String数组拆分。

第6 - 10行 :我们的String对象是数字的字符串。也就是说,我们会有 ”1.5“字符的String对象,而不是一对 1.5。我们将创建一个临时数组,称为向量,并保存这字符以备后用。

第12-14行:我们通过拿取String数组最后一个元素来取得标签。现在,我们可以想一想二进制化的标签。为了做到这一点,我们将收集所有的标签列表中的outcomeTypes ,这是我们迈向下一步的桥梁。

第16-18行:我们开始使用outcomeTypes名单来二值化这些标签。每个标签都有一定的地位,或索引,我们将使用该索引号映射到我们在这里创建的标签行。所以,如果i. setosa是标签,我们将把它放在outcomeTypes列表的末尾。从那里,我们将创建一个新的标签行,三个元素为它的大小,并在rowOutcome的对应位置标志为1,0为另两个。最后,我们将保存rowOutcome到我们之前创建的二维数组的结果。

当我们完成的时候,我们将有一排带着数字表示的标签。数据记录列为i. setosa会是这个样子的:

![final table](../img/final_table.png)

您在框上部的第一列看到的词只是为了向您说明我们的教程,也顺便提醒我们应该使用哪个词和号码。在框下部的数将会以向量的形式出现来处理数据。事实上,在底部,完成的排就是我们所说的量化数据。

第21行:现在我们可以开始考虑包装DL4J的数据。为了做到这一点,我们创建我们想要使用和附带的,二值化标签的,单个DataSet对象。

最后,我们将返回completedData列表,这列表是一个能在深度信念网运行的数据集。

## 创建一个神经网络( Neural Network: NN )

现在,我们准备建立一个深度信念网络或DBN来分类我们的输入。

使用DL4J创建任何一种类型的神经网络需要进行一些步骤。

首先,我们需要创建一个配置对象:

    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
    .hiddenUnit(RBM.HiddenUnit.RECTIFIED).momentum(5e-1f)
        .visibleUnit(RBM.VisibleUnit.GAUSSIAN).regularization(true)
        .regularizationCoefficient(2e-4f).dist(Distributions.uniform(gen))
        .activationFunction(Activations.tanh()).iterations(10000)
        .weightInit(WeightInit.DISTRIBUTION)
    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
        .learningRate(1e-3f).nIn(4).nOut(3).build();

这个拥有一切我们的DBN分类所需要的。正如你所看到的,有很多的参数,或“旋钮”,你将慢慢学会如何调整您的网,以提高它的性能。这些都是连接到DL4J深度学习引擎的所有零件。

这些包括(但不限于):动量,正规化的量(是或否)及其系数,迭代的次数,对所述学习率的速度,输出节点的数量,并且连接到每个节点层的转换(如高斯:Gaussian或整流) 。

我们也需要一个随机数产生器的客体:

        RandomGenerator gen = new MersenneTwister(123);

最后,我们创建了DBN本身: dbn

    DBN dbn = new DBN.Builder().configure(conf)
        .hiddenLayerSizes(new int[]{3})
        .build();
      dbn.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
      dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

我们来分析一下上面的代码。在第一行,我们把我们称为’conf’的配置对象作为一个参数。然后,我们指定隐藏层的大小。我们可以分离每个层的阵列。在这种情况下,有一个单一的隐含层,三个节点长。

现在,我们可以开始准备刚才我们做的DataSet对象。我们把它放在一个单独的irisLoad()功能。

    DataSet ourDataSet = loadIris(0, 150);
    ourDataSet.normalizeZeroMeanZeroUnitVariance();
    dbn.fit(ourDataSet);

注意上面的第二行。在很多机器学习模型里,标准化数据以确保离群值不扭曲模型是很重要的。[数字正常化意味着调整可能会从不同的比例(在几十或几百或几百万)至到常见的比例去测量。比如说, 0和1之间只能“苹果与苹果之间比较”,如果一切是“苹果规模”...]

最后,我们调用适合训练模型的数据集。

通过在数据集对模式的训练,您的算法将学习如何提取数据的那些特定特征,这对于分类目标输入的信号有帮助,它区分了各个种类。

培训是基于各种机器提取的特征尝试进行反复的分类。它包括两个主要步骤:比较这些猜测与在测试组中的原数据;当它接近或远离正确的答案时会有网的”奖励“或”惩罚“。有了足够的数据和培训,这网可能会用来分类无人监督的鸢尾花数据,其精度将会相当高。

如果debug已开启,你应该在运行的最后一行看到一些输出。

## 评估我们的结果

在使用我们的 fit()之后,考虑使用以下的代码片段的代码。

    Evaluation eval = new Evaluation();
    INDArray output = d.output(next.getFeatureMatrix());
    eval.eval(next.getLabels(),output);
    System.out.printf("Score: %s\n", eval.stats());
    log.info("Score " + eval.stats());

DL4J使用的评估对象(Evaluation Object)来收集有关模型的性能统计信息。该INDArray输出由DataSet.getFeatureMatrix()和输出的链式调用创建。该getFeatureMatrix调用将返回所有NDArray()数据输入,而这些输入是被送入 output()的。此方法将标记的输入的概率,在这案例就是我们的特征矩阵。 eval本身只是收集缺失以及收集模型的预测和实际结果的点击率。

评价对象(Evalution object)包含了许多有用的调用,如F1()。这方法会使用概率的形式来估计模型的准确度( 如果F1的得分低于模型的平均数,这可能表示其归类约77%的准确率)。其他方法包括precision():它告诉我们模型预测结果的可靠性(给予相同的输入下);而recall(),它会告知我们获取了多少个正确结果。

在这个例子中,我们将会拥有以下这些

     Actual Class 0 was predicted with Predicted 0 with count 50 times

     Actual Class 1 was predicted with Predicted 1 with count 1 times

     Actual Class 1 was predicted with Predicted 2 with count 49 times

     Actual Class 2 was predicted with Predicted 2 with count 50 times

    ====================F1Scores========================
                     0.767064393939394
    ====================================================

当您的网络已受培训后,您会看到这样的F1分数。在机器学习,F1分是一个度量来确定分类进行的性能。它是一个0和1之间的数字,这数字将告诉您在训练过程中您的网络的运作性能。它类似于一个百分比,1为100%的预测准确性。它基本上是您的网络的正确猜测的概率。

我们的模型没好好地调整(回到旋钮! ) ,而这仅仅是第一个关卡,但是很不错了!

最后,鸢尾花数据集在一个能运作的网络的可视化将是这个样子:

![Alt text](../img/iris_dataset.png)

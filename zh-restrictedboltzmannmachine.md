---
title: "受限玻尔兹曼机(RBM)"
layout: zh-default
---

# 受限玻尔兹曼机(RBM)

本网站正在更新中，如想要获得最新的信息，[请参考](../restrictedboltzmannmachine.html) 

引用Geoff Hinton(一个谷歌研究员,也是一名大学教授),玻尔兹曼机是“一个对称连接,利用类似神经元作单位来随机决定开关的网络”。(随机的意思是“[随机确定的](http://deeplearning4j.org/glossary.html#stochasticgradientdescent)” )

受限玻尔兹曼机“拥有一层可见的单位和一层隐藏单元,它们是无明显可见的或隐藏的隐藏连接的。”这个“受限”来自它的节点连接的强加限制:层内连接是不允许的,但一个层的每一个节点会连接到的下一个节点,这就是“对称”。

所以, RBM的'节点必须形成一个对称的二分图,数据将从底部的可视层( V0 -V3 )到顶部的隐藏层(H0 -H 2),如下:

![Alt text](../img/bipartite_graph.png)

一个训练有素的限制波尔兹曼机将通过可视层学习输入的数据的结构;它通过一次又一次的数据重建,每一次的重建都会增加它们的相似性度(与原始数据作为基准来比较)。这由RBM重组的数据与原数据的相似性度是使用损失函数来衡量。

RBM对于维度(dimensionality),降维(reduction),分类(classification),协同(collaborative),过滤(filtering),特征(feature),学习(learning)和课题建模(topic modeling)都非常有用。因为RBM的操作非常简单,使用受限玻尔兹曼机为神经网络是我们的第一个选择。

## 参数和K

请参考[所有单层网络的共同参数](http://deeplearning4j.org/singlelayernetwork.html)。

变量k是运行[对比分歧](http://deeplearning4j.org/glossary.html#contrastivedivergence)的次数。每一次的对比分歧运行就如马尔可夫链构(Markov chain)构成限制波尔兹曼机。通常它的值是1 。

## 在Iris启动RBM


		public class RBMIrisExample {		
 		
     private static Logger log = LoggerFactory.getLogger(RBMIrisExample.class);		
 		
     public static void main(String[] args) throws IOException {		
         // Customizing params		
         Nd4j.MAX_SLICES_TO_PRINT = -1;		
         Nd4j.MAX_ELEMENTS_PER_SLICE = -1;		
         Nd4j.ENFORCE_NUMERICAL_STABILITY = true;		
         final int numRows = 4;		
         final int numColumns = 1;		
         int outputNum = 10;		
         int numSamples = 150;		
         int batchSize = 150;		
         int iterations = 100;		
         int seed = 123;		
         int listenerFreq = iterations/2;		
 		
         log.info("Load data....");		
         DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);		
         // Loads data into generator and format consumable for NN		
         DataSet iris = iter.next();		
 		
         iris.normalizeZeroMeanZeroUnitVariance();		
 		
         log.info("Build model....");		
         NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().regularization(true)		
                 .miniBatch(true)		
                 // Gaussian for visible; Rectified for hidden		
                 // Set contrastive divergence to 1		
                 .layer(new RBM.Builder().l2(1e-1).l1(1e-3)		
                         .nIn(numRows * numColumns) // Input nodes		
                         .nOut(outputNum) // Output nodes		
                         .activation("relu") // Activation function type		
                         .weightInit(WeightInit.RELU) // Weight initialization		
                         .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)		
                         .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)		
                         .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)		
                         .build())		
                 .seed(seed) // Locks in weight initialization for tuning		
                 .iterations(iterations)		
                 .learningRate(1e-3) // Backprop step size		
                 // Speed of modifying learning rate		
                 .optimizationAlgo(OptimizationAlgorithm.LBFGS)		
                         // ^^ Calculates gradients		
                 .build();		
         Layer model = LayerFactories.getFactory(conf.getLayer()).create(conf);		
         model.setListeners(new ScoreIterationListener(listenerFreq));		
 		
         log.info("Evaluate weights....");		
         INDArray w = model.getParam(DefaultParamInitializer.WEIGHT_KEY);		
         log.info("Weights: " + w);		
         log.info("Scaling the dataset");		
         iris.scale();		
         log.info("Train model....");		
         for(int i = 0; i < 20; i++) {		
             log.info("Epoch "+i+":");		
             model.fit(iris.getFeatureMatrix());		
         }		
     }		
     // A single layer learns features unsupervised.	
    }

 
 ## 连续RBMs(CRBMs)
 
 连续RMBs(CRBMs)是RBM通过不同类型对比分歧抽样来接受连续输入(例如:数字低于整数)的一种形式。这允许CRBM处理被归一化到0与1之间的小数的图像像素或计数向量。
 
应当注意的是,深度学习网的每一层包括四个要素:输入,系数,偏置和转化。

其输入是从之前的层输入的数字数据和一个向量(或作为原数据)。系数是给予权重的特点,而这特点是通过各节点层的。偏置确保了那一层的一些节点再不管什么情况下都会被激活。转化是一个额外的算法,它会处理在通过每个层之后的数据。

这些额外的算法,以及它们的组合在每个层次都不同。我们发现最有效的CRBMs是使用高斯转型(Gaussian transformation)在可见层(或输入)和隐藏层上的整流线性单位转换。[我们发现这在面部重建特别有用](http://deeplearning4j.org/facial-reconstruction-tutorial.html)。对于RBMs如何处理二进制数据,您只要把这两者的转换变成二进制的。

综上所述:Geoff Hinton曾指出,而我们也可以证实,高斯转换(Gaussian transformation)不能与RBMs的隐藏层好好配合,这也是重建出现的时候;即这些都是重要的层。在此,我们采用线性整流单位转换在[深度信念网](http://deeplearning4j.org/deepbeliefnetwork.html)是因为它比二进制转换的功能多。

## 结论及下一步

你可以把RBM的'输出数字解为百分比。如果每次重组后的数字不是零,这是一个很好的迹象说明RBM正在学习输入。我们在后面的教程会有一个更好的例子。

如果想要了解更多关于如何运行有限玻尔兹曼机打勾的机制,请点击[这里](http://deeplearning4j.org/understandingRBMs.html)。

接下来,我们将向您展示如何实施深度信念网,这简单来说就是把很多个受限玻尔兹曼机串在一起。

---
title: WellDressed推荐引擎
layout: default
---

# WellDressed推荐引擎

[*作者：Stephan Duquesnoy*](https://twitter.com/stephanduq)

## 背景

我个人的专业背景是戏剧和IT。 

以前有几年时间，我曾经营过一家为娱乐行业提供外包服务的美术工作室，给游戏、电影创作原画和视觉资产。我的专长是人物设计。我一直是比较讲究模式的人，所以我的设计作品往往植根于我自己研究过的调和设计模式，我也曾用算法和可教授概念来描述这些模式。 

出于实验的目的，我将这种算法转化成了一个可以提示与天气状况相匹配的和谐色调的程序。后来我又将服饰加入程序中，Well Dressed由此诞生。 

[*Well Dressed*](http://welldressed-app.com/)是一款根据你的相貌以及天气、场合、预算等因素推荐着装的应用程序。它在[2015年的爱尔兰Web Summit峰会上首次亮相](http://goos3d.ie/best-startups-at-web-summit-2015/)，到目前为止还是一个单人运作的项目。顺便一提，虽然我比较擅长于基于模式的思维方式，但我并没有这方面的学术背景。我缺乏耐心，更渴望创造，不太在意能否完全理解自己所做的事情。

## 问题

为了打造Well Dressed，我需要许多代表不同服装类型的数据类别，因为程序要给出的是着装风格建议。 

但是商家提供的数据源只有寥寥几种分类选项，而且更麻烦的是，不同商家的数据源还有很大差异。所以我需要一种解决方案来分析数据源中可得的信息，并且设法使之匹配我自己的数据库。 

我的第一套解决方案基于关键词，采用了复杂的规则及权重系统。这个方案多少有些效果，但还不够理想。准确率达到了65%，而我每天必须重新检查每一件新衣服，确保数据正确无误，这要耗费许多本来可以用于市场营销或者产品开发的时间。

## 数据预处理

原始数据来自世界各地的服装店的数据源。 

我决定专注于“名称”（Title）、“说明”（Description）和 “类别/关键词”（Category/Keyword）这几个字段。名称和说明字段给出重要的具体提示，描述每件服装是什么。类别则是比较宽泛的识别标志。 

我不用图像数据来识别服装。这似乎与最寻常的服装分类思路背道而驰，但服装设计师其实不喜欢每年重复同样的款式，他们始终都在设法融合不同服装样式的视觉效果（希望能借此创造新潮流）：休闲衬衫酷似连帽衫，慢跑裤看起来像牛仔裤，皮夹克和海军呢大衣有几分相似，等等。 

文本数据通常才是识别服装类型的唯一标签。不过，我会用图像来确定一种风格和设计，这需要借助于Opencv。 

所有的数据在交给Deeplearning4j处理之前都已按性别和年龄组织完毕。

## 数据示例

      Title: Navy Pink Floral Silk Tie (海军粉色印花丝质领带)
      Description: Every T. M.Lewin tie is made from the finest quality 100% pure silk and hand-finished to perfection, with wool interlining.Our classic ties are available in a range of different colors and patterns.Approx. Width at Widest Point: 8.5cm Approx. Length: 150cm 100% Silk 100% Wool Interlining Dry Clean Only Catalogue Number - 49089 Approx. Width at Widest Point: 8.5cm; Approx. Length: 150cm; 100% Silk; 100% Wool Interlining; Dry Clean Only; Catalogue Number - 49089 (每条T. M. Lewin领带都采用最优质的100%纯真丝，由手工精制而成，衬里为羊毛。我们的经典领带系列提供各种不同的颜色和图案。最宽处宽度约为：8.5cm 长度约为：150cm 100%真丝 100%羊毛衬里 只可干洗 货号－49089 最宽处宽度约为：8.5cm；长度约为：150cm；100%真丝；100%羊毛衬里；只可干洗；货号－49089)
      Categories: Woven Silk Ties (针织丝质领带)

第一步是从数据中剔除品牌名称。有些品牌名称可能包含服装名称，例如：Armani Jeans（阿玛尼牛仔）。另一些品牌名称在数据库中出现频率特别高，例如Levi’s，可能会导致网络只将Levi’s的牛仔裤识别为牛仔裤。

所有单词全部小写。我不知道大小写会不会有影响，但小写感觉比较合理。我会删去所有的数字和标点，确保处理的数据中只有单词。

我还会删去停用词，包括常见的停用词以及服装行业经常出现的一些词，例如Mens（男士的）、fashion（时尚）、style（风格）、clothes（衣服）、colors（颜色）等。

向量化之前的数据：

      Title: navy pink floral silk tie
      Description: tie made finest quality pure silk hand-finished perfection wool interlining classic ties available range different approx width widest point cm approx length cm silk wool interlining dry clean only catalogue number approx width widest point cm approx length silk wool interlining dry clean only catalogue number
      Categories: woven silk ties

我用[*word2vec*](./cn/zh-word2vec.html)来生成向量。一个词至少需要出现10次，每个词集（wordset）得到40个向量。随后将每种名称、说明和类别对应的所有向量相加，得到120个向量。这从理论角度看来可能不像是个好主意。我曾一度反对这种思路，因为我估计最后得出的向量可能会过于分散，无法起到任何作用。但是实际应用的效果非常好。

## 数据加工管道

数据加工管道方面需要一些特殊的操作。数据库中有许多牛仔裤和T恤衫，而晚礼服和腰封则很少——考虑到市场规模上的差异，这其实也是预料之中的结果。但是为了保证定型正常进行，所有数据仍然需要分布均匀。

数据库里共有84种服装类型。我的方法是，对于一种服装类型，数据集里的每84条数据中必定随机出现一件与之相匹配的服装。我会打乱各种服装在84条数据中出现的顺序。如此一来就得到了分布均匀的数据集，但也有可能会导致腰封这类十分少见的服装发生过拟合。不过我后来发现这只是一种理论上的风险，实际应用的效果很棒。 

我的数据集共有84000条数据。构建整个数据集的过程就是将上述每84条数据随机出现一次的方法重复1000遍。

数据集完成后，再用一个标量对其进行标准化。这种方法的一个问题是每次都需要重新设定标量，因为随机因数会使每次定型所用的数据集发生变化。 

## 定型

我决定要定型多个特化的小型神经网络，而不是建立一个可以对任何数据进行分类的大型网络。 

所以我为每家服装店定型一个网络。每家店都用不同的方式生成自己的数据源，也因此各有各的模式，要针对这些模式来定型网络是很容易的。由于数据的差异特别小，很快就能建立一个非常准确的神经网络。除了针对各家服装店的网络，我也为每种语言专门建立了一个网络；这些网络的准确率不如服装店专用网络，和我预想的一样。针对各种语言的网络有着不同的目的。 

•	我用服装店专用网络来对新服装的数据进行分类。分类后，服装信息可以立即在应用程序上发布。
•	我用语言专用网络来对新服装店的数据进行分类，然后很快就能得到为新服装店建立专用网络所需的数据集。这种时候我本来就要检查每一件服装，网络准确率低一些也没关系。整个流程的速度已经大大加快了。

以下是服装店专用网络当前的设置：

        int outputNum = sergeData.labelSize;
        int numSamples = sergeData.labelSize * 10 * 100;
        int batchSize = sergeData.labelSize * 5;
        int iterations = 500;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = 500;

    MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .batchSize(batchSize)
                .learningRate(1e-6)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-6)
                .regularization(true)
                .l2(1e-4)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(120)
                        .nOut(80)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.NESTEROVS)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(80)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

有几点需要指出：

我采用的是可变的标签和批次大小。不同的服装店通常会有不同的产品类别。但由于这种差异比较小，单一网络设计可以得到很好的效果。

一个微批次（minibatch）的大小是所有记录的0.5%。网络经过定型产生一个准确的模型大约需要200个周期（epoch）。

我之前曾经用过9个向量的输入，但效果不太理想。后来我增加到120个向量，并且将节点数量控制在特征数量与标签数量之间，于是学习效果有了大幅改善。

虽然有些地方需要进一步完善，但目前的效果还是相当不错的。

**效果**：

* 每个服装店专用网络的定型需要2小时。
* 准确率为95～98%
* F1值为0.93～0.96

**硬件配置**：

* MacBook pro 2013
* CPU：2.4 GHz Intel Core i7
* 内存：8 GB 1600 MHz DDR3

*（编者注：一些匿名的大型电商网站均表示，基于DL4J的推荐系统让它们的广告覆盖率提升了200%。）*

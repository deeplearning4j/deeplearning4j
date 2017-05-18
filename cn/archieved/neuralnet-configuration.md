---
title: NeuralNetConfiguration类
layout: cn-default
---

# NeuralNetConfiguration类
*DL4J神经网络构建器基础*

DL4J中的所有神经网络均以NeuralNetConfiguration构建器为基础创建。该对象极其灵活，无论您需要实现什么类型的神经网络层，几乎都可以通过它来构建。这个类的参数组合及配置可用于设定不同类型的层，包括受限玻尔兹曼机（RBM）、深度置信网络（DBN）、卷积神经网络（CNN）、自动编码器等。各项参数及其默认设置的说明如下： 

开始构建单个层的Java类的方法：

	NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()

您可以用以下方式为这个类追加参数：

	new NeuralNetConfiguration.Builder().iterations(100).layer(new RBM()).nIn(784).nOut(10)

参数：

- **activationFunction**：*string*，每个隐藏层节点的激活函数，
	- 默认 = "sigmoid"
	- 选项：
		- "abs"
		- "acos"
		- "asin"
		- "atan"
		- "ceil"
		- "cos"
		- "exp"
		- "floor"
		- "hardtanh"
		- "identity"
		- "maxout"
		- "negative"
		- "pow"
		- "relu"
		- "round"
		- "sigmoid"
		- "sign"
		- "sin"
		- "softmax"
		- "sqrt"
		- "stabilize"
		- "tahn"
		- 用nd4j.getExecutioner创建自定义函数
- **applySparsity**：*boolean*，有活动的二进制隐藏单元时使用
	- 默认 = false
- **batch**：*int*，输入神经网络的数据量
	- 默认 = 0
- **constrainGradientToUnitNorm**：*boolean*，帮助梯度收敛，让损失变得更小、更平滑（避免梯度膨胀）
	- 默认 = false
- **convolutionType**：*ConvolutionLayer.ConvolutionType class*，卷积层类型
	- 默认 = ConvolutionLayer.ConvolutionType.MAX
- **corruptionLevel**：*double*，对输入数据进行污染的程度
	- 默认 = 0.3
- **dist**：*Distribution class*，权重初始化所用的分布
	- 默认 = new NormalDistribution(1e-3,1)
	- 选项：
		- NormalDistribution
		- UniformDistribution
		- BinomialDistribution
- **dropOut**：*double*，随机丢弃一定数量的活动单元/节点，将其置零（不激活）
	- 默认 = 0
- **featureMapSize**：*int[]*，卷积内核大小（也称为接受场）
	- 默认 = {2,2}
- **filterSize**：*int[]*，为降采样层创建张量数据结构 = 特征映射图数量（深度切片数量） x 通道数量 x 特征映射图空间（输入数据矩阵的行与列）
	- 默认 = {2,2,2,2}
	- 例如：5, 1, numRows, numColumns
	- 行 = 批次或总体的数据样本量；列 = 每个数据样本的特征数量
- **hiddenUnit**：*RBM.HiddenUnit*，RBM隐藏单元/节点类型
	- 默认 = RBM.HiddenUnit.BINARY
- **inputPreProcessor**：(*int*, *class*) {层数, 数据处理器类} 对层的输入数据进行转换/预处理
	- 例如：.inputPreProcessor(0,new ConvolutionInputPreProcessor(numRows,numColumns))
	- 将2维张量转换为4维
	- 行 = 批次；列 = 输入数据点的数量
- **iterations**：*int*，定型迭代次数
- **k**：*int*，RBM各层的预定型中，用马尔可夫链进行预测的对比散度算法的运行次数
	- 默认 = 1
- **kernel**：*int[]*，内核大小（用于卷积网络）
	- 默认 = 5
- **l1**：*double*，L1正则化
	- 默认 = 0.0
- **l2**：*double*，L2正则化
	- 默认 = 0.0
- **layer**：*Layer class*，设定层的结构
- **lossFunction**：*LossFunctions class*，作用于网络输出的误差变换函数
	- 默认 = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY
	- 选项：
		- MSE
		- EXPLL
		- XENT
		- MCXENT
		- RMSE_XENT
		- SQUARED_LOSS
		- RECONSTRUCTION_CROSSENTROPY
		- NEGATIVELOGLIKELIHOOD
- **learningRate**：*Double*，步幅，亦即在搜索空间中移动时改变参数向量的速度（学习速率越大，得到最终结果的速度越快，但有可能错过最佳值；速率较小，所需的定型时间可能会大幅增加）
优化函数的变化速率
	- 默认 = 1e-1f
- **minimize**：*boolean*，设定目标是最小化还是最大化
	- 默认 = false
- **momentum**：*double*，动量，用于减少权重变化的波动
	- 默认 = 0.5
- **momentumAfter**：*Map[Integer, Double]* （n次迭代，动量）n次迭代之后的动量
- **nIn**：*int*，输入数据点的数量
- **nOut**：*int*，输出节点数量
- **numIterations**：*int*，网络定型的迭代次数
	- 默认 = 1000
- **numLineSearchIterations**：*int*
	- 默认 = 100
- **optimizationAlgo**：*OptimizationAlgorithm class*，反向传播算法
	- 默认 = OptimizationAlgorithm.CONJUGATE_GRADIENT
	- 选项：
		- GRADIENT_DESCENT
		- CONJUGATE_GRADIENT
		- HESSIAN_FREE
		- LBFGS
		- ITERATION_GRADIENT_DESCENT
- **preProcessor**：(*int*, *class*) {层数, 数据处理器类} 对层的输出数据进行转换/预处理
	- 例如：.preProcessor(0, new ConvolutionPostProcessor())
- **renderWeightsEveryNumEpochs**：*int*，每过几个epoch显示权重，默认 = -1
- **resetAdaGradIterations**：*int*，在n次迭代之后重置AdaGrad历史梯度
	- 默认 = -1
- **rng**：*Random class*，用一个随机种子来确保定型的初始权重保持一致
	- 默认 = new DefaultRandom()
	- 示例 = .rng(new DefaultRandom(3))
- **stride**：*int[]*，降采样类层的大小
	- 默认 = {2,2}
- **sparsity**：*double*
	- 默认 = 0
- **stepFunction**：*StepFunction class*，算法在学习过程中的权重调整幅度
	- 默认 = new GradientStepFunction()
- **useAdaGrad**：*boolean*，在反向传播算法中采用AdaGrad学习速率适应
	- 默认 = true
- **useRegularization**：*boolean*，采用正则化
	- 默认 = false
- **variables**：*List[String]*，梯度的键，确保能够有序地获取和设定梯度
	 - 默认 = new ArrayList<>()
- **visibleUnit**：*RBM.VisibleUnit*，RBM可见单元/节点的类型，默认 = RBM.VisibleUnit.BINARY
- **weightInit**：*WeightInit class*，权重初始化方式
	- 默认 = WeightInit.VI
	- 选项：
		- WeightInit.DISTRIBUTION：用基于输入数据形状的分布来初始化权重
		- WeightInit.NORMALIZED：用正态分布来初始化权重
		- WeightInit.SIZE：用受限均匀分布来初始化权重，由数据形状决定最小值和最大值
		- WeightInit.UNIFORM：用受限均匀分布来初始化权重（指定最小值和最大值）
		- WeightInit.VI：用归一化方差来初始化权重（Glorot）
		- WeightInit.ZERO：初始权重设为零
- **weightShape**：*int[]*

该类的更多详情参见[JavaDoc](http://deeplearning4j.org/doc/)。

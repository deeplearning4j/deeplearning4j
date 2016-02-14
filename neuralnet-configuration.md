---
title: NeuralNetConfiguration Class
layout: default
---

# NeuralNetConfiguration Class:
*DL4J Neural Net Builder Basics*

For almost any neural net you build in DL4J the foundation is the NeuralNetConfiguration constructor. This object provides significant flexibility on building out almost any type of neural network layer you want to implement. Parameter combinations and configurations for this class define different types of layers such as RBMs, DBNs, CNNs, Auto, etc. Below are a list of parameters with default settings: 

How to start constructing the class in Java for single layer:

	NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()

Append parameters onto this class by linking them up as follows:

	new NeuralNetConfiguration.Builder().iterations(100).layer(new RBM()).nIn(784).nOut(10)

Parameters:

- **activationFunction**: *string*, activation function on each hidden layer node,
	- default="sigmoid"
	- Options:
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
		- create customized functions with nd4j.getExecutioner
- **applySparsity**: *boolean*, use when binary hidden nets are active
	- default = false
- **batch**: *int*, amount of data to input into the neural net
	- default = 0
- **constrainGradientToUnitNorm**: *boolean*, helps gradient converge and makes loss smaller and smoother (prevents exploding gradients)
	- default = false
- **convolutionType**: *ConvolutionLayer.ConvolutionType class*, convolution layer type
	- default = ConvolutionLayer.ConvolutionType.MAX
- **corruptionLevel**: *double*, how much to corrupt the input data
	- default = 0.3
- **dist**: *Distribution class*, distribution to use for weight initialization
	- default = new NormalDistribution(1e-3,1)
	- Options:
		- NormalDistribution
		- UniformDistribution
		- BinomialDistribution
- **dropOut**: *double*, randomly drop a certain amount of active units/nodes to zero (no activation)
	- default = 0
- **featureMapSize**: *int[]*, kernel convolution size (also refered to as receptive field)
	- default = {2,2}
- **filterSize**: *int[]* ,creates tensor data structure for subsampling layers = number of feature maps (number of depth slices) x number of channels x feature map space (rows & cols of input data matrix)
	- default = {2,2,2,2}
	- ex: 5, 1, numRows, numColumns
	- rows = batch or total data samples & columns = number of features per data sample
- **hiddenUnit**: *RBM.HiddenUnit*, type of RBM hidden units/nodes
	- default = RBM.HiddenUnit.BINARY
- **inputPreProcessor**: (*int*, *class*) {layer number, data processor class} transform/process input data shape to layer
	- ex: .inputPreProcessor(0,new ConvolutionInputPreProcessor(numRows,numColumns))
	- transform 2d to 4d tensor
	- rows = batch & columns = number of data points passed in
- **iterations**: *int*, num training iteractions
- **k**: *int*, number steps of a Markov chain to compute a guess as part of the contrastive divergence in RBM layerwise pre-training
	- default = 1
- **kernel**: *int[]* size of kernel (used in convolutions)
	- default = 5
- **l1**: *double*, L1 regularization
	- default = 0.0
- **l2**: *double*, L2 regularization
	- default = 0.0
- **layer**: *Layer class*, sets layer structure
- **lossFunction**: *LossFunctions class*, error transformation function on net output
	- default = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY
	- Options:
		- MSE
		- EXPLL
		- XENT
		- MCXENT
		- RMSE_XENT
		- SQUARED_LOSS
		- RECONSTRUCTION_CROSSENTROPY
		- NEGATIVELOGLIKELIHOOD
- **learningRate**: *double*, step size aka how fast we change the parameter vector as we move through search space (larger can drive faster to goal but overshoot and smaller can lead to a lot longer training times to hit goal)
rate of change in optimization function
	- default = 1e-1f
- **minimize**: *boolean*, setting objective to minimize or maximize
	- default = false
- **momentum**: *double*, diminsh the fluctuations of weight changes by ch
	- default = 0.5
- **momentumAfter**: *Map[Integer, Double]* (n iterations, momentum), momentum after n iterations
- **nIn**: *int*, number of input data points
- **nOut**: *int*, number of output nodes
- **numIterations**: *int* number of iterations to train the net
	- default = 1000
- **numLineSearchIterations**: *int*
	- default = 100
- **optimizationAlgo**: *OptimizationAlgorithm class*, backprop method
	- default = OptimizationAlgorithm.CONJUGATE_GRADIENT
	- Options:
		- GRADIENT_DESCENT
		- CONJUGATE_GRADIENT
		- HESSIAN_FREE
		- LBFGS
		- ITERATION_GRADIENT_DESCENT
- **preProcessor**: (*int*, *class*) {layer number, data processor class}, transform/process output data shape from layer
	- ex1: .preProcessor(0, new ConvolutionPostProcessor())
- **renderWeightsEveryNumEpochs**: *int*, default = -1
- **resetAdaGradIterations**: *int*, reset AdaGrad historical gradient after n iteractions
	- default = -1
- **rng**: *Random class*, applies seed to ensure the same initial weights are used when training
	- default = new DefaultRandom()
	- example = .rng(new DefaultRandom(3))
- **stride**: *int[]*, size for subsampling type layers
	- default = {2,2}
- **sparsity**: *double*
	- default = 0
- **stepFunction**: *StepFunction class*, how much an algorithm adjusts the weights as it learns
	- default = new GradientStepFunction()
- **useAdaGrad**: *boolean*, applies AdaGrad learning rate adaption in backprop method
	- default=true
- **useRegularization**: *boolean*, applies regularization to net
	- default=false
- **variables**: *List[String]*, gradient keys used for ensuring order when getting and setting gardient
	 - default = new ArrayList<>()
- **visibleUnit**: *RBM.VisibleUnit*, type of RBM visible units/nodes, default = RBM.VisibleUnit.BINARY
- **weightInit**: *WeightInit class*, how to initialize the weights
	- default = WeightInit.VI
	- Options:
		- WeightInit.DISTRIBUTION: Sample weights from a distribution based on shape of input
		- WeightInit.NORMALIZED: Sample weights from normalized distribution
		- WeightInit.SIZE:Sample weights from bound uniform distribution using shape for min and max
		- WeightInit.UNIFORM: Sample weights from bound uniform distribution (specify min and max)
		- WeightInit.VI: Sample weights from variance normalized initialization (Glorot)
		- WeightInit.ZERO: sGenerate weights as zeros
- **weightShape**: *int[]*

For more information on this class, checkout [Javadocs](http://deeplearning4j.org/doc/).

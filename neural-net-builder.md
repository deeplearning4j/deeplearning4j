---
title: 
layout: default
---

# NeuralNetConfiguration Class:
## *DL4J Neural Net Builder Basics*

For almost any neural net you build in DL4J the foundation is the NeuralNetConfiguration constructor. This object provides significant flexibility on building out almost any type of neural network structure you want to implement. Parameter combinations and configurations for this class define different types of neural nets such as RBMs or Convolutional to name a few. Below provides a brief overview of getting started with this class and key configuration parameters (aka attributes). 

How to start constructing the class in Java:

	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

Append parameters onto this class by linking them up as follows:

	new NeuralNetConfiguration.Builder().iterations(100).layer(new RBM()).nIn(784).nOut(10)

Parameters:

- **activationFunction**: *string*, activation function on each hidden layer node,
	- default="sigmoid"
	- Options:
		-"sigmoid"
		- "tahn"
		- "reclu"
		- create customized functions with nd4j.getExecutioner
- **applySparsity**: *boolean*, use when binary hidden nets are active
	- default = false
- **batch****: *int*, amount of data to input into the neural net
	- default = 0
- **constrainGradientToUnitNorm**: *boolean*, helps gradient converge and makes loss smaller and smoother
	- default = false
- **convolutionType**: *ConvolutionDownSampleLayer.ConvolutionType class*, convolution layer type
	- default = ConvolutionDownSampleLayer.ConvolutionType.MAX
- **corruptionLevel**: *double*, how much to corrupt the input data
	- default = 0.3
- **dist**: *Distribution class*, distribution to use for weight initialization
	- default = new NormalDistribution(1e-3,1)
- **dropOut**: *double*, randomly drop a certain amount of active units/nodes to zero (no activation)
	- default = 0
- **featureMapSize**: *int[]*, size of feature space sample - similar to kernal
	- default = {2,2}
- **filterSize**: *int[]* ,creates tensor data structure for layers = feature maps x number of channels x  feature map space (rows & cols of input data matrix)
	- default = {2,2,2,2}
	- ex: 5, 1, numRows, numColumns
	- rows = batch or total data samples & columns = number of features per data sample
- **hiddenLayerSizes**: *int[]*, number of nodes in the layers
	- one layer format = new int[]{32} = initiate array of ints with 32 nodes (spaces)
	- multiple layers format = new int[]{32,20,40,32} = layer 1 is 32 nodes, layer 2 is 20 nodes, etc
- **hiddenUnit**: *RBM.HiddenUnit*, type of RBM hidden units/nodes
	- default = RBM.HiddenUnit.BINARY
- **inputPreProcessor**: (*int*, *class*) {layer number, data processor class} transform/process input data shape to layer
	- ex: .inputPreProcessor(0,new ConvolutionInputPreProcessor(numRows,numColumns))
	- transform 2d to 4d tensor
	- rows = batch & columns = number of data points passed in
- **iterations**: *int*, num training iteractions
- **k**: *int*, number steps of a Markov chain to compute a guess as part of the contrastive divergence in RBM layerwise pre-training
	- default = 1
- **kernal**: *int[]* size of kernal (used in convolutions)
	- default = 5
- **l1**: *double*, L1 regularization
	- default = 0.0
- **l2**: *double*, L2 regularization
	- default = 0.0
- **layer**: *Layer class*, sets layer structure
- **list**: *int*, number of layers (does not count input layer)
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
- **lr**: *double*
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
- **override**: (*int*, *class*) {layer number, data processor class}, override with specific layer configuation
- **preProcessor**: (*int*, *class*) {layer number, data processor class}, transform/process output data shape from layer
	- ex1: .preProcessor(0, new ConvolutionPostProcessor())
- **renderWeightsEveryNumEpochs**: *int*, default = -1
- **resetAdaGradIterations**: *int*, reset AdaGrad historical gradient after n iteractions
	- default = -1
- **rng**: *Random class*, used for sampling
	- default = new DefaultRandom()
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
		- WeightInit.VI: variance normalized initialization (Glorot)
		- WeightInit.ZERO: straight zeros
		- WeightInit.SIZE:
		- WeightInit.DISTRIBUTION: sample weights from distribution
		- WeightInit.NORMALIZED:
		- WeightInit.UNIFORM:
- **weightShape**: *int[]*

For more information on this class, checkout [Javadocs](http://deeplearning4j.org/doc/).
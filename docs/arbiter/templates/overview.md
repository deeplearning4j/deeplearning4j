---
title: Arbiter Overview
short_title: Overview
description: Introduction to using Arbiter for hyperparameter optimization.
category: Arbiter
weight: 0
---

## Hyperparameter Optimization

Machine learning techniques have a set of parameters that have to be chosen before any training can begin. These parameters are referred to as hyperparameters. Some examples of hyperparameters are ‘k’ in k-nearest-neighbors and the regularization parameter in Support Vector Machines. Neural Networks, in particular, have a wide variety of hyperparameters. Some of these define the architecture of the neural network like the number of layers and their size. Other define the learning process like the learning rate and regularization. 

Traditionally these choices are made based on existing rules of thumb or after extensive trial and error, both of which are less than ideal. Undoubtedly the choice of these parameters can have a significant impact on the results obtained after learning. Hyperparameter optimization attempts to automate this process using software that applies search strategies. 

## Arbiter

Arbiter is part of the DL4J Suite of Machine Learning/Deep Learning tools for the enterprise. It is dedicated to the hyperparameter optimization of neural networks created or imported into dl4j. It allows users to set up search spaces for the hyperparameters and run either grid search or random search to select the best configuration based on a given scoring metric. 

When to use Arbiter?
Arbiter can be used to find good performing models, potentially saving you time tuning your model's hyperparameters, at the expense of greater computational time. Note however that Arbiter doesn't completely automate the neural network tuning process, the user still needs to specify a search space. This search space defines the range of valid values for each hyperparameter (example: minimum and maximum allowable learning rate). If this search space is chosen poorly, Arbiter may not be able to find any good models.

Add the following to your pom.xml to include Arbiter in your project where ${arbiter.version} is the latest release of the dl4j stack.

```xml
<!-- Arbiter - used for hyperparameter optimization (grid/random search) -->
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>arbiter-deeplearning4j</artifactId>
    <version>{{page.version}}</version>
</dependency>
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>arbiter-ui_2.11</artifactId>
    <version>{{page.version}}</version>
</dependency>
```

Arbiter also comes with a handy UI that helps visualize the results from the optimizations runs. 

As a prerequisite to using Arbiter users should be familiar with the NeuralNetworkConfiguration, MultilayerNetworkConfiguration and ComputationGraphconfiguration classes in DL4J.

## Usage
This section will provide an overview of the important constructs necessary to use Arbiter. The sections that follow will dive into the details. 

At the highest level, setting up hyperparameter optimization involves setting up an OptimizationConfiguration and running it via IOptimizationRunner. 

Below is some code that demonstrates the fluent builder pattern in OptimizationConfiguration:

```java
OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
    .candidateGenerator(candidateGenerator)
    .dataSource(dataSourceClass,dataSourceProperties)
    .modelSaver(modelSaver)
    .scoreFunction(scoreFunction)
    .terminationConditions(terminationConditions)
    .build();
```

As indicated above setting up an optimization configuration requires:
CandidateGenerator: Proposes candidates (i.e., hyperparameter configurations) for evaluation. Candidates are generated based on some strategy. Currently random search and grid search are supported. Valid configurations for the candidates are determined by the hyperparameter space associated with the candidate generator.
DataSource: DataSource is used under the hood to provide data to the generated candidates for training and test
ModelSaver: Specifies how the results of each hyperparameter optimization run should be saved. For example, whether saving should be done to local disk, to a database, to HDFS, or simply stored in memory.
ScoreFunction: A metric that is a single number that we are seeking to minimize or maximize to determine the best candidate. Eg. Model loss or classification accuracy
TerminationCondition:  Determines when hyperparameter optimization should be stopped. Eg. A given number of candidates have been evaluated, a certain amount of computation time has passed.

The optimization configuration is then passed to an optimization runner along with a task creator. 

If candidates generated are MultiLayerNetworks this is set up as follows:

```java        
IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());
```

Alternatively if candidates generated are ComputationGraphs this is set up as follows:

```java        
IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator());
```

Currently the only option available for the runner is the LocalOptimizationRunner which is used to execute learning on a single machine (i.e, in the current JVM). In principle, other execution methods (for example, on Spark or cloud computing machines) could be implemented.

To summarize here are the steps to set up a hyperparameter optimization run:

1. Specify hyperparameter search space 
1. Specify a candidate generator for the hyperparameter search space 
1. The next section of steps can be done in any order:
1. Specify a data source
1. Specify a model saver
1. Specify a score function
1. Specify a termination condition
1. The next steps have to be done in order:
1. Use 2 to 6 above to construct an Optimization Configuration
1. Run with the Optimization Runner.


## Hyperparameter search space 

Arbiter’s `ParameterSpace<T>` class defines the acceptable ranges of values a given hyperparameter may take. ParameterSpace can be a simple, like a ParameterSpace that defines a continuous range of double values (say for learning rate) or complicated with multiple nested parameter spaces within like the case of a MultiLayerSpace (which defines a search space for a MultilayerConfiguration).


## MultiLayerSpace and ComputationGraphSpace

MultiLayerSpace and ComputationGraphSpace are Arbiter’s counterpart to dl4j’s MultiLayerConfiguration and ComputationGraphConfiguration. They are used to set up parameter spaces for valid hyperparameters in MultiLayerConfiguration and ComputationGraphConfiguration. 

In addition to these users can also set up the number of epochs or an early stopping configuration to indicate when training on each candidate neural net should stop. If both an EarlyStoppingConfiguration and the number of epochs are specified, early stopping will be used in preference.

Setting up MultiLayerSpace or ComputationGraphSpace are fairly straightforward once the user is familiar with Integer, Continuous and Discrete parameter spaces and LayerSpaces and UpdaterSpaces. 

The only caveat to be noted here is that while it is possible to set up weightConstraints, l1Bias and l2Bias as part of the NeuralNetConfiguration these have to be setup on a per layer/layerSpace basis in MultiLayerSpace. In general all properties/hyperparameters available through the builder will take either a fixed value or a parameter space of that type. This means that pretty much every aspect of the MultiLayerConfiguration can be swept to test out a variety of architectures and initial values.

Here is a simple example of a MultiLayerSpace:

```java
ParameterSpace<Boolean> biasSpace = new DiscreteParameterSpace<>(new Boolean[]{true, false});
ParameterSpace<Integer> firstLayerSize = new IntegerParameterSpace(10,30);
ParameterSpace<Integer> secondLayerSize = new MathOp<>(firstLayerSize, Op.MUL, 3);
ParameterSpace<Double> firstLayerLR = new ContinuousParameterSpace(0.01, 0.1);
ParameterSpace<Double> secondLayerLR = new MathOp<>(firstLayerLR, Op.ADD, 0.2);

MultiLayerSpace mls =
    new MultiLayerSpace.Builder().seed(12345)
            .hasBias(biasSpace)
            .layer(new DenseLayerSpace.Builder().nOut(firstLayerSize)
                    .updater(new AdamSpace(firstLayerLR))
                    .build())
            .layer(new OutputLayerSpace.Builder().nOut(secondLayerSize)
                    .updater(new AdamSpace(secondLayerLR))
                    .build())
            .setInputType(InputType.feedForward(10))
  .numEpochs(20).build(); //Data will be fit for a fixed number of epochs
```

Of particular note is Arbiter’s ability to vary the number of layers in the MultiLayerSpace. Here is a simple example demonstrating the same that also demonstrates setting up a parameter search space for a weighted loss function:

```java
ILossFunction[] weightedLossFns = new ILossFunction[]{
    new LossMCXENT(Nd4j.create(new double[]{1, 0.1})),
        new LossMCXENT(Nd4j.create(new double[]{1, 0.05})),
            new LossMCXENT(Nd4j.create(new double[]{1, 0.01}))};

DiscreteParameterSpace<ILossFunction> weightLossFn = new DiscreteParameterSpace<>(weightedLossFns);
MultiLayerSpace mls =
    new MultiLayerSpace.Builder().seed(12345)
        .addLayer(new DenseLayerSpace.Builder().nIn(10).nOut(10).build(),
            new IntegerParameterSpace(2, 5)) //2 to 5 identical layers
        .addLayer(new OutputLayerSpace.Builder()
            .iLossFunction(weightLossFn)
            .nIn(10).nOut(2).build())
        .backprop(true).pretrain(false).build();
```

The two to five layers created above will be identical (stacked). Currently Arbiter does not support the ability to create independent layers. 

Finally it is also possible to create a fixed number of identical layers as shown in the following example:

```java
DiscreteParameterSpace<Activation> activationSpace = new DiscreteParameterSpace(new Activation[]{Activation.IDENTITY, Activation.ELU, Activation.RELU});
MultiLayerSpace mls = new MultiLayerSpace.Builder().updater(new Sgd(0.005))
    .addLayer(new DenseLayerSpace.Builder().activation(activationSpace).nIn(10).nOut(10).build(),
        new FixedValue<Integer>(3))
    .addLayer(new OutputLayerSpace.Builder().iLossFunction(new LossMCXENT()).nIn(10).nOut(2).build())
    .backprop(true).build();
```

In this example with a grid search three separate architectures will be created. They will be identical in every way but in the chosen activation function in the non-output layers. Again it is to be noted that the layers created in each architecture are identical(stacked).

Creating ComputationGraphSpace is very similar to MultiLayerSpace. However there is currently only support for fixed graph structures. 

Here is a simple example demonstrating setting up a ComputationGraphSpace:

```java
ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                .l2(new ContinuousParameterSpace(0.2, 0.5))
                .addInputs("in")
                .addLayer("0",new  DenseLayerSpace.Builder().nIn(10).nOut(10).activation(
            new DiscreteParameterSpace<>(Activation.RELU,Activation.TANH).build(),"in")           

        .addLayer("1", new OutputLayerSpace.Builder().nIn(10).nOut(10)
                             .activation(Activation.SOFTMAX).build(), "0")
        .setOutputs("1").setInputTypes(InputType.feedForward(10)).build();
```

### JSON serialization.

MultiLayerSpace, ComputationGraphSpace and OptimizationConfiguration have `toJso`n methods as well as `fromJson` methods. You can store the JSON representation for further use.

Specifying a candidate generator
As mentioned earlier Arbiter currently supports grid search and random search.

Setting up a random search is straightforward and is shown below:
MultiLayerSpace mls;
...
CandidateGenerator candidateGenerator = new RandomSearchGenerator(mls);

Setting up a grid search is also simple. With a grid search the user also gets to specify a discretization count and a mode. The discretization count determines how many values a continuous parameter is binned into. For eg. a continuous parameter in range [0,1] is converted to [0.0, 0.5, 1.0] with a discretizationCount of 3. The mode determines the manner in which the candidates are generated. Candidates can be generated in Sequential (in order) or RandomOrder. With sequential order the first hyperparameter will be changed most rapidly and consequently the last hyperparameter will be changed the least rapidly. Note that both modes will result in the same set of candidates just in varying order.

Here is a simple example of how a grid search is set up with a discretization count of 4 in sequential order:

```java
CandidateGenerator candidateGenerator = new GridSearchCandidateGenerator(mls, 4,
 GridSearchCandidateGenerator.Mode.Sequential);
```


## Specifying a data source

The DataSource interface defines where data for training the different candidates come from. It is very straightforward to implement. Note that a no argument constructor is required to be defined. Depending on the needs of the user the DataSource implementation can be configured with properties, like the size of the minibatch. A simple implementation of the data source that uses the MNIST dataset is available in the example repo which is covered later in this guide.
It is important to note here that the number of epochs (as well as early stopping configurations) can be set via the MultiLayerSpace and ComputationGraphSpace builders. 


## Specifying a model/result saver 

Arbiter currently supports saving models either saving to disk in local memory (FileModelSaver) or storing results in-memory (InMemoryResultSaver). InMemoryResultSaver is obviously not recommended for large models. 

Setting them up are trivial. FileModelSaver constructor takes a path as String. It saves config, parameters and score to: baseDir/0/, baseDir/1/, etc where index is given by OptimizationResult.getIndex(). InMemoryResultSaver requires no arguments.

Specifying a score function
There are three main classes for score functions: EvaluationScoreFunction, ROCScoreFunction and RegressionScoreFunction. 

EvaluationScoreFunction uses a DL4J evaluation metric. Available metrics are ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC. Here is a simple example that uses accuracy:
        ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);

ROCScoreFunction calculates AUC (area under ROC curve) or AUPRC (area under precision/recall curve) on the test set. Different ROC types (ROC, ROCBinary and ROCMultiClass) are supported. Here is a simple example that uses AUC:
ScoreFunction sf = new ROCScoreFunction(ROCScoreFunction.ROCType.BINARY, ROCScoreFunction.Metric.AUC));

RegressionScoreFunction is used for regression and supports all DL4J RegressionEvaluation metrics (MSE, MAE, RMSE, RSE, PC, R2). Here is a simple example:
ScoreFunction sf = new RegressionScoreFunction(RegressionEvaluation.Metric.MSE);

## Specifying a termination condition

Arbiter currently only supports two kinds of termination conditions - MaxTimeCondition and MaxCandidatesCondition. MaxTimeCondition specifies a time after which hyperparameter optimization will be terminated. MaxCandidatesCondition specifies a maximum number of candidates after which hyperparameter optimization is terminated. Termination conditions can be specified as a list. Hyperparameter optimization stops if any of the conditions are met. 

Here is a simple example where the run is terminated at fifteen minutes or after training ten candidates which ever is met first:

```java
TerminationCondition[] terminationConditions = { 
	new MaxTimeCondition(15, TimeUnit.MINUTES),
    new MaxCandidatesCondition(10)
};
```


## Example Arbiter Run on MNIST data

The DL4J example repo contains a BasicHyperparameterOptimizationExample on MNIST data. Users can walk through this simple example here. This example also goes through setting up the Arbiter UI. Arbiter uses the same storage and persistence approach as DL4J's UI. More documentation on the UI can be found here. The UI can be accessed at  http://localhost:9000/arbiter.


## Tips for hyperparameter tuning

Please refer to the excellent section on hyperparameter optimization here from the CS231N class at Stanford. A summary of these techniques are below:
- Prefer random search over grid search. For a comparison of random and grid search methods, see Random Search for Hyper-parameter Optimization (Bergstra and Bengio, 2012).
- Run search from coarse to fine (Start with a coarse parameter search with one or two epochs, pick the best candidate to do a fine search on with more epochs, iterate)
- Use LogUniformDistribution for certain hyperparameter like the learning rate, l2 etc
- Be mindful of values that fall close to the borders of the parameter search space



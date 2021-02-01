# Arbiter

A tool dedicated to tuning (hyperparameter optimization) of machine learning models. Part of the DL4J Suite of Machine Learning / Deep Learning tools for the enterprise.


## Modules
Arbiter contains the following modules:

- arbiter-core: Defines the API and core functionality, and also contains functionality for the Arbiter UI
- arbiter-deeplearning4j: For hyperparameter optimization of DL4J models (MultiLayerNetwork and ComputationGraph networks)


## Hyperparameter Optimization Functionality

The open-source version of Arbiter currently defines two methods of hyperparameter optimization:

- Grid search
- Random search

For optimization of complex models such as neural networks (those with more than a few hyperparameters), random search is superior to grid search, though Bayesian hyperparameter optimization schemes 
For a comparison of random and grid search methods, see [Random Search for Hyper-parameter Optimization (Bergstra and Bengio, 2012)](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

### Core Concepts and Classes in Arbiter for Hyperparameter Optimization

In order to conduct hyperparameter optimization in Arbiter, it is necessary for the user to understand and define the following:

- **Parameter Space**: A ```ParameterSpace<P>``` specifies the type and allowable values of hyperparameters for a model configuration of type ```P```. For example, ```P``` could be a MultiLayerConfiguration for DL4J
- **Candidate Generator**: A ```CandidateGenerator<C>``` is used to generate candidate models configurations of some type ```C```. The following implementations are defined in arbiter-core:
    - ```RandomSearchCandidateGenerator```
    - ```GridSearchCandidateGenerator```
- **Score Function**: A ```ScoreFunction<M,D>``` is used to score a model of type ```M``` given data of type ```D```. For example, in DL4J a score function might be used to calculate the classification accuracy from a DataSetIterator
    - A key concept here is that they score is a single numerical (double precision) value that we either want to minimize or maximize - this is the goal of hyperparameter optimization
- **Termination Conditions**: One or more ```TerminationCondition``` instances must be provided to the ```OptimizationConfiguration```. ```TerminationCondition``` instances are used to control when hyperparameter optimization should be stopped. Some built-in termination conditions:
    - ```MaxCandidatesCondition```: Terminate if more than the specified number of candidate hyperparameter configurations have been executed
    - ```MaxTimeCondition```: Terminate after a specified amount of time has elapsed since starting the optimization
- **Result Saver**: The ```ResultSaver<C,M,A>``` interface is used to specify how the results of each hyperparameter optimization run should be saved. For example, whether saving should be done to local disk, to a database, to HDFS, or simply stored in memory.
    - Note that ```ResultSaver.saveModel``` method returns a ```ResultReference``` object, which provides a mechanism for re-loading both the model and score from wherever it may be saved.
- **Optimization Configuration**: An ```OptimizationConfiguration<C,M,D,A>``` ties together the above configuration options in a fluent (builder) pattern.
- **Candidate Executor**: The ```CandidateExecutor<C,M,D,A>``` interface provides a layer of abstraction between the configuration and execution of each instance of learning. Currently, the only option is the ```LocalCandidateExecutor```, which is used to execute learning on a single machine (in the current JVM). In principle, other execution methods (for example, on Spark or cloud computing machines) could be implemented.
- **Optimization Runner**: The ```OptimizationRunner``` uses an ```OptimizationConfiguration``` and a ```CandidateExecutor``` to actually run the optimization, and save the results.


### Optimization of DeepLearning4J Models

(This section: forthcoming)

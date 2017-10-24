---
title: DeepLearning4j Deployment
layout: default
---

# DeepLearning4j: AI Model Deployment

This section is about deploying deep learning models. We will cover using Apache Spark to deploy neural networks, how to save and load previously trained neural networks, and then go over other uses of neural networks such as regression and building more complex models.

- [**Apache Spark**](#spark) 
- [**Saving & Loading Neural Networks**](#saving) 
- [**Regression**](#regression)
- [**More Complex Architectures**](#complex)

## <a name="spark">Spark</a>

[Apache Spark](https://spark.apache.org/) is a general cluster-computing framework, which was originally developed at the AMP lab at UC Berkeley. Spark is helpful for deep learning because training neural networks is often a computationally heavy task. Spark can be used intelligently to parallelize training and accelerate the process. DL4J integrates tightly with Spark for distributed training of neural networks. To learn about running Spark jobs, look [here](https://spark.apache.org/docs/latest/quick-start.html).

The basic notion of how Spark is used to train neural networks is simple. First, the Spark driver (or master) starts with a configuration of the neural network and parameters. The training data is then split up into subsets, which are distributed to the worker nodes along with the neural network configuration and parameters. Thus, each worker node will have a different split of the data. The worker nodes then update the parameters based on their split of the data. The neural networks will now have different parameters across worker nodes. The parameters are then averaged across the workers and the resulting neural network is then sent back to the master node. The process will then repeat for the next data. 

To use Spark with Dl4J, one of the following classes should be used: SparkDl4jMultiLayer or SparkComputationGraph. As their names imply, they rely on the MultiLayerNetwork and ComputationGraph classes of Dl4J.

### Example

We will go over a basic outline of how to use Spark with DL4J. We first will want to start a Spark context.

```
SparkConf sparkConf = new SparkConf();	
JavaSparkContext sc = new JavaSparkContext(sparkConf);
```
We must then get the data into a format neural networks understand, which are typically some sort of DataSetIterators. These steps are similar to what we have seen before so we assume that trainData is a DataSetIterator containing the training split of the data. For a Spark application, we must then convert the data into a JavaRDD<DataSet>. A RDD is a resilient distributed dataset, which provides an interface to partition data across a cluster. In this example, we do this by creating a ArrayList of DataSets and then convert it into a JavaRDD<DataSet>. 

```
List<DataSet> trainDataList = new ArrayList<>();

while (trainData.hasNext()) {
    trainDataList.add(trainData.next());
}

JavaRDD<DataSet> JtrainData = sc.parallelize(trainDataList);
```

Once the data is ready, we can then set up the configuration of the neural network. Note that this is the same process as before; we can either use MultiLayerNetwork or ComputationGraph. In this example, we assume  ComputationGraphConfiguration conf contains the configuration for the neural network.

Before intiializing a SparkComputationGraph we must outline the process of parallel training. To use the ParameterAveraging method, we can configure it using a TrainingMaster. The dataSetObjectSize specifies how many observations are contained in a DataSet object. This will depend on how the DataSetIterator was set up in this example. The batchSizePerWorker controls the minibatch size used for parameter updates for each worker and controls how frequently parameters are averaged and redistributed in terms of the minibatch sizes of a worker. Lastly, workerPrefetchNumBatches specifies how many minibatches Spark workers should prefetch to avoid waiting for data to be loaded. A larger workerPrefetchNumBatches parameter will requires more memory. 

```
TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(int dataSetObjectSize)
    .averagingFrequency(5)
    .workerPrefetchNumBatches(2)
    .batchSizePerWorker(BATCH_SIZE)
    .build();
```

We can now finally create a Spark network using the Spark context, neural network configuration, and training master configuration.  Training the neural network is easy. You just need to call the fit function and the SparkComputationGraph will handle Spark training in the background.

```
SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf, tm);
sparkNet.fit(JtrainData);
```
       
### Submitting a Spark job

Once the program is ready, you will need to complete a couple of steps to submit the job to a Spark cluster. First, you will need to create a JAR file of the project the java file is in. Using IntelliJ, the way to do this is as follows: File > Project Structure > Artifacts > green plus sign near top of window > JAR > From module with dependencies > select the main class > extract to the target JAR > Build on make > Apply and OK. Then choose Build on the toolbar and choose Build Artifacats. 

Then on an actual Spark cluster, you can send a job using the spark-submit command as shown below with the appropriate java class and jar file name:

```
spark-submit --master yarn --num-executors 3 --executor-cores 5  --class [class] [jar file]
```

## <a name="saving">Saving and Loading Neural Network Models</a>

Once a neural network is fully trained, then to actual deploy the neural network we will want to save how the model is configured and its parameters. We will show two examples of this, once using MultiLayerNetwork and another using ComputationGraph.

### MultiLayerNetwork

To save a MultiLayerNetwork, we will first start by assuming net is a MultiLayerNetwork that has previously been trained. 

```
MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();
```

Then we will choose a location to save the network. Next, we will save the actual model using the MultiLayerNetwork, location, and saveUpdater. saveUpdater should be set to true if you want to train your neural network further in the future. Otherwise, this can be set to false.

```
File locationToSave = new File("MyMultiLayerNetwork.zip"); 
boolean saveUpdater = true; //Updater: the state for Momentum, RMSProp, Adagrad etc.
ModelSerializer.writeModel(net, locationToSave, saveUpdater);
```

To then load the neural network from a saved location, we just require the restoreMultiLayerNetwork function from ModelSerializer.

```
MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
```
### ComputationGraph

The process is very similar for ComputationGraph. We again assume net is the neural network that has previously been trained.

```
ComputationGraph net = new ComputationGraph(conf);
net.init();
```

We then save the model like before using the location, saveUpdater, and saveUpdater.

```
//Save the model
File locationToSave = new File("MyComputationGraph.zip");     
boolean saveUpdater = true;                                         
ModelSerializer.writeModel(net, locationToSave, saveUpdater);
```
To load the ComputationGraph from a saved location, we use the restoreComputationGraph function from ModelSerializer.

```
//Load the model
ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
```
This sums up the process for saving and loading neural network models. We can see that the process to do so is very simple!

## <a name="regression">Regression</a>

Neural networks can perform regression tasks as well. Unlike classification, regression maps one set of continuous values to another set of continuous values after supervised training. For example, predicting the height of a child using the height of the parents is a regression task, while predicting the animal label of an image is a classification task.

Setting up neural network configuration for a regression is similar to before. We can use the MultiLayerNetwork class like before. At the end, you can add an output layer as shown below. We can see that the MSE loss function and identity activation is used. 

```
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .weightInit(WeightInit.XAVIER)
    .updater(new Nesterovs(0.9))
    .list()
    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation(Activation.TANH).build())
    .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        .activation(Activation.TANH).build())
    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true).build();
```

Fortunately, the code for training and evaluating the neural network is the same as before. 

## <a name="complex">More Complex Architectures</a>

In this section we will cover more complex uses of ComputationGraph. Recall that ComputationGraph is a class in DL4J that is more flexible than MultiLayerNetwork in that ComputationGraph can take in multiple input arrays, multiple output arrays, and layers connected to other layers in a structure that does not have to resemble a stack. The ComputationGraph should be used whenever the stack structure is not used for the network configuration. Otherwise MultiLayerNetwork should be used, since they are simpler to configure.

### Graph Vertices

The core building block for a ComputationGraph is the graph vertex. There are multiple types of vertices, such as layervertexes, input vertices, element-wise operation, merge vertices, and subset vertices. We will cover these in more detail.

The LayerVertex are graph vertices with neural network layers and are added using the .addLayer(String, layer, String,...) method. The first input is the name of the layer and the last arguments are the inputs to the layer. 

InputVertex can be specified by .addInputs(String) method in the ComputationGraph configuration. The input strings are user-defined labels, which are able to be referenced later in the configuration. The provided strings define the number of inputs and the order of the strings defines the order of the arrays in the fit methods.

ElementWiseVertex are vertices that can do element-wise oeprations, such as addition or subtraction, of the activations out of other vertices. The input activations should all be the same size and the output size is the same as the input size, since the operations are applied element-wise.

The MergeVertex merges or concatenates the input activations. For example, if the inputs are size 5 and 10 activations, then the output from the MergeVertex will be 15 activations in size. For convolutional network activations the inputs are merged along the depth dimension. Thus, if one activation has 4 features and and another has 5 features, then the resulting output will have dimensions (4+5) x width x height activations. 

The SubsetVertex is a vertex that can be used to obtain a subset of the activations out of a particular vertex. For example, if you want to obtain the first 5 activations of a vertex, you can use the method .addVertex("subset1", new SubsetVertex(0,4), "layername"). This method will obtain the 0th to 5th (inclusive) activations out of the vertex. 

### Examples 

We will go over 3 examples of more complex neural network architectures. The first will be an example of a 2-layer recurrent network with skip connections, meaning that the input will hit both the first and second layers of the network. The second will be an example with multiple inputs and a MergeVertex, and the last will be an example of multi-task learning neural network. 

The code for the first example is shown below. We can see that both layers take in the input layer named "input." Additionally, the second layer "L2" takes in the first layer "L1" as well. 

```
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    .learningRate(0.01)
    .graphBuilder()
    .addInputs("input") //can use any label for this
    .addLayer("L1", new GravesLSTM.Builder().nIn(5).nOut(5).build(), "input")
    .addLayer("L2",new RnnOutputLayer.Builder().nIn(5+5).nOut(5).build(), "input", "L1")
    .setOutputs("L2")   //We need to specify the network outputs and their order
    .build();

    ComputationGraph net = new ComputationGraph(conf);
    net.init();
```

The second example concatenates the two input arrays using a MergeVertex and the code is as follows:

```
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    .learningRate(0.01)
    .graphBuilder()
    .addInputs("input1", "input2")
    .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input1")
    .addLayer("L2", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input2")
    .addVertex("merge", new MergeVertex(), "L1", "L2")
    .addLayer("out", new OutputLayer.Builder().nIn(4+4).nOut(3).build(), "merge")
    .setOutputs("out")
    .build();
```

Note that we have two inputs "input1" and "input2" which are added to the configuration using the same .addInputs() method. We have two hidden layers "L1" and "L2" which have activations that are concatenated using the "merge" vertex. 

The third example is of a multi-task learning where multiple independent predictions are made from the same network. This example network performs classification and regression simultaneously with "out1" for classification and "out2" for regression. As shown below, both "out1" and "out2" layers take in the first layer "L1" and are set to be the outputs of the network using the .setOutputs() method of the class. 

```
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    .learningRate(0.01)
    .graphBuilder()
    .addInputs("input")
    .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input")
    .addLayer("out1", new OutputLayer.Builder()
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(4).nOut(3).build(), "L1")
    .addLayer("out2", new OutputLayer.Builder()
        .lossFunction(LossFunctions.LossFunction.MSE)
        .nIn(4).nOut(2).build(), "L1")
    .setOutputs("out1","out2")
    .build();
```

From these examples, we can see that it is not very difficult to make atypical network structures using DL4J due to the flexibility of the ComputationGraph class. Thus, ComputationGraph neural networks should be deployed whenever MultiLayerNetworks cannot be used to create the architecture you desire.

### DL4J's Programming Guide  

[1. Intro: Deep Learning, Defined](01_intro)
[2. Process Overview](02_process)
[3. Program & Code Structure](03_code_structure)
[4. Convolutional Network Example](04_convnet)
[5. LSTM Network Example](05_lstm)
[6. Feed-Forward Network Example](06_feedforwardnet)
[7. Natural Language Processing](07_nlp)
[8. AI Model Deployment](08_deploy)
[9. Troubleshooting Neural Networks](09_troubleshooting)

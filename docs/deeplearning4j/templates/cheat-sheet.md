---
title: Deeplearning4j Cheat Sheet
short_title: Cheat Sheet
description: Snippets for common functionality in Eclipse Deeplearning4j.
category: Get Started
weight: 2
---

## Cheat sheet code snippets

The Eclipse Deeplearning4j libraries come with a lot of functionality, and we've put together this cheat sheet to help users assemble neural networks and use tensors faster.

## Neural networks

Code for configuring common parameters and layers for both `MultiLayerNetwork` and `ComputationGraph`. See [MultiLayerNetwork](/api/{{page.version}}/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html) and [ComputationGraph](/api/{{page.version}}/org/deeplearning4j/nn/graph/ComputationGraph.html) for full API.

**Sequential networks**

Most network configurations can use `MultiLayerNetwork` class if they are sequential and simple.

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(1234)
    // parameters below are copied to every layer in the network
    // for inputs like dropOut() or activation() you should do this per layer
    // only specify the parameters you need
    .updater(new AdaGrad())
    .activation(Activation.RELU)
    .dropOut(0.8)
    .l1(0.001)
    .l2(1e-4)
    .weightInit(WeightInit.XAVIER)
    .weightInit(Distribution.TruncatedNormalDistribution)
    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
    .gradientNormalizationThreshold(1e-3)
    .list()
    // layers in the network, added sequentially
    // parameters set per-layer override the parameters above
    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .build())
    .layer(new ActivationLayer(Activation.RELU))
    .layer(new ConvolutionLayer.Builder(1,1)
            .nIn(1024)
            .nOut(2048)
            .stride(1,1)
            .convolutionMode(ConvolutionMode.Same)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.IDENTITY)
            .build())
    .layer(new GravesLSTM.Builder()
            .activation(Activation.TANH)
            .nIn(inputNum)
            .nOut(100)
            .build())
    .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
            .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true)
    .build();

MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(conf);
```

**Complex networks**

Networks that have complex graphs and "branching" such as *Inception* need to use `ComputationGraph`.

```java
ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
	.seed(seed)
    // parameters below are copied to every layer in the network
    // for inputs like dropOut() or activation() you should do this per layer
    // only specify the parameters you need
    .activation(Activation.IDENTITY)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(updater)
    .weightInit(WeightInit.RELU)
    .l2(5e-5)
    .miniBatch(true)
    .cacheMode(cacheMode)
    .trainingWorkspaceMode(workspaceMode)
    .inferenceWorkspaceMode(workspaceMode)
    .cudnnAlgoMode(cudnnAlgoMode)
    .convolutionMode(ConvolutionMode.Same)
    .graphBuilder()
    // layers in the network, added sequentially
    // parameters set per-layer override the parameters above
    // note that you must name each layer and manually specify its input
    .addInputs("input1")
    .addLayer("stem-cnn1", new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}, new int[] {3, 3})
    	.nIn(inputShape[0])
    	.nOut(64)
	    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
	    .build(),"input1")
    .addLayer("stem-batch1", new BatchNormalization.Builder(false)
    	.nIn(64)
    	.nOut(64)
    	.build(), "stem-cnn1")
    .addLayer("stem-activation1", new ActivationLayer.Builder()
    	.activation(Activation.RELU)
    	.build(), "stem-batch1")
    .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
        .activation(Activation.SOFTMAX).nOut(numClasses).lambda(1e-4).alpha(0.9)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
        "stem-activation1")
    .setOutputs("lossLayer")
    .setInputTypes(InputType.convolutional(224, 224, 3))
    .backprop(true).pretrain(false).build();

ComputationGraph neuralNetwork = new ComputationGraph(graph);
```

### Layers

- [Common](./deeplearning4j-nn-layers)
- [Convolutional](./deeplearning4j-nn-convolutional)
- [Recurrent](./deeplearning4j-nn-recurrent)
- [Vertices](./deeplearning4j-nn-vertices)
- [Autoencoders](./deeplearning4j-nn-autoencoders)
- [Custom Layers](./deeplearning4j-nn-custom-layer)

### Configuration

- [Activations](./deeplearning4j-nn-activations)
- [Updaters](./deeplearning4j-nn-updaters)
- [Listeners](./deeplearning4j-nn-listeners)


## Training

The code snippet below creates a basic pipeline that loads images from disk, applies random transformations, and fits them to a neural network. It also sets up a UI instance so you can visualize progress, and uses early stopping to terminate training early. You can adapt this pipeline for many different use cases.

```java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
int numExamples = Math.toIntExact(fileSplit.length());
int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
InputSplit trainData = inputSplit[0];
InputSplit testData = inputSplit[1];

boolean shuffle = false;
ImageTransform flipTransform1 = new FlipImageTransform(rng);
ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
ImageTransform warpTransform = new WarpImageTransform(rng, 42);
List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
	new Pair<>(flipTransform1,0.9),
    new Pair<>(flipTransform2,0.8),
    new Pair<>(warpTransform,0.5));

ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

// training dataset
ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(trainData, null);
DataSetIterator trainingIterator = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);

// testing dataset
ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(testData, null);
DataSetIterator testingIterator = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numLabels);

// early stopping configuration, model saver, and trainer
EarlyStoppingModelSaver saver = new LocalFileModelSaver(System.getProperty("user.dir"));
EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
    .evaluateEveryNEpochs(1)
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
    .scoreCalculator(new DataSetLossCalculator(testingIterator, true))     //Calculate test set score
    .modelSaver(saver)
    .build();

EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, neuralNetwork, trainingIterator);

// begin training
trainer.fit();
```

### Data components

- [Record Readers](./datavec-readers)
- [Normalization](./datavec-normalization)
- [Iterators](./deeplearning4j-nn-iterators)
- [Early Stopping](./deeplearning4j-nn-early-stopping)

### Prebuilt image transforms

- [org.datavec.image.transform](/api/{{page.version}}/org/datavec/image/transform/package-frame.html)


## Complex Transformation

DataVec comes with a portable `TransformProcess` class that allows for more complex data wrangling and data conversion. It works well with both 2D and sequence datasets.

```java
Schema schema = new Schema.Builder()
    .addColumnsDouble("Sepal length", "Sepal width", "Petal length", "Petal width")
    .addColumnCategorical("Species", "Iris-setosa", "Iris-versicolor", "Iris-virginica")
    .build();

TransformProcess tp = new TransformProcess.Builder(schema)
    .categoricalToInteger("Species")
    .build();

// do the transformation on spark
JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);
```

We recommend having a look at the [DataVec examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/datavec-examples/src/main/java/org/datavec/transform) before creating more complex transformations.

### Utilities

- [Transforms](./datavec-transforms)
- [Executors](./datavec-executors)
- [Schema](./datavec-schema)
- [Analysis](./datavec-analysis)
- [Conditions](./datavec-conditions)
- [Filters](./datavec-filters)
- [Operations](./datavec-operations)
- [Visualization](./datavec-visualization)
- [Reductions](./datavec-reductions)


## Evaluation

Both `MultiLayerNetwork` and `ComputationGraph` come with built-in `.eval()` methods that allow you to pass a dataset iterator and return evaluation results.

```java
// returns evaluation class with accuracy, precision, recall, and other class statistics
Evaluation eval = neuralNetwork.eval(testIterator);
System.out.println(eval.accuracy());
System.out.println(eval.precision());
System.out.println(eval.recall());

// ROC for Area Under Curve on multi-class datasets (not binary classes)
ROCMultiClass roc = neuralNetwork.doEvaluation(testIterator, new ROCMultiClass());
System.out.println(roc.calculateAverageAuc());
System.out.println(roc.calculateAverageAucPR());
```

For advanced evaluation the code snippet below can be adapted into training pipelines. This is when the built-in `neuralNetwork.eval()` method outputs confusing results or if you need to examine raw data.

```java
//Evaluate the model on the test set
Evaluation eval = new Evaluation(numClasses);
INDArray output = neuralNetwork.output(testData.getFeatures());
eval.eval(testData.getLabels(), output, testMetaData); //Note we are passing in the test set metadata here

//Get a list of prediction errors, from the Evaluation object
//Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
List<Prediction> predictionErrors = eval.getPredictionErrors();
System.out.println("\n\n+++++ Prediction Errors +++++");
for(Prediction p : predictionErrors){
    System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
        + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
}

//We can also load the raw data:
List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);
for(int i=0; i<predictionErrors.size(); i++ ){
    Prediction p = predictionErrors.get(i);
    RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
    INDArray features = predictionErrorExamples.getFeatures().getRow(i);
    INDArray labels = predictionErrorExamples.getLabels().getRow(i);
    List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

    INDArray networkPrediction = model.output(features);

    System.out.println(meta.getLocation() + ": "
        + "\tRaw Data: " + rawData
        + "\tNormalized: " + features
        + "\tLabels: " + labels
        + "\tPredictions: " + networkPrediction);
}

//Some other useful evaluation methods:
List<Prediction> list1 = eval.getPredictions(1,2);                  //Predictions: actual class 1, predicted class 2
List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //All predictions for actual class 2
```

### Utilities

- [Evaluation](./deeplearning4j-nn-evaluation)

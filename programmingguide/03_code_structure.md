---
title: DeepLearning4j: Programming & Code Structure
layout: default
---

# DeepLearning4j: Program & Code Structure

Below, we outline a common structure to be followed for preparing data and configuring, training, and evaluating a neural network. This structure can be used for various neural network architectures and data.

- [**ETL and Data Processing**](#etl)
- [**Neural Network Building**](#nnbuilding)
- [**Training**](#train)
- [**Evaluation**](#eval)

## <a name="etl">ETL and Data Processing</a>

DataVec is a library for vectorization and ETL (extracting, transforming, and loading data) focused on machine learning. The library is used primarily for getting raw data into a format that neural networks can read. DataVec can convert all major types of data, such as text, CSV, audio, images, and video. It can also apply scaling, normalization, or other transformations to the vectorized data. Furthermore, DataVec can be extended for specialized inputs such as exotic image formats. 

### Schemas and TransformProcesses

Schemas are used to define the layout of tabular data. The basic way to initialize a `Schema` is as follows:

     Schema inputDataSchema = new Schema.Builder()
        .addColumnString("DateTimeString")
        .addColumnsString("CustomerID", "MerchantID")
        .addColumnInteger("NumItemsInTransaction")
        .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
        .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)
        .build();

The columns should be added in the `Schema` in the order they appear in the data. In the `TransactionAMountUSD` column, we further specify that the amount should be non-negeative and have no maximum limit, NaN, or infinite values. 

Once the `Schema` is defined, you can print it out to look at the details.

    System.out.println(inputDataSchema);

To perform transformation processes on the data using the `Schema` we defined, a `TransformProcess` is needed. `TransformProcesses` can remove unnecesasry columns, filter out observations, create new variables, rename columns, and more. Below is code for defining a `TransformProcess`, which takes the `inputDataSchema` as input. 

    TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
        .removeColumns("CustomerID","MerchantID")
        .conditionalReplaceValueTransform(
                "TransactionAmountUSD",     //Column to operate on
                new DoubleWritable(0.0),    //New value to use, when the condition is satisfied
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0
        .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
        .renameColumn("DateTimeString", "DateTime")

The above `TransformProcess` first removes the unnecessary columns `CustomerID` and `MerchantID` of our `Schema`. The next transformation acts on the `TransactionAmountUSD` and replaces values less than 0 with 0. Lastly, the `TransformProcess` converts `DateTimeString` to a format like `2016/01/01 17:50.000` and renames the column as `DateTime`.

A `TransformProcess` can then used to output a final `Schema`, which we can view. This is done as follows:

    Schema outputSchema = tp.getFinalSchema();
    System.out.println(outputSchema);

To actually execute the transformations, Spark is needed. The first step is to set up a Spark Context which is done as follows:

    SparkConf conf = new SparkConf();
    conf.setMaster("local[*]");
    conf.setAppName("DataVec Example");

    JavaSparkContext sc = new JavaSparkContext(conf);

We will assume that the data is contained in `BasicDataVecExample/exampledata.csv`, and we will create a `JavaRDD` from the raw data.

    String path =new ClassPathResource("BasicDataVecExample/exampledata.csv").getFile().getAbsolutePath();
    JavaRDD<String> stringData = sc.textFile(path);

Lastly, a `RecordReader` is needed (we'll explain `RecordReaders` in detail in the next section). They are needed to parse the data into the record format.
        
    RecordReader rr = new CSVRecordReader();
    JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

<!-- grammar, I think this can be phrased better...same feedback for other sections -->
Finally, to execute the transformation on the parsed data, a `SparkTransformExecutor` is used.

    JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);

### RecordReaders

`RecordReaders` are a class in the DataVec library used to serialize or parse data into records; i.e., a collection of elements indexed by a unique ID. They are the first step to getting the data into a format that neural networks can understand. Depending on the data, different subclasses of `RecordReaders` should be used. For example, if the data is in a CSV file, then `CSVRecordReader` should be  used, and if the data is contained in an image, then `ImageRecordReader` should be used, and so forth.

To initialize a `RecordReader`, simply use code similar to the lines below.

    CSVRecordReader features = new CSVRecordReader();
    recordReader.initialize(new FileSplit( new File(path)));

In the example above, the data is assumed to be in CSV format, which is why `CSVRecordReader` is used. The `CSVRecordReader` can also take in optional parameters to skip a specified number of lines and specify a delimiter.

For an image, `ImageRecordReader` should be used. We can see that there are only a few differences in parameters between different `RecordReaders` in the below example.

    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
    recordReader.initialize(new FileSplit(parentDir));

Here, `labelMaker` is a `ParentPathLabelGenerator`, which is used if the labels aren't manually specified. The `ParentPathLabelGenerator` will parse the parent directory and use the name of the subdirectories containing the data as the label and class names.  

### DatasetIterators

Once the raw data is converted into record format, they will need to be processed into `DataSet` objects, which can be fed directly into a neural network. To do this, `DataSetIterators` are used to traverse records sequentially using a `RecordReader` object. `DataSetIterators` iterate through the input datasets, fetch examples at each iteration, and load them in a `DataSet` object, which is a `INDArray`, or n-dimensional array. The number of examples fetched at each iteration depends on the batch size specified for neural network training. Once the `DataSet` object is created, the data is ready for use.

Below is an example of initializing a `DataSetIterator`. The parameters are the DataVec `RecordReader`, batch size, the offset of the label index, and the total number of label classes.

    DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum)

<!-- you can cut down on wordiness by combining sentence fragments, say something like "a MultiDataSetIterator is used for multiple..." -->
Other types of `DataSetIterators` can also be used depending on the data and the neural network: For example, a `MultiDataSetIterator` can be used if the neural network has multiple inputs or outputsshould. This is similar to a `DataSetIterator`, but multiple inputs and outputs can be defined. These inputs and outputs need to be in `RecordReader` format like before. In the example below, there are one set of inputs and outputs each but more can be added if needed.

    MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("trainFeatures", trainFeatures)
                .addInput("trainFeatures")
                .addReader("trainLabels", trainLabels)
                .addOutput("trainLabels")
                .build();

Once the `DataSetIterator` is initialized, further transformations can be applied. For example, you can scale the data as follows:

    DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);

Finally, when we are content with the data, we can create `DataSet` objects by iterating through the data. The `DataSetIterator` will fetch a batch of data points in a DataSet format. 

    DataSet next = iterator.next();

With these `DataSet` objects, the data is now ready to be read by a neural network.

### Apache Spark

If Apache Spark is used for the job, a `JavaRDD<DataSet>` object must be created before passing the data into a neural network.  The following snippet of code is one way to obtain a `JavaRDD<DataSet>`. We will assume that `trainData` is a `DataSetIterator` and that `sc` is a `JavaSparkContext`.

    List<DataSet> trainDataList = new ArrayList<>();

    while (trainData.hasNext()) {
        trainDataList.add(trainData.next());
    }

    JavaRDD<DataSet> JtrainData = sc.parallelize(trainDataList);

As before, once a `JavaRDD<DataSet>` is created, the data is ready to be fed into a neural network using Spark. 

## <a name="nnbuilding">Building a Neural Network</a>

Two classes can be used for building neural networks: `MultiLayerNetwork` and `ComputationGraph`. We will provide a brief overview on each of these networks.

### MultiLayerNetwork 

In order to create a `MultiLayerNetwork`, we must first define its configuration. This can be done using `MultiLayerCongfiguration` as shown below. 

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

The `MultiLayerConfiguration` can be used to define the parameter and structure of the neural network. This is one example of a configuration:

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.05)
        // ... other hyperparameters
        .list()
        .backprop(true)
        .build();

To add hidden layers, you can call on the `layer()` function of the `NeuralNetworkConfiguration.Builder()`. Within `layer()`, you can define the type of layer and the number of inputs and and outputs. An example of the first hidden layer (0th index) is 

    .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                .build())

 which has 784 inputs and 250 outputs. In this example, an ordinary dense hidden layer was used, but different types of layers, such as convolutional layers can be defined. 

 An example for a convolutional layer is shown below with a kernel size of (5,5), stride of (1,1), and padding of (2,2). The nIn parameter specifies the number of channels and and nOut is the number of filters used in the neural network.

    .layer(new ConvolutionLayer.Builder(5,5)
        .nIn(3)
        .nOut(16)
        .stride(1,1)
        .padding(2,2)
        .name("convolutional layer")
        .activation(Activation.RELU)
        .build())

Thus, the process for building different neural network architectures are similar with the main difference being the type of layer used.

Once the configuration is defined for a `MultiLayerNetwork`, use the following line:

        MultiLayerNetwork model = new MultiLayerNetwork(config);

### ComputationGraph

To create a `ComputationGraphConfiguration` use the following line:

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()

Note that the code is slightly different for configuring a `ComputationGraph` rather than a `MultiLayerNetwork`. However, similar to before, `ComputationGraphConfiguration` can be used the define neural network parameters such as the learning rate, optimization algorithm, and network structure. For example, the following code chunk configures a `ComputationGraph` using stochastic gradient descent and a learning rate of 0.1. It also defines the network structure to include one hidden layer with 500 nodes and an output layer which outputs a probability. 

    ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
	   .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	   .learningRate(0.1)
	   // other parameters
	   .graphBuilder()
	   .setInputTypes(InputType.feedForward(100))
	   // add neural network layers
       .build();

To create a hidden layer, the `addLayer()` function is used once `graphBuilder()` is called. Note that this is different from the `layer()` function used to add layers in a `MultiLayerNetwork`. Within `addLayer()`, `DenseLayer.Builder()` is used like before and an example of this is shown below. We see that we can also name layers and the name of its input. 

        .addLayer("dense1", new DenseLayer.Builder() .nIn(100).nOut(500).build(),"trainFeatures")

Once the configuration is defined, we initialize the `ComputationGraph` using 

        ComputationGraph model = new ComputationGraph(config);

Now all that is left is to train and evaluate the neural network we initialized above. 

## <a name="train">Neural Net Training</a>

Writing code to train a neural network is easy once the net's configuration is defined. Typically, two loops wil be needed: one to iterate through the number of epochs, and another to iterate through batches of training data. Note that the latter loop will implicitly be done behind the scenes with either `MultiLayerNetwork` or `ComputationGraph`. For now, we'll assume the training split of the data is already contained in `DataSetIterators` called `trainData`. The process of converting data into this format will be convered in the next chapter.

    // Training a neural network
    for (int epoch = 0; epoch < NB_EPOCHS; epoch++) { // outer loop over epochs
        model.fit(trainData); // implicit inner loop over minibatches
    }

As you can see, the code to train a neural network is quite concise. You call `.fit()` on `model`. This is because `MultiLayerNetwork` and `ComputationGraph` handles the details and intricacies so we don't have to worry about it. 

### Monitoring the Neural Network during Training

In order to monitor how the neural network is processing after each epoch, we can use the following command before we train the neural network. This collects information, such as the loss, from the neural network and outputs them after a certain number of iterations. 

    model.setListeners(new ScoreIterationListener(10));

You can also use the UI to help train your neural network by following the instructions from the previous chapter.

## <a name="eval">Evaluation</a>

Once the neural network is fully trained, the remaining task is to evaluate the neural network on the test data, which we split from the training data earlier. We assume that the testing split of the data is contained in a `DataSetIterator` called `testData`. An example code snipper for doing this is shown below:

    Evaluation eval = model.evaluate(testData);
    log.info(eval.stats());

Similar code can be used on the validation test to evaluate the neural network between epochs within the training loop as well.

Specific metrics such as AUC (area under curve) for a ROC (receiving operator characteristic) curve can also be evaluated if the output is appropriate. An example of this is shown below:

	ROC roc = new ROC();
    while (testData.hasNext()) {
        DataSet batch = testData.next();
        INDArray[] output = model.output(batch.getFeatures());
        roc.eval(batch.getLabels(0), output[0]);
    }
    log.info("FINAL TEST AUC: " + roc.calculateAUC());

In the next few chapters, we will see specific examples of this process for convolutional, recurrent, and feed-forward networks.

### DL4J's Programming Guide  

* [1. Intro: Deep Learning, Defined](01_intro)
* [2. Process Overview](02_process)
* [3. Program & Code Structure](03_code_structure)
* [4. Convolutional Network Example](04_convnet)
* [5. LSTM Network Example](05_lstm)
* [6. Feed-Forward Network Example](06_feedforwardnet)
* [7. Natural Language Processing](07_nlp)
* [8. AI Model Deployment](08_deploy)
* [9. Troubleshooting Neural Networks](09_troubleshooting)

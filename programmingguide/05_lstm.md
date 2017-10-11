title: DeepLearning4j: LSTM Network
layout: default

------

# DeepLearning4j: LSTM Network Example

In this section, we will cover an example of an LSTM (long short term memory) neural network. A LSTM network is a type of recurrent network where the input is usually sequential in nature such as speech or text. At each timestep of the data, the neural network takes in input features and the output is then passed onto the next timestep. If the output is also time series in nature, then the neural network saves the output at each timestep. If not, then only the output of the last timestep is taken. Thus, we can see that recurrent neural networks differ from a regular feed forward neural network (FNN), since FNN's only take in one set of features per observation and produce one output. Recurrent neural networks are not bound to this constraint.

An LSTM is a special type of recurrent network with a special gated cell where the network can store and remove information from. The gate of the cell filters the content to keep only relevant information. At each timestep, the neural network takes in both the input features and the content from the gated cell. LSTM's are useful becaues they prevent vanishing or exploding gradients that hinder a neural network from learning. For a more in depth overview of LSTM's, look [here](https://deeplearning4j.org/lstm.html).

## Data and ETL

The data used for this example is from the [2012 Physionet Challenge](https://physionet.org/challenge/2012/). It consists of ICU stays of 4000 patients and is comprised of 86 features and a binary mortality label. The number of timesteps varies between each patient and the time elapsed between timesteps vary as well. The features of an observation is stored in its own csv file with the label in a separate csv file in a different directory. All the feature files are contained in the same directory and the same goes for the label files.

To start the ETL process, we use CSVSequenceRecordReaders to parse the data into record format. We take the first 3200 observations as the training portion of the data.
		
		private static File featuresDir = new File(path); // path to directory containing feature files
		public static final int NB_TRAIN_EXAMPLES = 3200;

		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");  // number of rows to skip + delimiter
        trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

Once we have the RecordReaders, we need to initialize the DataSetIterator. 

        int numLabelClasses = 2;

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

The DataSetIterator for SequenceRecordReaders differs slightly from the one used for the feed forward neural network in the earlier chapter. It additionally requires the alignment mode, which aligns the input and output of each observation. In this case, we have one output at the last input timestep, so we use ALIGN_END, which will also zero pad the label. 

## Building a LSTM Neural Network using ComputationGraph

Now that the training data is in a format that a neural network can read, we can start building the neural network. Unlike previous examples, we will use the ComputationGraph class instead of the MultiLayerNetwork class.

		public static final double LEARNING_RATE = 0.05;
		public static final int lstmLayerSize = 300;
		public static final int NB_INPUTS = 86;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictMortality")
                .addLayer("L1", new GravesLSTM.Builder()
                        .nIn(NB_INPUTS)
                        .nOut(lstmLayerSize)
                        .activation(Activation.SOFTSIGN)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .build(), "trainFeatures")
                .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .nIn(lstmLayerSize).nOut(numLabelClasses).build(),"L1")
                .pretrain(false).backprop(true)
                .build();

We can see that the neural network code for building a ComputationGraph is similar to that of MultiLayerNetwork. To build the LSTM layer, the GravesLSTM.Builder() class is used with the specified dimension of a timestep and layer size. Furthermore, the output layer has a output dimension of 2, since we are classifying a binary mortality label using the data. 

Next we initialize the actual ComputationGraph using the previously specified configuration.

	ComputationGraph model = new ComputationGraph(conf);
    model.init();

## Training and Evaluating a LSTM Neural Network

The code to train a LSTM network does not change from previous examples. The ComputationGraph class takes care of the training process and only the training loop for the number of epochs is needed. Here we train the neural network using 15 epochs or passes through the training set.

		public static final int NB_EPOCHS = 15;

		for( int i=0; i<NB_EPOCHS; i++ ){
            model.fit(trainData); 
        }

To evaluate the trained neural network, we compute the AUC (area under the curve) for a ROC curve with the output predictions from the neural network and the labels of the observations. Note that a perfect model will have an AUC of 1.0 and a randomly guessing model will achieve a AUC near 0.5. 

We assume the testing split of the data is contained in the DataSetIterator testData. The ETL process for the test split is similar to the process shown above for the training split.

		ROC roc = new ROC(100);
        while (testData.hasNext()) {
            DataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.evalTimeSeries(batch.getLabels(), output[0]);
        }
        log.info("FINAL TEST AUC: " + roc.calculateAUC());
---
layout: default
---

# Customized Image Data Pipeline

Deeplearning4j's examples run on benchmark datasets that don't present any obstacles in the data pipeline, because we abstracted them away. But real-world users start with raw, messy data, which they need to preprocess, vectorize and use to train a neural net for clustering or classification. 

*Canova* is our machine-learning vectorization library, and it is useful for customizing how you prepare data that a neural net can learn. This tutorial will walk through how it loads *Labeled Faces in the Wild*, a supervised set of 13,233 photographs representing 5,749 relatively famous people. (The full code example [lives on Github](https://github.com/deeplearning4j/Canova-examples/blob/master/src/main/java/datapipelines/ImageClassifierExample.java).)

## Loading Labels

        // Set path to the labeled images
        String labeledPath = System.getProperty("user.home")+"/lfw";
        
        //create array of strings called labels
         List<String> labels = new ArrayList<>(); 
        
        //traverse dataset to get each label
        for(File f : new File(labeledPath).list Files()) { 
            labels.add(f.getName());
        }

After you download a LFW dataset to work with, place it in the right path and feed the path into a variable called labelPath, you're ready to read and load the data, and create an array to hold the images' labels. 

## Reading Records, Iterating Over Data

Here's the code that helps transform raw images into data records we can work with:

        // Instantiating RecordReader. Specify height and width of images.
        RecordReader recordReader = new ImageRecordReader(28, 28, true,labels);
        // Point to data path. 
        recordReader.initialize(new FileSplit(new File(labeledPath)));

The RecordReader is a class in Canova that helps convert the byte-oriented input into data that's oriented toward a record; i.e. a collection of elements fixed in number and indexed with a unique ID. 

Here, LFW images have been scaled to 28 pixels x 28 pixels (rescaling images to other dimensions will means you change the parameters you feed to the ImageRecorder above, as well as the nIn hyperparameter below). The  [ImageRecordReader](https://github.com/deeplearning4j/Canova/blob/f03f32dd42f14af762bf443a04c4cfdcc172ac83/canova-nd4j/canova-nd4j-image/src/main/java/org/canova/image/recordreader/ImageRecordReader.java), which extends the RecordReader, is instructed to expect 28 x 28 pixel images to parse (you'll change the dimensions to match your custom images); "true" instructs it to append a label to the record; and "labels" is the array to which that label number is appended. Without the label, all you have is a bunch of pixels. 

        // Canova to DL4J
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784,labels.size());

With this line, we move to a Deeplearning4j class called DataSetIterator. Like all Iterators, this one traverses the elements of a list. It makes a pass through the list, accesses each item sequentially, keeps track of how far it has progressed by pointing to its current element, and modifies itself to point to the next element with each new step in the traversal.

The DataSetIterator, suprisingly enough, iterates through input datasets, fetching one or more new examples with each iteration, and loading those examples into a DataSet object that neural nets can work with. The line above also tells the [RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/3e5c6a942864ced574c7715ae548d5e3cb22982c/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.java) to convert the image to a straight line of elements, rather than a 28 x 28 grid; it also specifies the number of labels possible. 

## Configuring the Model

Below is a neural net configuration. Many of the hyperparameters have been covered in the [Iris tutorial](../iris-flower-dataset-tutorial.html), so we'll focus here on just a few distinguishing characteristics. 

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .constrainGradientToUnitNorm(true)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(1,1e-5))
                .iterations(100).learningRate(1e-3)
                .nIn(784).nOut(labels.size())
                .visibleUnit(org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.RECTIFIED)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .list(4).hiddenLayerSizes(600, 250, 100).override(3, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 3) {
                            builder.layer(new org.deeplearning4j.nn.conf.layers.OutputLayer());
                            builder.activationFunction("softmax");
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                }).build();

* *optimizationAlgo* relies on conjugate gradient, rather than LBFGS. 
* *iterations* are set to just 100. 
* *nIn* is set to 784, so each pixel in an image gets an input node. If your images change dimensions, nIn should, too.
* *layer* is set to RBM, and the zero-indexed *list* operator below that is set to 4, which means three RBM hidden layers, which together form a deep-belief network. 
* *hiddenLayerSizes* sets the number of nodes in each hidden layer, making them progressively smaller. This is dimensionality reduction at work. 
* *override* creates a classification layer at the end with a softmax function. Softmax is a sigmoid function that allows for multinomial classification. 
* *lossFunction* is set to Monte Carlo Cross Entropy. This loss function will be used to train the classification layer. 

## Building and Training the Model

At the end of the configuration, call build and pass the network's configuration into a MultiLayerNetwork object.

                }).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(10)));

The second line sets the iteration listeners, which keep track of their progress across the dataset, and monitor the error produced at the end of each iteration. 

Calling next on the iterator advances it one step, and returns the new value it points to. We call that method below the neural net's configuration, and the call looks like this. 

        // Training
        while(iter.hasNext()){
            DataSet next = iter.next();
            network.fit(next);
        }

## Evaluating the Model

With this example, we haven't shown you how to split your training dataset from your test set. Instead you'll see this:

        // Using the same training data as test. 
        
        iter.reset();
        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }
        
        System.out.println(eval.stats());

On Iris, which doesn't employ an Iterator because of the dataset's small size, we split, test and train datasets like this:

        log.info("Split data....");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

To split test and train on large datasets, you'll have to iterate through both the test and training datasets. For the moment, we'll leave that to you. 

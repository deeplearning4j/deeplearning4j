---
title: Running Deep Learning on Distributed GPUs With Spark
layout: default
---

# Running Deep Learning on Distributed GPUs With Spark

Deeplearning4j trains deep neural networks on distributed GPUs using Spark. Specifically, we use Spark to load data and GPUs to process it with cuDNN. 

Deeplearning4j includes libraries for the automatic tuning of neural networks, deployment of those neural-net models, visualization and integrations with other data pipeline tools that make dealing with data on production clusters easier. Much easier. 

This post is a simple introduction to each of those technologies, which we'll define below. It looks at each individually, and at the end it shows with code how Deeplearning4j pulls them together in an image-processing example. 

## Apache Spark

As an open-source, distributed run-time, Spark can orchestrate multiple host threads. It was the Apache Foundation’s most popular project last year. Deeplearning4j only relies on Spark as a data-access layer, since we have heavy computation needs that require more speed and capacity than Spark currently provides. It’s basically fast ETL. 

## CUDA

Now, CUDA is NVIDIA's parallel computing platform and API model, a software layer that gives access to GPUs' lower-level instructions, and which works with C, C++ and FORTRAN. 

## cuDNN
cuDNN stands for the CUDA Deep Neural Network Library, and it was created by the GPU maker NVIDIA. cuDNN is a library of primitives for standard deep learning routines: forward and backward convolution, pooling, normalization, and activation layers. 

cuDNN is one of the fastest libraries for deep convolutional networks (and more recently, for recurrent nets). It ranks at or near the top of several [image-processing benchmarks](https://github.com/soumith/convnet-benchmarks) conducted by Soumith Chintala of Facebook. Deeplearning4j wraps cuDNN, and gives the Java community easy access to it. 

[Deeplearning4j](http://deeplearning4j.org/) is the most widely used open-source deep learning tool for the JVM, including the Java, Scala and Clojure communities. Its aim is to bring deep learning to the production stack, integrating tightly with popular big data frameworks like Hadoop and Spark. DL4J works with all major data types – images, text, time series and sound – and includes algorithms such as convolutional nets, recurrent nets like LSTMs, NLP tools like word2vec and doc2vec, and various types of autoencoder.

Deeplearning4j is part of a free enterprise distribution called the Skymind Intelligence Layer, or SKIL. It is one of several open-source libraries maintained by Skymind engineers. 

* [ND4J](http://nd4j.org/), or n-dimensional arrays for Java, is the scientific computing library that performs the linear algebra and calculus necessary to train neural nets for DL4J. 
* [libnd4j](https://github.com/deeplearning4j/libnd4j) is the C++ library that accelerates ND4J. 
* [DataVec](https://github.com/deeplearning4j/DataVec) is used to vectorize all types of data.
* [JavaCPP](https://github.com/bytedeco/javacpp) is the glue code that creates a bridge between Java and C++. DL4J talks to cuDNN using JavaCPP.

## Java & C++ Communication: Doesn't Java Slow Down CUDA?

Usually. But we optimized communication by putting operations off heap. JavaCPP implements a `Pointer` class that makes it easy to do [off-heap operations](https://dzone.com/articles/heap-vs-heap-memory-usage) (i.e. data doesn't hit the garbage collector). This allows us to benefit from lower latency and memory management while benefiting from the managed garbage collector where it matters. This is the approach taken by many distributed systems frameworks and databases such as such as Apache Flink, Spark and Hbase.

Java isn't good at linear algebra operations. They should be handled by C++, where we can benefit from hardware acceleration of floating-point operations. That's what libnd4j is for.

## Distributed Deep Learning With Parameter Averaging

TKTKTKTKTKTK

## Code Example

Here’s an example of Deeplearning4j code that runs LeNet on Spark using GPUs.

First we configure Spark and load the data:

    public static void main(String[] args) throws Exception {

        //Create spark context, and load data into memory
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("MNIST");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        int examplesPerDataSetObject = 32;
        DataSetIterator mnistTrain = new MnistDataSetIterator(32, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(32, false, 12345);
        List<DataSet> trainData = new ArrayList<>();
        List<DataSet> testData = new ArrayList<>();
        while(mnistTrain.hasNext()) trainData.add(mnistTrain.next());
        Collections.shuffle(trainData,new Random(12345));
        while(mnistTest.hasNext()) testData.add(mnistTest.next());

        //Get training data. Note that using parallelize isn't recommended for real problems
        JavaRDD<DataSet> train = sc.parallelize(trainData);
        JavaRDD<DataSet> test = sc.parallelize(testData);

Then we configure the neural network:

        //Set up network configuration (as per standard DL4J networks)
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 1;
        int seed = 123;

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .nOut(50)
                        .stride(2,2)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .nOut(200).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

Note that above, we also have a more complex (but versatile) [computation graph api](http://deeplearning4j.org/compgraph) for those familiar with other frameworks.

Also of note here is the builder pattern being used. Since java doesn't have key word args, the fluent builder
pattern is known as a best practice in java land due to the complimenting tools such as intellij for handling
code completion. Despite its verbose nature, its also very easy to wrap in a more concise language such as 
clojure or scala.

Then we tell Spark how to perform parameter averaging:

        //Create Spark multi layer network from configuration
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5) //Do 5 minibatch fit operations per worker, then average and redistribute parameters
                .batchSizePerWorker(examplesPerDataSetObject) //Num of examples that each worker uses per fit operation
                .build();

        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm);

And finally, we train the network by calling `.fit()` on `sparkNetwork`.

        //Train network
        log.info("--- Starting network training ---");
        int nEpochs = 5;
        for( int i=0; i<nEpochs; i++ ){
            sparkNetwork.fit(train);
            System.out.println("----- Epoch " + i + " complete -----");

            //Evaluate using Spark:
            Evaluation evaluation = sparkNetwork.evaluate(test);
            System.out.println(evaluation.stats());
        }

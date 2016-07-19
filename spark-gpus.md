---
title: Running Deep Learning on Distributed GPUs With Spark
layout: default
---

# Running Deep Learning on Distributed GPUs With Spark

Deeplearning4j trains deep neural networks on distributed GPUs using Spark. Specifically, we show the use Spark to load data and GPUs to process images with cuDNN. 

Deeplearning4j includes libraries for the automatic tuning of neural networks, deployment of those neural-net models, visualization and integrations with other data pipeline tools that make dealing with data on production clusters easier much easier. 

This post is a simple introduction to each of those technologies, which we'll define below. It looks at each individually, and at the end it shows with code how Deeplearning4j pulls them together in an image-processing example.

In this post, we will cover the below technologies and their interactions:
 
 1. Apache Spark
 2. CUDA
 3. cuDNN
 4. DL4j ecosystem(deeplearning4j,nd4j,datavec,javacpp)

## Apache Spark

As an open-source, distributed run-time, Spark can orchestrate multiple host threads. It was the Apache Foundation’s most popular project last year. Deeplearning4j only relies on Spark as a data-access layer for a cluster, since we have heavy computation needs that require more speed and capacity than Spark currently provides. It’s basically fast ETL (extract transform load) or data storage and access for the hadoop ecosystem (HDFS or hadoop file system). The goal is to leverage hadoop's data locality mechanisms while speeding up compute with native computations.

## CUDA

Now, CUDA is NVIDIA's parallel computing platform and API model, a software layer that gives access to GPUs' lower-level instructions, and which works with C, C++ and FORTRAN. Deeplearning4j interacts with the gpu and cuda via a mix of custom cuda kernels and java native interface.

## cuDNN
cuDNN stands for the CUDA Deep Neural Network Library, and it was created by the GPU maker NVIDIA. cuDNN is a library of primitives for standard deep learning routines: forward and backward convolution, pooling, normalization, and activation layers. 

cuDNN is one of the fastest libraries for deep convolutional networks (and more recently, for recurrent nets). It ranks at or near the top of several [image-processing benchmarks](https://github.com/soumith/convnet-benchmarks) conducted by Soumith Chintala of Facebook. Deeplearning4j wraps cuDNN via Java Native Interface, and gives the Java community easy access to it. 

## Deeplearning4j, ND4J, DataVec and JavaCPP

[Deeplearning4j](http://deeplearning4j.org/) is the most widely used open-source deep learning tool for the JVM, including the Java, Scala and Clojure communities. Its aim is to bring deep learning to the production stack, integrating tightly with popular big data frameworks like Hadoop and Spark. DL4J works with all major data types – images, text, time series and sound – and includes algorithms such as convolutional nets, recurrent nets like LSTMs, NLP tools like word2vec and doc2vec, and various types of autoencoders.

Deeplearning4j is part of a set of open source libraries for building deep learning applications on the java virtual machine. It is one of several open-source libraries maintained by Skymind engineers. 

* [ND4J](http://nd4j.org/), or n-dimensional arrays for Java, is the scientific computing library that performs the linear algebra and calculus necessary to train neural nets for DL4J. 
* [libnd4j](https://github.com/deeplearning4j/libnd4j) is the C++ library that accelerates ND4J. 
* [DataVec](https://github.com/deeplearning4j/DataVec) is used to vectorize all types of data.
* [JavaCPP](https://github.com/bytedeco/javacpp) is the glue code that creates a bridge between Java and C++. DL4J talks to cuDNN using JavaCPP.



## Spark and DL4j

Deeplearning4j also comes with built in spark integration for handling distributed training of neural nets across a cluster. We use data parallelism (explained below) to scale out training on multiple computers leveraging a GPU (or 4) on each node. We use spark for data access.

A distributed file system allows us to move compute to the data rather than the other way around, allowing us to benefit from an easy to setup way of doing distributed training.


## Java & C++ Communication: Doesn't Java Slow Down CUDA?

Usually. But we optimized communication by putting operations off heap. JavaCPP implements a `Pointer` class that makes it easy to do [off-heap operations](https://dzone.com/articles/heap-vs-heap-memory-usage) (i.e. data doesn't hit the garbage collector). This allows us to benefit from lower latency and memory management while benefiting from the managed garbage collector where it matters. This is the approach taken by many distributed systems frameworks and databases such as such as Apache Flink, Spark and Hbase.

Java isn't good at linear algebra operations. They should be handled by C++, where we can benefit from hardware acceleration of floating-point operations. That's what libnd4j is for.

## Distributed Deep Learning With Parameter Averaging

There are two main methods for the distributed training of neural networks: data parallelism and model parallelism. 

With data parallelism, you subdivide a very large dataset into batches, and distribute those batches to parallel models running on separate hardware to train simultaneously. 

Imagine training on an encyclopedia, subdividing it into batches of 10 pages, and distributing 10 batches to 10 models to train, then averaging the parameters of those trained models in one master model, and pushing the updated weights of the master model out to the distributed models. The model parameters are then averaged at the end of training to yield a single model.

Deeplearning4j relies on data parallelism and uses Spark for distributed host thread orchestration across a cluster.

##Does Parameter Averaging work?
See references at the bottom of this post for some papers to dig in to.



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

We are going to release  a scala wrapper very similar to the keras framework taking advantage
of some of the nicer constructs of the scala language which should help usability quite a bit.

These configurations can also be defined via yaml or json.



##Distributed Training with spark

Then we tell Spark how to perform parameter averaging:

        //Create Spark multi layer network from configuration
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true) //save things like adagrad squared gradient histories
                .averagingFrequency(5) //Do 5 minibatch fit operations per worker, then average and redistribute parameters
                .batchSizePerWorker(examplesPerDataSetObject) //Number of examples that each worker uses per fit operation
                .build();

        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm);

And finally, we train the network by calling `.fit()` on `sparkNetwork`.




        //Train network
        log.info("--- Starting network training ---");
        int nEpochs = 5;
        for( int i = 0; i < nEpochs; i++ ){
            sparkNetwork.fit(train);
            System.out.println("----- Epoch " + i + " complete -----");

            //Evaluate using Spark:
            Evaluation evaluation = sparkNetwork.evaluate(test);
            System.out.println(evaluation.stats());
        }




##References

[1] Training with intra-block parallel optimization and blockwise model-update filtering. In 2016

IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages

5880–5884. IEEE, 2016.

[2] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior,

Paul Tucker, Ke Yang, Quoc V Le, et al. Large scale distributed deep networks. In Advances in

Neural Information Processing Systems, pages 1223–1231, 2012.

[3] Augustus Odena. Faster asynchronous sgd. arXiv preprint arXiv:1601.04033, 2016.

[4] Nikko Strom. Scalable distributed dnn training using commodity gpu cloud computing. In Six-
teenth Annual Conference of the International Speech Communication Association, 2015. http:

//nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf.

[5] Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-sgd for distributed

deep learning. CoRR, abs/1511.05950, 2015. http://arxiv.org/abs/1511.05950.

[6]: http://arxiv.org/abs/1404.5997

---
title: 
layout: default
---

# The Data Pipeline

Data pipelines are a fundamental concept for large-scale machine learning. Pipelines do two things: they move data from one place to another, and they transform it from one shape to another, with the purpose of delivering it to an algorithm in a digestable form.

Here are a few things they do: 

* data retrieval 
* vectorization
* feature engineering
* iterators
* model data ingestion

We use the word ‘data pipeline’ here to denote the entire process, from original state and location to final destination vector, rather than the more limited process of going from featurized data to prediction results.

### Data Retrieval

All data lives somewhere. It could be on Hadoop, in the Amazon cloud, or on a local file system. Data retrieval goes and gets it, and also includes the process of adding new data to existing records.

### Vectorization

While data can mean anything from random digits to various raw media, it must be fed to deep-learning nets as a **vector**. Nothing else counts. Neural nets are on an all-vector diet. These vectors can contain either continuous or discrete data. 

Consider the IRIS data set, and how it can be vectorized. IRIS consists of four physical measurements of three species of flower (we cover it thoroughly in [another example](), but it will be put to other purposes below).

Here is an example of a single record 

|Sepal length|Sepal width|Petal length|Petal width |Species|
|------------|-----------|------------|------------|-------|
|         5.1|        3.5|         1.4|         0.2|I. setosa|

The IRIS data set contains three possible outcomes, therefore three columns for labels. Since the above example is *I.setosa*, a **1** marks that column, and **0**'s mark the others.

Other data sets may allow for multiple columns be marked with a **1**, depending on their model and problem domain. 
<!--EXAMPLE?-->

Since each subspecies is singular and exclusive, Iris only admits one label per data record.

|I. setosa|I. virginica|I. versicolor|
|---------|------------|-------------|
|            1|              0|               0|

Here's a vectorization, all words removed:

|Sepal length|Sepal width|Petal length|Petal width|I. setosa|I. virginica|I. versicolor|
|---|---|---|---|---|---|---|
|5.1|3.5|1.4|0.2|1|0|0|

### Feature Engineering

IRIS has limited dimensions to work with, but the species' petal and sepal lengths are close enough to present challenges in classification, which is what makes the dataset valuable for training. 

![Iris dot plot](http://i.imgur.com/sRxy2at.png)

Now let's look at an image from the MNIST dataset of handwritten numerals:

![MNIST numeral](http://i.imgur.com/w9SCkej.png)

It is interesting to note that most data -- images and otherwise -- are flattened to a long, almost two-dimensional, tape, a single line of numbers, which can be fed into the deep-learning algorithm.

Images themselves are nothing more than collections of pixels, and those pixels are simply discrete values. Sometimes three bytes, sometimes just 1, all of them strung together. Consider this

    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  13 13 13 13 0  0  17 17 17 17 0  0 11 11 11 11  0  0 15 15 15 15  0
    0  13 0  0  0  0  0  17 0  0  0  0  0 11  0  0  0  0  0 15  0  0 15  0
    0  13 13 13 0  0  0  17 17 17 0  0  0 11 11 11  0  0  0 15 15 15 15  0
    0  13 0  0  0  0  0  17 0  0  0  0  0 11  0  0  0  0  0 15  0  0  0  0
    0  13 0  0  0  0  0  17 17 17 17 0  0 11 11 11 11  0  0 15  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

You can barely make it out but the image seems to be displaying the word 'FEEP'. Here, 11, 13, 15 and 17 are different values that pixel color or intensity can take. But let's say this picture is greyscale and the different numbers represent gradations of black and white from 0 to 255, lightest to darkest.

Now FEEP is made of ones, which will make more sense for the pipeline and the model at the end.

    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  1  1  1  1  0  0  1  1  1  1  0  0  1  1  1  1  0  0  1  1  1  1  0
    0  1  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  0  0  1  0  0  1  0
    0  1  1  1  0  0  0  1  1  1  0  0  0  1  1  1  0  0  0  1  1  1  1  0
    0  1  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  0
    0  1  0  0  0  0  0  1  1  1  1  0  0  1  1  1  1  0  0  1  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

In our example, we display the image as being a two-dimensional matrix, but what's stopping us from representing it as a single row vector? It's tape for the Turing Machine, and this row vector is suitable food for a deep learning algorithm.

If we wanted to detect whether or not this says 'FEEP' or not, our labels would be similar to IRIS.

|FEEP|unFEEP|
|-----|------|
| 1   |   0  |

### Iterators

DL4J uses an abstraction called an iterator to simplify data fetching. 

The iterator systematically examines the contents of a container or an object one by one. In this case, we have a dataset with rows. (You can also look at them in batches.) Here's the code for implementing an iterator:

    public class IrisDataSetIterator extends BaseDatasetIterator {

	private static final long serialVersionUID = -2022454995728680368L;

	    public IrisDataSetIterator(int batch,int numExamples) {
		    super(batch,numExamples,new **IrisDataFetcher**());
	   }
    }

Next step: The **Fetcher** is where the real work comes in.

    public class **IrisDataFetcher** extends BaseDataFetcher {
          @Override
	  public void fetch(int numExamples) {
 		 …
		 // Implement your own intelligent way of 
		 // getting data from here and eventually call
		 // the function below
		 initializeCurrFromList(List<DataSet>)
	   }
    }

OK, let's consider how we implement an iterator:

DL4J uses an object called **DataSet** to load data into a neural network. It's an easy way to store data we want to predict on and the labels associated with it. First and Second are both NDArrays. 

|First (data to predict)| Second (outcome, or labels)|
|-----------------|---------------------------------|
|ribbit|frog|
|ribbit|frog|
|bark|dog|
|meow|cat|

Contained within a DataSet object are two NDArrays, a fundamental object that DL4J uses for numeric computation such as linear algebra and matrix manipulation. N-dimensional arrays are scalable, multi-dimensional arrays suitable for sophisticated math and frequently used in scientific computing.

One NDArrays will hold the data’s attributes, while the other holds the label. 

Most programmers are familiar with datasets contained in the CSV (comma-separated value) file type, and the IRIS dataset is no exception. Let’s see how to parse an Iris CSV and put it into objects DL4J can understand. 

    File f = new File(“Iris.dat”);
    InputStream fis = new FileInputStream(f);

    List<String> lines = org.apache.commons.io.IOUtils.readLines(fis);
    INDArray data = Nd4j.ones(to, 4);
    List<String> outcomeTypes = new ArrayList<>();
    double[][] outcomes = new double[lines.size()][3];

Let’s break this down: iris.dat has the data we need in a CSV file.

We use IOUtils, an Apache library, to help read the data from a file stream. Please note that readLines will copy all of it into memory (generally you shouldn’t do that in production). Instead consider a **BufferedReader** object.

*data* will hold our raw numeric data, and *outcomeTypes* will be a sort of map that contains our labels. *completedData*, the outcome of the code below, will contain all of our data, including binarized labels.

*outcomes* itself will be a two-dimensional array of doubles that has as many rows as we have records (i.e. lines in iris.dat), and as many columns as we have labels (i.e. the three species of iris). This will contain our binarized labels.

Take a look at this code segment

    1  for(int i = from; i < to; i++) {
    2   String line = lines.get(i);
    3   String[] split = line.split(",");
    4
    5    // turn the 4 numeric values into doubles and add them
    6   double[] vector = new double[4];
    7   for(int i = 0; i < 4; i++)
    8        vector[i] = Double.parseDouble(line[i]);
    9        data.putRow(row,Nd4j.create(vector));
	10
    11  String outcome = split[split.length - 1];
    12  if(!outcomeTypes.contains(outcome))
    13      outcomeTypes.add(outcome);
	14
    15  double[] rowOutcome = new double[3];
    16  rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
    17  outcomes[i] = rowOutcome;
    18  }
    19
    20 DataSet completedData = new DataSet(data, Nd4j.create(outcomes));

OK, break it down.

Line 3: Since we’re dealing with CSV data, we can just use *split* to tokenize on each comma and store the data in the String array **split**.

Lines 6-9: Our String objects are actually strings of numbers. That is, instead of a double of 1.5, we have a String object with the characters “1.5”. We’ll create a temporary array called vector and store the strings there for use later. 

Line 11-14: We need to get the labels. We do that by taking the last element of our String array. Now we can think about binarizing the label. To do that, we’ll collect all the labels in the list outcomeTypes, which is our bridge to the next step.

Lines 16-18: We start to binarize the labels with our outcomeTypes list. Each label has a certain position, or index, and we’ll use that index number to map onto the label row we make here. So, if *I. Setosa* is the label, we’ll put it at the end of the outcomeTypes list. From there, we’ll create a new label row, three elements in size, and mark the corresponding position in rowOutcome as 1, with 0 for all other label cells. Finally, we save rowOutcome into the 2D array outcomes that we made earlier. 

By the time we finish, there's a new row with a numeric representation of *I. setosa* that looks like

|Sepal length|Sepal width|Petal length|Petal width|I. setosa|I. virginica|I. versicolor|
|--|--|--|--|--|--|--|
|5.1|3.5|1.4|0.2|1|0|0|

The words you see in the upper boxes are there for human eyes -- the machines do not need them. The numbers in lower boxes are fed as a vector for further data processing. That is to say, the bottom row is *vectorized data*. Numbers in a line. 

Line 21: Now we can start to think about packaging the data for DL4J. To do that, we create a single **DataSet** object with the data we want to work with and the accompanying, binarized labels.

Finally, we’ll return the list completedData, a dataset our DBN can work with. 

When you actually get to using the iterator, what you want to do is to use the .next() call to get a **DataSet** object.

### Model Data Ingestion

Here's how to create a deep-belief network, or DBN, to classify our inputs. With DL4J, it involves several steps. 

First, we create a configuration object:

    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
    .hiddenUnit(RBM.HiddenUnit.RECTIFIED).momentum(5e-1f) //this expresses decimals as floats. Remember e?
        .visibleUnit(RBM.VisibleUnit.GAUSSIAN).regularization(true)
        .regularizationCoefficient(2e-4f).dist(Distributions.uniform(gen))
        .activationFunction(Activations.tanh()).iterations(10000)
        .weightInit(WeightInit.DISTRIBUTION)
    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
        .learningRate(1e-3f).nIn(4).nOut(3).build();

This has everything that our DBN classifier will need. As you can see, there are a lot of parameters, or ‘knobs’, that you will learn to adjust over time to improve your nets’ performance. These are the pedals, clutch and steering wheel attached to your deep-learning engine. 

These include but are not limited to: the amount of momentum, regularization (yes or no) and its coefficient, the number of iterations, the velocity of the learning rate, the number of output nodes, and the transforms attached to each visible and hidden node layer (such as Gaussian or Rectified). 

We also need a random number generator object

        RandomGenerator gen = new MersenneTwister(123);

Finally, we create the DBN itself, which we’ll just call dbn.

    DBN dbn = new DBN.Builder().configure(conf)
        .hiddenLayerSizes(new int[]{3})
        .build();
      dbn.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
   
    dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

Let’s analyze the code above. On the first line, we take the Configuration object we called ‘conf’ and we pass it in as a parameter to the dbn. Then we specify the hidden layer size. We can do so for each layer of the array. In this case, there is a single hidden layer three nodes long. 

Now we can prepare the DataSet object we made earlier. We’ll put it in a separate function called loadIris().

    DataSet ourDataSet = loadIris(0, 150);
    ourDataSet.normalizeZeroMeanZeroUnitVariance();
    dbn.fit(ourDataSet);

Note line #2. In many machine-learning models, it’s important to normalize the data to ensure that outliers don’t distort the model. (The normalization of numbers means adjusting values that may be measured on different scales (in tens or hundreds or millions) to a notionally common scale, say, between 0 and 1. You can only compare apples to apples if everything is apple-scale.

Finally, we call fit to train the model on the data set. 

Training a model on a dataset, your algorithm learns to extract the features of the data which are useful signals for classifying the target input. 

Training is basically 1) the repeated attempt to classify inputs based on various machine-extracted features; 2) the comparison of those guesses with the real answers in a test set; and 3) the reward or punishment of the net as it moves closer to or farther from the correct answers. 

With sufficient training, this net could be unleashed to classify unsupervised data about irises with a solid expectation of what its accuracy would be. 

You should see some output from running that last line, if debugs are turned on. 

**Evaluating our results**

Consider the code snippet below that would come after our fit() call.

    Evaluation eval = new Evaluation();
    INDArray output = d.output(next.getFeatureMatrix());
    eval.eval(next.getLabels(),output);
    System.out.printf("Score: %s\n", eval.stats());
    log.info("Score " + eval.stats());

DL4J uses an **Evaluation** object that collects statistics about the model’s performance. The INDArray output is created by a chained call of DataSet.getFeatureMatrix() and output. The getFeatureMatrix call returns an NDArray of all of our data inputs that is fed into output(). This method will label the probabilities of an input, in this case our feature matrix. *eval* itself just collects misses and hits of predicted and real outcomes of the model. 

The *Evaluation* object itself contains many useful calls, such as *f1()*. This method esTimates the accuracy of a model in the form of a probability (The f1 score below means the model considers its ability to classify to be about 77 percent accurate). Other methods include precision(), which tells us how well a model can reliably predict the same results given the same input, and recall() which tells us how many of the correct results we retrieved.

In this example, we have the following

    Actual Class 0 was predicted with Predicted 0 with count 50 times

    Actual Class 1 was predicted with Predicted 1 with count 1 times

     Actual Class 1 was predicted with Predicted 2 with count 49 times

    Actual Class 2 was predicted with Predicted 2 with count 50 times

    ====================F1Scores========================
                     0.767064393939394
    ====================================================

We deliberately did not tune our model well (tweaking the knobs is extra credit). But this is not bad for a first pass.  Most work with neural nets involves fine tuning the parameters.

# Distributed DL4J

DL4J can be used to build up a cluster of workers to speed up network training. If you have engineered a lot of features then this will be necessary. Models with thousands, millions or *billions* of features are not unheard of at places like Google.

### The moving parts 

DL4J cluster have a few moving parts 

* Hazelcast
* Akka
* Zookeeper
* DL4J
	* Blas, JBlas, JCuda, JCublas, ND4J
* Amazon Web Service

### Hazelcast

Hazelcast is an in-memory data grid that can be distributed across many nodes. Declaring global objects in Hazelcast is  simple. In DL4J, you can see our use in **BaseHazelCastStateTracker.java**. Specifically, look at the constructor for BaseHazelCastStateTracker. 

    private transient HazelcastInstance h;
    private volatile transient IMap<String,Boolean> workerEnabled;
    public final static String WORKER_ENABLED = "workerenabled";
    ...
    h = Hazelcast.newHazelcastInstance(config);
    ...
    workerEnabled = h.getMap(WORKER_ENABLED);

Here we can see that getting an object, in this case, **workerEnabled**, is fairly simple. 

For DL4J, the StateTracker is used to keep track of a list of workers, the batch size, and other necessary components. Mostly, the StateTracker is used to toss **DataSet** objects around to different workers to train on.
    
### Akka

While Hazelcast offers parallelism, we now need to think about concurrency, too. For that we can use Akka.
Akka itself uses the concept of the *actor pattern* to implement threads and mercifully hide the complexity away from programmers.

Looking through DL4J code, you will see references to WorkerActor and MasterActor. In our system, you don't generally have to worry about implementing your own WorkerActor. Indeed, we have ActorNetworkRunnerApp to help you. More on that later.

You only need to worry about running your own Master. It's as simple as something like

		1  ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
	    2  Conf c = new Conf();
		3  NeuralNetConfiguration neuralNetConf = new NeuralNetConfiguration.Builder()
		4      .nIn(NUMBER_OF_INPUTS).nOut(NUMBER_OF_OUTPUTS).build();
		5  ...
		6  c.setMultiLayerClazz(DBN.class);
		7  ...
		8  runner.setup(c);
		9  runner.train();
		10 runner.shutdown();

Note that there are two things with the word "conf" in their names. 

**All things relating to the *system* or *cluster* are in the *Conf* object.**

**Everything you want to use relating to the neural network is in the *NeuralNetConfiguration* object.**

Line 1-4: This gets the overall setup going. There can be many other options to set between lines 3 and 4. There are many knobs and parameters to set and tune in most neural networks. (For example, the learning rate.)

Line 6: This line is very important. It tells the **Conf** object what model to distribute and run over the cluster. 

Line 8-10: Running the model is as simple as these three lines. *setup()* will run for a short while but train() will run the longest.

### Zookeeper

Remember those **Conf** and **NeuralNetworkConfiguration** objects mentioned earlier? Here are a few examples in action

	1	ZookeeperConfigurationRetriever retriever = new;
	2		ZookeeperConfigurationRetriever(host, 2181, "master");
	3	Conf conf = retriever.retrieve();
	4	String address = conf.getMasterUrl();

Lines 1-2: We retrieve our Zookeeper object.
Lines 3: Retrieve a complete Conf object.
Lines 4: Now you can use that Conf object like you would anywhere.
  
You generally don't need to worry about using these methods or objects directly, but you're welcome to if you wish.

### ND4J & Jcublas

Another component is the [ND4J](http://nd4j.org) package. Think of them as powerful arrays for scientific computing in Java.

JCublas gives ND4J the ability to use GPUs. 

While GPUs have traditionally been used for video games, they can also manipulate large arrays or matrices, including linear algebra routines that would be common in neural networks. Right now, DL4J/ND4J supports CUDA dev kit 6.0 by default. You are free to download ND4J and change it to your preferred version of CUDA.

To add CUDA spport to your project, add this to your *pom.xml* file, so Maven will build the dependencies for you.

	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-jcublas</artifactId>
		<version>0.0.3.1</version>
	</dependency>

### Running Distributed DL4J

For this exercise, consider the program MnistMultiThreadedExample that's included with DL4J. It shows how to run a cluster of multithreaded workers classifying the numeral-image set MNIST.

Everything you need is ready after you build DL4J. To do that, download DL4J from Github:

    git clone https://github.com/agibsonccc/java-deeplearning

For this exercise, let us use [IntelliJ](http://www.jetbrains.com/idea/download/).  Select "Open Project"

![enter image description here](http://i.imgur.com/lpoe46t.png)

You should see the project open up. 

![](http://i.imgur.com/wMxM3SM.png)

Now go ahead and hit **Control-N** (or **COMMAND-N** for Macs), and start to type in **MnistMultithreadedExample**. This will be the master thread worker.

![](http://i.imgur.com/JhrKhOz.png)

Now do the same for **ActorNetworkRunnerApp**. This is the worker thread.

Now, go to Run > Edit Configurations

You should now see a window like this

![](http://i.imgur.com/IdAVC46.png)

It will allow you to put debugging settings. There are two of fields for **VM Options** and **Program arguments**.

### Worker Threads

Run this command on the *host-worker* program. In this case, **ActorNetworkRunnerApp**

* **VM Options**:  

	-server -XX:+UseTLAB -XX:+UseParNewGC 	-XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 	-XX:CMSInitiatingOccupancyFraction=60 -XX:+CMSParallelRemarkEnabled 
	-XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled 

* **Program Arguments**:  

	-ad 127.0.0.1 -t worker -a dbn

*ActorNetworkRunnerApp* is included with DL4J. The line above is all you generally need to execute any code you write for the master thread.

### Master Thread

Run this command on the *host-master* program. In this case, it's **MnistMultiThreadedExample**. For our example master thread, look for the file called **MnistMultiThreadedExample.java** in the DL4J distribution.

* **VM Options**: 

	-server -XX:+UseTLAB -Dhazelcast.interface=127.0.0.1 -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60 -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled 

For AWS clusters, keep in mind that you need two extra VM Options for authentication:

 	-Dorg.deeplearning4j.aws.accessKey= -- access key //(not your login)
	-Dorg.deeplearning4j.aws.accessSecret= access secret //( not your login's password)
 
Let's take a look at what these flags mean:

* -cp "lib/*" -- This is a directory that tells the JVM where our class path is located. 
* -DHazelcast.interface=  -- This tells the MasterActor which network interface to bind to. If you have a system with multiple network interfaces then you may need to specify which one. Otherwise,  you can leave it out.
* DistributedDBN -- This tells the JVM what the name of our class is named. This class should have the aforementioned calls to ActorNetworkRunner.
* -ad -- This tells the worker thread where to connect to go look for work.

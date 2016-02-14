---
title: Distributed DL4J
layout: default
---

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

Please see your [quick start](http://deeplearning4j.org/quickstart.html) for more information on adding the necessary dependencies.

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

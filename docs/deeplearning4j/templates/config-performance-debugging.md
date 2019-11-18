---
title: Deeplearning4j and ND4J - Debugging Performance Issues
short_title: Performance Issues Debugging
description: How to debug performance issues in Deeplearning4j and ND4J
category: Configuration
weight: 11
---

# DL4J and ND4J: How to Debugging Performance Issues

This page is a how-to guide for debugging performance issues encountered when training neural networks with Deeplearning4j.
Much of the information also applies to debugging performance issues encountered when using ND4J.

Deeplearning4j and ND4J provide excellent performance in most cases (utilizing optimized c++ code for all numerical operations as well as high performance libraries such as NVIDIA cuDNN and Intel MKL). However, sometimes bottlenecks or misconfiguration issues may limit performance to well below the maximum. This page is intended to be a guide to help users identify the cause of poor performance, and provide steps to fix these issues.

Performance issues may include:
1. Poor CPU/GPU utilization
2. Slower than expected training or operation execution

To start, here's a summary of some possible causes of performance issues:
1. Wrong ND4J backend is used (for example, CPU backend when GPU backend is expected)
2. Not using cuDNN when using CUDA GPUs
3. ETL (data loading) bottlenecks
4. Garbage collection overheads
5. Small batch sizes
6. Multi-threaded use of MultiLayerNetwork/ComputationGraph for inference (not thread safe)
7. Double precision floating point data type used when single precision should be used
8. Not using workspaces for memory management (enabled by default)
9. Poorly configured network
10. Layer or operation is CPU-only
11. CPU: Lack of hardware support for modern AVX etc extensions
12. Other processes using CPU or GPU resources
13. CPU: Lack of configuration of OMP_NUM_THREADS when using many models/threads simultaneously

Finally, this page has a short section on [Debugging Performance Issues with JVM Profiling](#profiling)

## Step 1: Check if correct backend is used

ND4J (and by extension, Deeplearning4j) can perform computation on either the CPU or GPU.
The device used for computation is determined by your project dependencies - you include ```nd4j-native-platform``` to use CPUs for computation or ```nd4j-cuda-x.x-platform``` to use GPUs for computation (where ```x.x``` is your CUDA version - such as 9.2, 10.0 etc).

It is straightforward to check which backend is used. ND4J will log the backend upon initialization.

For CPU execution, you will expect output that looks something like:
```
o.n.l.f.Nd4jBackend - Loaded [CpuBackend] backend
o.n.n.NativeOpsHolder - Number of threads used for NativeOps: 8
o.n.n.Nd4jBlas - Number of threads used for BLAS: 8
o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CPU]; OS: [Windows 10]
o.n.l.a.o.e.DefaultOpExecutioner - Cores: [16]; Memory: [7.1GB];
o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [MKL]
```

For CUDA execution, you would expect the output to look something like:
```
13:08:09,042 INFO  ~ Loaded [JCublasBackend] backend
13:08:13,061 INFO  ~ Number of threads used for NativeOps: 32
13:08:14,265 INFO  ~ Number of threads used for BLAS: 0
13:08:14,274 INFO  ~ Backend used: [CUDA]; OS: [Windows 10]
13:08:14,274 INFO  ~ Cores: [16]; Memory: [7.1GB];
13:08:14,274 INFO  ~ Blas vendor: [CUBLAS]
13:08:14,274 INFO  ~ Device Name: [TITAN X (Pascal)]; CC: [6.1]; Total/free memory: [12884901888]
```

Pay attention to the ```Loaded [X] backend``` and ```Backend used: [X]``` messages to confirm that the correct backend is used.
If the incorrect backend is being used, check your program dependencies to ensure tho correct backend has been included.


## Step 2: Check for cuDNN

If you are using CPUs only (nd4j-native backend) then you can skip to step 3 as cuDNN only applies when using NVIDIA GPUs (```nd4j-cuda-x.x-platform``` dependency).

cuDNN is NVIDIA's library for accelerating neural network training on NVIDIA GPUs.
Deeplearning4j can make use of cuDNN to accelerate a number of layers - including ConvolutionLayer, SubsamplingLayer, BatchNormalization, Dropout, LocalResponseNormalization and LSTM. When training on GPUs, cuDNN should always be used if possible as it is usually much faster than the built-in layer implementations.

Instructions for configuring CuDNN can be found [here](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn).
In summary, include the ```deeplearning4j-cuda-x.x``` dependency (where ```x.x``` is your CUDA version - such as 9.2 or 10.0). The network configuration does not need to change to utilize cuDNN - cuDNN simply needs to be available along with the deeplearning4j-cuda module.


**How to determine if CuDNN is used or not**

Not all DL4J layer types are supported in cuDNN. DL4J layers with cuDNN support include ConvolutionLayer, SubsamplingLayer, BatchNormalization, Dropout, LocalResponseNormalization and LSTM.

To check if cuDNN is being used, the simplest approach is to look at the log output when running inference or training:
If cuDNN is NOT available when you are using a layer that supports it, you will see a message such as:
```
o.d.n.l.c.ConvolutionLayer - cuDNN not found: use cuDNN for better GPU performance by including the deeplearning4j-cuda module. For more information, please refer to: https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn
java.lang.ClassNotFoundException: org.deeplearning4j.nn.layers.convolution.CudnnConvolutionHelper
	at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:335)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
	at java.lang.Class.forName0(Native Method)
```

If cuDNN is available and was loaded successfully, no message will be logged.

Alternatively, you can confirm that cuDNN is used by using the following code:
```
MultiLayerNetwork net = ...
LayerHelper h = net.getLayer(0).getHelper();    //Index 0: assume layer 0 is a ConvolutionLayer in this example
System.out.println("Layer helper: " + (h == null ? null : h.getClass().getName()));
```
Note that you will need to do at least one forward pass or fit call to initialize the cuDNN layer helper.

If cuDNN is available and was loaded successfully, you will see the following printed:
```
Layer helper: org.deeplearning4j.nn.layers.convolution.CudnnConvolutionHelper
```
whereas if cuDNN is not available or could not be loaded successfully (you will get a warning or error logged also):
```
Layer helper: null
```



## Step 3: Check for ETL (Data Loading) Bottlenecks

Neural network training requires data to be in memory before training can proceed. If the data is not loaded fast enough, the network will have to wait until data is available.
DL4J uses asynchronous prefetch of data to improve performance by default. Under normal circumstances, this asynchronous prefetching means the network should never be waiting around for data (except on the very first iteration) - the next minibatch is loaded in another thread while training is proceeding in the main thread.

However, when data loading takes longer than the iteration time, data can be a bottleneck. For example, if a network takes 100ms to perform fitting on a single minibatch, but data loading takes 200ms, then we have a bottleneck: the network will have to wait 100ms per iteration (200ms loading - 100ms loading in parallel with training) before continuing the next iteration.
Conversely, if network fit operation was 100ms and data loading was 50ms, then no data loading bottleck will occur, as the 50ms loading time can be completed asynchronously within one iteration.

**How to check for ETL / data loading bottlenecks**

The way to identify ETL bottlenecks is simple: add PerformanceListener to your network, and train as normal.
For example:
```
MultiLayerNetwork net = ...
net.setListeners(new PerformanceListener(1));       //Logs ETL and iteration speed on each iteration
```
When training, you will see output such as:
```
.d.o.l.PerformanceListener - ETL: 0 ms; iteration 16; iteration time: 65 ms; samples/sec: 492.308; batches/sec: 15.384; 
```
The above output shows that there is no ETL bottleneck (i.e., ```ETL: 0 ms```). However, if ETL time is greater than 0 consistently (after the first iteration), an ETL bottleneck is present.

**How to identify the cause of an ETL bottleneck**

There are a number of possible causes of ETL bottlenecks. These include (but are not limited to):
* Slow hard drives
* Network latency or throughput issues (when reading from remote or network storage)
* Computationally intensive or inefficient ETL (especially for custom ETL pipelines)

One useful way to get more information is to perform profiling, as described in the [profiling section](#profiling) later in this page.
For custom ETL pipelines, adding logging for the various stages can help. Finally, another approach to use a process of elimination - for example, measuring the latency and throughput of reading raw files from disk or from remote storage vs. measuring the time to actually process the data from its raw format.

## Step 4: Check for Garbage Collection Overhead

Java uses garbage collection for management of on-heap memory (see [this link](https://stackify.com/what-is-java-garbage-collection/) for example for an explanation).
Note that DL4J and ND4J use off-heap memory for storage of all INDArrays (see the [memory page](https://deeplearning4j.org/docs/latest/deeplearning4j-config-memory) for details). 

Even though DL4J/ND4J array memory is off-heap, garbage collection can still cause performance issues.

In summary:
* Garbage collection will sometimes (temporarily and briefly) pause/stop application execution ("stop the world")
* These GC pauses slow down program execution
* The overall performance impact of GC pauses depends on both the frequency of GC pauses, and the duration of GC pauses
* The frequency is controllable (in part) by ND4J, using ```Nd4j.getMemoryManager().setAutoGcWindow(10000);``` and ```Nd4j.getMemoryManager().togglePeriodicGc(false);```
* Not every GC event is caused by or controlled by the above ND4J configuration.

In our experience, garbage collection time depends strongly on the number of objects in the JVM heap memory.
As a rough guide:
* Less than 100,000 objects in heap memory: short GC events (usually not a performance problem)
* 100,000-500,000 objects: GC overhead becomes noticeable, often in the 50-250ms range per full GC event
* 500,000 or more objects: GC can be a bottleneck if performed frequently. Performance may still be good if GC events are infrequent (for example, every 10 seconds or less).
* 10 million or more objects: GC is a major bottleneck even if infrequently called, with each full GC takes multiple seconds

**How to configure ND4J garbage collection settings**

In simple terms, there are two settings of note:
```
Nd4j.getMemoryManager().setAutoGcWindow(10000);             //Set to 10 seconds (10000ms) between System.gc() calls
Nd4j.getMemoryManager().togglePeriodicGc(false);            //Disable periodic GC calls
```

If you suspect garbage collection overhead is having an impact on performance, try changing these settings.
The main downside to reducing the frequency or disabling periodic GC entirely is when you are not using [workspaces](https://deeplearning4j.org/docs/latest/deeplearning4j-config-workspaces), though workspaces are enabled by default for all neural networks in Deeplearning4j.


Side note: if you are using DL4J for training on Spark, setting these values on the master/driver will not impact the settings on the worker. Instead, see [this guide](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-howto#gc).

**How to determine GC impact using PerformanceListener**

*NOTE: this feature was added after 1.0.0-beta3 and will be available in future releases*
To determine the impact of garbage collection using PerformanceListener, you can use the following:

```
int listenerFrequency = 1;
boolean reportScore = true;
boolean reportGC = true;
net.setListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));
```

This will report GC activity:
```
o.d.o.l.PerformanceListener - ETL: 0 ms; iteration 30; iteration time: 17 ms; samples/sec: 588.235; batches/sec: 58.824; score: 0.7229335801186025; GC: [PS Scavenge: 2 (1ms)], [PS MarkSweep: 2 (24ms)];
```
The garbage collection activity is reported for all available garbage collectors - the ```GC: [PS Scavenge: 2 (1ms)], [PS MarkSweep: 2 (24ms)]``` means that garbage collection was performed 2 times since the last PerformanceListener reporting, and took 1ms and 24ms total respectively for the two GC algorithms, respectively.

Keep in mind: PerformanceListener reports GC events every N iterations (as configured by the user). Thus, if PerformanceListener is configured to report statistics every 10 iterations, the garbage collection stats would be for the period of time corresponding to the last 10 iterations.

**How to determine GC impact using ```-verbose:gc```**

Another useful tool is the ```-verbose:gc```, ```-XX:+PrintGCDetails``` ```-XX:+PrintGCTimeStamps``` command line options.
For more details, see [Oracle Command Line Options](https://www.oracle.com/technetwork/java/javase/clopts-139448.html#gbmpt) and [Oracle GC Portal Documentation](https://www.oracle.com/technetwork/articles/javase/gcportal-136937.html)

These options can be passed to the JVM on launch (when using ```java -jar``` or ```java -cp```) or can be added to IDE launch options (for example, in IntelliJ: these should be placed in the "VM Options" field in Run/Debug Configurations - see [Setting Configuration Options](https://www.jetbrains.com/help/idea/setting-configuration-options.html))

When these options are enabled, you will have information reported on each GC event, such as:
```
5.938: [GC (System.gc()) [PSYoungGen: 5578K->96K(153088K)] 9499K->4016K(502784K), 0.0006252 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
5.939: [Full GC (System.gc()) [PSYoungGen: 96K->0K(153088K)] [ParOldGen: 3920K->3911K(349696K)] 4016K->3911K(502784K), [Metaspace: 22598K->22598K(1069056K)], 0.0117132 secs] [Times: user=0.02 sys=0.00, real=0.01 secs]
```

This information can be used to determine the frequency, cause (System.gc() calls, allocation failure, etc) and duration of GC events.


**How to determine GC impact using a profiler**

An alternative approach is to use a profiler to collect garbage collection information.

For example, [YourKit Java Profiler](https://www.yourkit.com) can be used to determine both the frequency and duration of garbage collection - see [Garbage collection telemetry](https://www.yourkit.com/docs/java/help/garbage_collection.jsp) for more details.

[Other tools](https://www.cubrid.org/blog/how-to-monitor-java-garbage-collection/), such as VisualVM can also be used to monitor GC activity.


**How to determine number (and type) of JVM heap objects using memory dumps**

If you determine that garbage collection is a problem, and suspect that this is due to the number of objects in memory, you can perform a heap dump.

To perform a heap dump:
* Step 1: Run your program
* Step 2: While running, determine the process ID
    - One approach is to use jps:
        - For basic details, run ```jps``` on the command line. If jps is not on the system PATH, it can be found (on Windows) at ```C:\Program Files\Java\jdk<VERSION>\bin\jps.exe```
        - For more details on each process, run ```jps -lv``` instead
    - Alternatively, you can use the ```top``` command on Linux or Task Manager (Windows) to find the PID (on Windows, the PID column may not be enabled by default)
* Step 3: Create a heap dump using ```jmap -dump:format=b,file=file_name.hprof 123``` where ```123``` is the process id (PID) to create the heap dump for

A number of alternatives for generating heap dumps can be found [here](https://www.yourkit.com/docs/java/help/hprof_snapshots.jsp).

After a memory dump has been collected, it can be opened in tools such as YourKit profiler and VisualVM to determine the number, type and size of objects.
With this information, you should be able to pinpoint the cause of the large number of objects and make changes to your code to reduce or eliminate the objects that are causing the garbage collection overhead.

## Step 5: Check Minibatch Size

Another common cause of performance issues is a poorly chosen minibatch size.
A minibatch is a number of examples used together for one step of inference and training. Minibatch sizes of 32 to 128 are commonly used, though smaller or larger are sometimes used.

In summary:
* If minibatch size is too small (for example, training or inference with 1 example at a time), poor hardware utilization and lower overall throughput is expected
* If minibatch size is too large
    - Hardware utilization will usually be good
    - Iteration times will slow down
    - Memory utilization may be too high (leading to out-of-memory errors)

For inference, avoid using minibatch size of 1, as throughput will suffer. Unless there are strict latency requirements, you should use larger minibatch sizes as this will give you the best hardware utilization and hence throughput, and is especially important for GPUs.

For training, you should never use a minibatch size of 1 as overall performance and hardware utilization will be reduced. Network convergence may also suffer. Start with a minibatch size of 32-128, if memory will allow this to be used.

For serving predictions in multi-threaded applications (such as a web server), [ParallelInference](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-scaleout/deeplearning4j-scaleout-parallelwrapper/src/main/java/org/deeplearning4j/parallelism/ParallelInference.java) should be used.


## Step 6: Ensure you are not using a single MultiLayerNetwork/ComputationGraph for inference from multiple threads

MultiLayerNetwork and ComputationGraph are not considered thread-safe, and should not be used from multiple threads.
That said, most operations such as fit, output, etc use synchronized blocks. These synchronized methods should avoid hard to understand exceptions (race conditions due to concurrent use), they will limit throughput to a single thread (though, note that native operation parallelism will still be parallelized as normal).
In summary, using the one network from multiple threads should be avoided as it is not thread safe and can be a performance bottleneck.


For inference from multiple threads, you should use one model per thread (as this avoids locks) or for serving predictions in multi-threaded applications (such as a web server), use [ParallelInference](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-scaleout/deeplearning4j-scaleout-parallelwrapper/src/main/java/org/deeplearning4j/parallelism/ParallelInference.java).

## Step 7: Check Data Types

As of 1.0.0-beta3 and earlier, ND4J has a global datatype setting that determines the datatype of all arrays.
The default value is 32-bit floating point. The data type can be set using ```Nd4j.setDataType(DataBuffer.Type.FLOAT);``` for example.

For best performance, this value should be left as its default. If 64-bit floating point precision (double precision) is used instead, performance can be significantly reduced, especially on GPUs - most consumer NVIDIA GPUs have very poor double precision performance (and half precision/FP16). On Tesla series cards, double precision performance is usually much better than for consumer (GeForce) cards, though is still usually half or less of the single precision performance.
Wikipedia has a summary of the single and double precision performance of NVIDIA GPUs [here](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units).

Performance on CPUs can also be reduced for double precision due to the additional memory batchwidth requirements vs. float precision.

You can check the data type setting using:
```
System.out.println("ND4J Data Type Setting: " + Nd4j.dataType());
```

## Step 8: Check workspace configuration for memory management (enabled by default)

For details on workspaces, see the [workspaces page](https://deeplearning4j.org/docs/latest/deeplearning4j-config-workspaces).

In summary, workspaces are enabled by default for all Deeplearning4j networks, and enabling them improves performance and reduces memory requirements.
There are very few reasons to disable workspaces.

You can check that workspaces are enabled for your MultiLayerNetwork using:
```
System.out.println("Training workspace config: " + net.getLayerWiseConfigurations().getTrainingWorkspaceMode());
System.out.println("Inference workspace config: " + net.getLayerWiseConfigurations().getInferenceWorkspaceMode());
```
or for a ComputationGraph using:
```
System.out.println("Training workspace config: " + cg.getConfiguration().getTrainingWorkspaceMode());
System.out.println("Inference workspace config: " + cg.getConfiguration().getInferenceWorkspaceMode());
```

You want to see the output as ```ENABLED``` output for both training and inference.
To change the workspace configuration, use the setter methods, for example: ```net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.ENABLED);```


## Step 9: Check for a badly configured network or network with layer bottlenecks

Another possible cause (especially for newer users) is a poorly designed network.
A network may be poorly designed if:
* It has too many layers. A rough guideline:
    - More than about 100 layers for a CNN may be too many
    - More than about 10 layers for a RNN/LSTM network may be too many
    - More than about 20 feed-forward layers may be too many for a MLP
* The input/activations are too large
    - For CNNs, inputs in the range of 224x224 (for image classification) to 600x600 (for object detection and segmentation) are used. Large image sizes (such as 500x500) are computationally demanding, and much larger than this should be considered too large in most cases.
    - For RNNs, the sequence length matters. If you are using sequences longer than a few hundred steps, you should use [truncated backpropgation through time](https://deeplearning4j.org/docs/latest/deeplearning4j-nn-recurrent#tbptt) if possible. 
* The output number of classes is too large
    - Classification with more than about 10,000 classes can become a performance bottleneck with standard softmax output layers
* The layers are too large
    - For CNNs, most layers have kernel sizes in the range 2x2 to 7x7, with channels equal to 32 to 1024 (with larger number of channels appearing later in the network). Much larger than this may cause a performance bottleneck.
    - For MLPs, most layers have at most 2048 units/neurons (often much smaller). Much larger than this may be too large.
    - For RNNs such as LSTMs, layers are typically in the range of 128 to 512, though the largest RNNs may use around 1024 units per layer.
* The network has too many parameters
    - This is usually a consequence of the other issues already mentioned - too many layers, too large input, too many output classes
    - For comparison, less than 1 million parameters would be considered small, and more than about 100 million parameters would be considered very large.
    - You can check the number of parameters using ```MultiLayerNetwork/ComputationGraph.numParams()``` or ```MultiLayerNetwork/ComputationGraph.summary()```

Note that these are guidelines only, and some reasonable network may exceed the numbers specified here. Some networks can become very large, such as those commonly used for imagenet classification or object detection. However, in these cases, the network is usually carefully designed to provide a good tradeoff between accuracy and computation time.

If your network architecture is significantly outside of the guidelines specified here, you may want to reconsider the design to improve performance.


## Step 10: Check for CPU-only ops (when using GPUs)

If you are using CPUs only (nd4j-native backend), you can skip this step, as it only applies when using the GPU (nd4j-cuda) backend.

As of 1.0.0-beta3, a handful of recently added operations do not yet have GPU implementations. Thus, when these layer are used in a network, they will execute on CPU only, irrespective of the nd4j-backend used. GPU support for these layers will be added in an upcoming release.

The layers without GPU support as of 1.0.0-beta3 include:
* Convolution3D
* Upsampling1D/2D/3D
* Deconvolution2D
* LocallyConnected1D/2D
* SpaceToBatch
* SpaceToDepth

Unfortunately, there is no workaround or fix for now, until these operations have GPU implementations completed.


## Step 11: Check CPU support for hardware extensions (AVX etc)

If you are running on a GPU, this section does not apply.

When running on older CPUs or those that lack modern AVX extensions such as AVX2 and AVX512, performance will be reduced compared to running on CPUs with these features.
Though there is not much you can do about the lack of such features, it is worth knowing about if you are comparing performance between different CPU models.

In summary, CPU models with AVX2 support will perform better than those without it; similarly, AVX512 is an improvement over AVX2.

For more details on AVX, see the [Wikipedia AVX article](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)


## Step 12: Check other processes using CPU or GPU resources

Another obvious cause of performance issues is other processes using CPU or GPU resources.

For CPU, it is straightforward to see if other processes are using resources using tools such as ```top``` (for Linux) or task managed (for Windows).

For NVIDIA CUDA GPUs, nvidia-smi can be used. nvidia-smi is usually installed with the NVIDIA display drivers, and (when run) shows the overall GPU and memory utilization, as well as the GPU utilization of programs running on the system.

On Linux, this is usually on the system path by default.
On Windows, it may be found at ```C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi```


## Step 13: Check OMP_NUM_THREADS performing concurrent inference using CPU in multiple threads simultaneously

If you are using GPUs (nd4j-cuda backend), you can skip this section.

One issue to be aware of when running multiple DL4J networks (or ND4J operations generally) concurrently in multiple threads is the OpenMP number of threads setting.
In summary, in ND4J we use OpenMP pallelism at the c++ level to increase operation performance. By default, ND4J will use a value equal to the number of physical CPU cores (*not logical cores*) as this will give optimal performance 

This also applies if the CPU resources are shared with other computationally demanding processes.

In either case, you may see better overall throughput by reducing the number of OpenMP threads by setting the OMP_NUM_THREADS environment variable - see [ND4JEnvironmentVars](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-common/src/main/java/org/nd4j/config/ND4JEnvironmentVars.java) for details.

One reason for reducing OMP_NUM_THREADS improving overall performance is due to reduced [cache thrashing](https://en.wikipedia.org/wiki/Thrashing_(computer_science)).


# <a href="profiling">Debugging Performance Issues with JVM Profiling</a>

Profiling is a process whereby you can trace how long each method in your code takes to execute, to identify and debug performance bottlenecks.

A full guide to profiling is beyond the scope of this page, but the summary is that you can trace how long each method takes to execute (and where it is being called from) using a profiling tool. This information can then be used to identify bottlenecks (and their causes) in your program.


## How to Perform Profiling

Multiple options are available for performing profiling locally.
We suggest using either [YourKit Java Profiler](https://www.yourkit.com/java/profiler/features/) or [VisualVM](https://visualvm.github.io/) for profiling.

The YourKit profiling documentation is quite good. To perform profiling with YourKit:
* Install and start YourKit Profiler
* Start your application with the profiler enabled. For details, see [Running applications with the profiler](https://www.yourkit.com/docs/java/help/running_with_profiler.jsp) and [Local profiling](https://www.yourkit.com/docs/java/help/local_profiling.jsp)
    - Note that IDE integrations are available - see [IDE integration](https://www.yourkit.com/docs/java/help/ide_integration.jsp)
* Collect a snapshot and analyze

Note that YourKit provides multiple different types of profiling: Sampling, tracing, and call counting.
Each type of profiling has different pros and cons, such as accuracy vs. overhead. For more details, see [Sampling, tracing, call counting](https://www.yourkit.com/docs/java/help/cpu_intro.jsp)

VisualVM also supports profiling - see the Profiling Applications section of the [VisualVM documentation](https://visualvm.github.io/documentation.html) for more details.

## Profiling on Spark

When debugging performance issues for Spark training or inference jobs, it can often be useful to perform profiling here also.

One approach that we have used internally is to combine manual profiling settings (```-agentpath``` JVM argument) with spark-submit arguments for YourKit profiler.

To perform profiling in this manner, 5 steps are required:
1. Download YourKit profiler to a location on each worker (must be the same location on each worker) and (optionally) the driver
2. [Optional] Copy the profiling configuration onto each worker (must be the same location on each worker)
3. Create a local output directory for storing the profiling result files on each worker
4. Launch the Spark job with the appropriate configuration (see example below)
5. The snapshots will be saved when the Spark job completes (or is cancelled) to the specified directories.

For example, to perform tracing on both the driver and the workers, 
```
spark-submit
    --conf 'spark.executor.extraJavaOptions=-agentpath:/home/user/YourKit-JavaProfiler-2018.04/bin/linux-x86-64/libyjpagent.so=tracing,port=10001,dir=/home/user/yourkit_snapshots/executor/,tracing_settings_path=/home/user/yourkitconf.txt'
    --conf 'spark.driver.extraJavaOptions=-agentpath:/home/user/YourKit-JavaProfiler-2018.04/bin/linux-x86-64/libyjpagent.so=tracing,port=10001,dir=/home/user/yourkit_snapshots/driver/,tracing_settings_path=/home/user/yourkitconf.txt'
    <other spark submit arguments>
```

The configuration (tracing_settings_path) is optional. A sample tracing settings file is provided below:
```
walltime=*
adaptive=true
adaptive_min_method_invocation_count=1000
adaptive_max_average_method_time_ns=100000
```


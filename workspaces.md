As of 0.9.0 (or 0.8.1-SNAPSHOT) ND4j offers additional memory management model: Workspaces. Basically it allows you to reuse memory in cyclic workloads, without JVM Garbage Collector  use for off-heap memory tracking. In other words: at the end of Workspace loop all INDArrays memory content is invalidated.

Here are some [examples](https://github.com/deeplearning4j/dl4j-examples/blob/58cc1b56515458003fdd7b606f6451aee851b8c3/nd4j-examples/src/main/java/org/nd4j/examples/Nd4jEx15_Workspaces.java) how to use it with ND4j.
Basic idea is very simple: you can do your stuff within workspace(s), and if you need to get INDArray out of it (i.e. to move result out of workspace) - you just call INDArray.detach() method, and you'll get independent INDArray copy.

## Neural networks:
For DL4j users Workspaces give better performance just out of box. All you need to do is to choose affordable modes for training & inference of a given model

 .trainingWorkspaceMode(WorkspaceMode.SEPARATE) and/or .inferenceWorkspaceMode(WorkspaceMode.SINGLE) in your neural network configuration. 

Difference between **SEPARATE** and **SINGLE** workspaces is tradeoff between performance & memory footprint:
* **SEPARATE** is slightly slower, but uses less memory.
* **SINGLE** is slightly faster, but uses more memory.

However, it’s totally fine to use different modes for training inference: i.e. use SEPARATE for training, and use SINGLE for inference, since inference only involves feed-forward loop, without backpropagation or updaters involved.

So, with workspaces enabled all memory used during training will be reusable, and tracked without JVM GC interference.
The only exclusion is output() method, which uses workspaces (if enabled) internally for feed-forward loop, and then detaches resulting INDArray from workspaces, thus providing you with independent INDArray, which will be handled by JVM GC.

***Please note***: by default training workspace mode is set to **NONE** for now.

## ParallelWrapper & ParallelInference
For ParallelWrapper there’s also workspace mode configuration option was added, so each of trainer threads will use separate workspace, attached to designated device.

```
ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(8)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(2)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(5)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(false)

            // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            .useLegacyAveraging(false)

            .workspaceMode(WorkspaceMode.SINGLE)

            .build();
```

## Iterators:
We provide asynchronous prefetch iterators, AsyncDataSetIterator and AsyncMultiDataSetIterator, which are usually used internally. These iterators are optionally using special cyclic workspace mode for smaller memory footprint. Size of workspace in this case will be determined by memory requirements of first DataSet coming out of underlying iterator and buffer size defined by user. However workspace will be adjusted if memory requirements will change over time (i.e. if you’re using variable length time series)

***Caution***: if you’re using custom iterator or RecordReader, please make sure you’re not initializing something huge within first next() call, do that in your constructor, to avoid undesired workspace growth.

***Caution***: with AsyncDataSetIterator being used, DataSets are supposed to be used before calling for next() DataSet. So, you're not supposed to store them in any way without detach() call, because otherwise memory used for INDArrays within DataSet will be overwritten within AsyncDataSetIterator eventually.

If, for some reason, you don’t want your iterator to be wrapped into asynchronous prefetch (i.e. for debugging purposes), there’s special wrappers provided: AsyncShieldDataSetIterator and AsyncShieldMultiDataSetIterator. Basically that’s just thin wrappers that prevent prefetch.

## Garbage Collector:
If your training process uses workspaces, it’s recommended to disable (or reduce frequency of) periodic gc calls. That can be done using this:

```
// this will limit frequency of gc calls to 5000 milliseconds
Nd4j.getMemoryManager().setAutoGcWindow(5000)

// OR you could totally disable it
Nd4j.getMemoryManager().togglePeriodicGc(false);
```

So, you can put that somewhere before your `model.fit(...)` call.







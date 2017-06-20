---
title: Workspaces guide
layout: default
---

# Workspaces Guide

As of 0.9.0 (or 0.8.1-SNAPSHOT), ND4J offers an additional memory management model: workspaces. Basically, it allows you to reuse memory in cyclic workloads, without the JVM Garbage Collector, for off-heap memory tracking. In other words, at the end of the workspace loop, all INDArrays memory content is invalidated.

Here are some [examples](https://github.com/deeplearning4j/dl4j-examples/blob/58cc1b56515458003fdd7b606f6451aee851b8c3/nd4j-examples/src/main/java/org/nd4j/examples/Nd4jEx15_Workspaces.java) on how to use it with ND4J.
The basic idea is very simple. You can do your stuff within a workspace(s), and if you need to get an INDArray out of it (i.e. to move result out of the workspace), you just call the INDArray.detach() method and you'll get an independent INDArray copy.

## Neural Networks

For DL4J users, workspaces provide better performance out of box. All you need to do is choose affordable modes for the training & inference of a given model.

 `.trainingWorkspaceMode(WorkspaceMode.SEPARATE)` and/or `.inferenceWorkspaceMode(WorkspaceMode.SINGLE)` in your neural network configuration. 

The difference between **SEPARATE** and **SINGLE** workspaces is a tradeoff between the performance & memory footprint:
* **SEPARATE** is slightly slower but uses less memory.
* **SINGLE** is slightly faster but uses more memory.

However, it’s totally fine to use different modes for training & inference (i.e. use SEPARATE for training, and use SINGLE for inference, since inference only involves a feed-forward loop without backpropagation or updaters involved).

So with workspaces enabled, all memory used during training will be reusable and tracked without the JVM GC interference.
The only exclusion is the output() method which uses workspaces (if enabled) internally for the feed-forward loop. Subsequently, it detaches the resulting INDArray from the workspaces, thus providing you with independent INDArray which will be handled by the JVM GC.

***Please note***: By default, the training workspace mode is set to **NONE** for now.

## ParallelWrapper & ParallelInference
For ParallelWrapper, the workspace mode configuration option was also added. As such, each of the trainer threads will use a separate workspace attached to the designated device.

```
ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Buffer size per worker.
            .prefetchBuffer(8)

            // set number of workers equal to number of GPUs.
            .workers(2)

            // rare averaging improves performance but might reduce model accuracy
            .averagingFrequency(5)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(false)

            // 3 options here: NONE, SINGLE, SEPARATE
            .workspaceMode(WorkspaceMode.SINGLE)

            .build();
```

## Iterators

We provide asynchronous prefetch iterators, AsyncDataSetIterator and AsyncMultiDataSetIterator, which are usually used internally. These iterators optionally use a special, cyclic workspace mode for a smaller memory footprint. The size of the workspace, in this case, will be determined by the memory requirements of the first DataSet coming out of the underlying iterator whereas the buffer size is defined by the user. However, the workspace will be adjusted if memory requirements change over time (i.e. if you’re using variable length time series).

***Caution***: If you’re using a custom iterator or the RecordReader, please make sure you’re not initializing something huge within the first next() call. Do that in your constructor to avoid undesired workspace growth.

***Caution***: With AsyncDataSetIterator being used, DataSets are supposed to be used before calling the next() DataSet. You are not supposed to store them, in any way, without the detach() call. Otherwise, the memory used for INDArrays within DataSet will be overwritten within AsyncDataSetIterator eventually.

If, for some reason, you don’t want your iterator to be wrapped into an asynchronous prefetch (i.e. for debugging purposes), there’s special wrappers provided: AsyncShieldDataSetIterator and AsyncShieldMultiDataSetIterator. Basically, those are just thin wrappers that prevent prefetch.

## Evaluation

Usually, evaluation assumes use of the model.output() method which essentially, returns an INDArray detached from the workspace. In the case of regular evaluations during training, it might be better to use the built-in methods for evaluation. I.e.:
```
Evaluation eval = new Evaluation(outputNum);
ROC roceval = new ROC(outputNum);
model.doEvaluation(iteratorTest, eval, roceval);
```

This piece of code will run a single cycle over `iteratorTest`, and it will update both (or less/more if required by your needs) IEvaluation implementations without any additional INDArray allocation. 

## Garbage Collector

If your training process uses workspaces, it’s recommended to disable (or reduce frequency of) periodic gc calls. That can be done using this:

```
// this will limit frequency of gc calls to 5000 milliseconds
Nd4j.getMemoryManager().setAutoGcWindow(5000)

// OR you could totally disable it
Nd4j.getMemoryManager().togglePeriodicGc(false);
```

You can put that somewhere before your `model.fit(...)` call.

## Workspace Destruction

There are possible situations, where you're short on RAM, and might want do release all workspaces created out of your control; e.g. during evaluation or training.

That could be done like so: `Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();`

This method will destroy all workspaces that were created within the calling thread. If you've created workspaces in some external threads on your own, you can use the same method in that thread, after workspaces aren't needed any more.

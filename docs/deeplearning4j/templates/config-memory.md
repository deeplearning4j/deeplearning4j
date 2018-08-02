---
title: Memory management in DL4J and ND4J
short_title: Memory Management
description: Setting available Memory/RAM for a DL4J application
category: Configuration
weight: 1
---

# Memory Management for ND4J/DL4J: How does it work?

ND4J uses off-heap memory to store NDArrays, to provide better performance while working with NDArrays from native code such as BLAS and CUDA libraries.

"Off-heap" means that the memory is allocated outside of the JVM (Java Virtual Machine) and hence isn't managed by the JVM's garbage collection (GC). On the Java/JVM side, we only hold pointers to the off-heap memory, which can be passed to the underlying C++ code via JNI for use in ND4J operations.

To manage memory allocations, we use two approaches:

- JVM Garbage Collector (GC) and WeakReference tracking
- MemoryWorkspaces - see [Workspaces guide](https://deeplearning4j.org/workspaces) for details

Despite the differences between these two approaches, the idea is the same: once an NDArray is no longer required on the Java side, the off-heap associated with it should be released so that it can be reused later. The difference between the GC and `MemoryWorkspaces` approaches is in when and how the memory is released.

- For JVM/GC memory: whenever an INDArray is collected by the garbage collector, its off-heap memory will be deallocated, assuming it is not used elsewhere.
- For `MemoryWorkspaces`: whenever an INDArray leaves the workspace scope - for example, when a layer finished forward pass/predictions - its memory may be reused without deallocation and reallocation. This results in better performance for cyclical workloads.

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH DEEP LEARNING</a>
</p>

## Configuring Memory Limits

With DL4J/ND4J, there are two types of memory limits to be aware of and configure: The on-heap JVM memory limit, and the off-heap memory limit, where NDArrays live. Both limits are controlled via Java command-line arguments:

* `-Xms` - this defines how much memory JVM heap will use at application start.

* `-Xmx` - this allows you to specify JVM heap memory limit (maximum, at any point). Only allocated up to this amount (at the discretion of the JVM) if required.

* `-Dorg.bytedeco.javacpp.maxbytes`  - this allows you to specify the off-heap memory limit.

* `-Dorg.bytedeco.javacpp.maxphysicalbytes` - also for off-heap, this usually should be set equal to `maxbytes`.

Example: Configuring 1GB initial on-heap, 2GB max on-heap, 8GB off-heap:

```-Xms1G -Xmx2G -Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=8G```

## Gotchas: A few things to watch out for

* With GPU systems, the maxbytes and maxphysicalbytes settings currently also effectively defines the memory limit for the GPU, since the off-heap memory is mapped (via NDArrays) to the GPU - read more about this in the GPU-section below.

* For many applications, you want less RAM to be used in JVM heap, and more RAM to be used in off-heap, since all NDArrays are stored there. If you allocate too much to the JVM heap, there will not be enough memory left for the off-heap memory.

* If you get a "RuntimeException: Can't allocate [HOST] memory: xxx; threadId: yyy", you have run out of off-heap memory. You should most often use a WorkspaceConfiguration to handle your NDArrays allocation, in particular in e.g. training or evaluation/inference loops - if you do not, the NDArrays and their off-heap (and GPU) resources are reclaimed using the JVM GC, which might introduce severe latency and possible out of memory situations.

* If you don't specify JVM heap limit, it will use 1/4 of your total system RAM as the limit, by default.

* If you don't specify off-heap memory limit, x2 of JVM heap limit (Xmx) will be used by default. i.e. `-Xmx8G` will mean that 8GB can be used by JVM heap, and 16GB can be used by ND4j in off-heap.

* In limited memory environments, it's usually a bad idea to use high `-Xmx` value together with `-Xms` option. That is because doing so won't leave enough off-heap memory. Consider a 16GB system in which you set `-Xms14G`: 14GB of 16GB would be allocated to the JVM, leaving only 2GB for the off-heap memory, the OS and all other programs.

# Memory-mapped files

As of 0.9.2-SNAPSHOT, it's possible to use a memory-mapped file instead of RAM when using the `nd4j-native` backend. On one hand, it's slower then RAM, but on other hand, it allows you to allocate memory chunks in a manner impossible otherwise.

Here's sample code:

```
WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(1000000000)
                .policyLocation(LocationPolicy.MMAP)
                .build();
                
try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2")) {
    INDArray x = Nd4j.create(10000);
}
``` 
In this case, a 1GB temporary file will be created and mmap'ed, and NDArray `x` will be created in that space. Obviously, this option is mostly viable for cases when you need NDArrays that can't fit into your RAM.

# GPUs

When using GPUs, oftentimes your CPU RAM will be greater than GPU ram. When GPU RAM is less than CPU RAM, you need to monitor how much RAM is being used off-heap. You can check this based on the JavaCPP options specified above.

We allocate memory on the GPU equivalent to the amount of off-heap memory you specify. We don't use any more of your GPU than that. You are also allowed to specify heap space greater than your GPU (that's not encouraged, but it's possible). If you do so, your GPU will run out of RAM when trying to run jobs.

We also allocate off-heap memory on the CPU RAM as well. This is for efficient communicaton of CPU to GPU, and CPU accessing data from an NDArray without having to fetch data from the GPU each time you call for it.

If JavaCPP or your GPU throw an out-of-memory error (OOM), or even if your compute slows down due to GPU memory being limited, then you may want to either decrease batch size or increase the amount of off-heap memory that JavaCPP is allowed to allocate, if that's possible.

Try to run with an off-heap memory equal to your GPU's RAM. Also, always remember to set up a small JVM heap space using the `Xmx` option.

Note that if your GPU has < 2g of RAM, it's probably not usable for deep learning. You should consider using your CPU if this is the case. Typical deep-learning workloads should have 4GB of RAM *at minimum*. Even that is small. 8GB of RAM on a GPU is recommended for deep learning workloads.

It is possible to use HOST-only memory with a CUDA backend. That can be done using workspaces.

Example:
```
WorkspaceConfiguration basicConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.HOST_ONLY) // <--- this option does this trick
                .policySpill(SpillPolicy.EXTERNAL)
                .build();
```

It's not recommended to use HOST-only arrays directly, since they will dramatically reduce performance. But they might be useful as in-memory cache pairs with the `INDArray.unsafeDuplication()` method.
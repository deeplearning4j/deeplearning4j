---
title: Memory management in DL4J and ND4J
layout: default
---

# Memory Management for ND4J/DL4J: How does it work?

ND4J uses off-heap memory to store NDArrays, to provide better performance while working with NDArrays from native code (such as BLAS and CUDA libraries).
Off-heap means that the memory is allocated outside of the JVM (Java Virtual Machine) and hence isn't managed by the JVM's garbage collection (GC). On the Java/JVM side we only hold pointers to the off-heap memory, which can be passed to the underlying C++ code (via JNI) for use in ND4J operations.

To manage memory allocations we use two approaches:

- JVM Garbage Collector (GC) and WeakReference tracking
- MemoryWorkspaces - see [Workspaces guide](https://deeplearning4j.org/workspaces) for details

Despite the differences between these two approaches, the idea stays the same: once (on the Java side) some NDArray is no longer required, the off-heap associated with it should be released so it can be later reused. The difference between the GC and MemoryWorkspaces approaches is in when and how the memory is released.

- For JVM/GC memory: whenever an INDArray is collected by the garbage collector, its off-heap memory (assuming it is not used elsewhere) will be deallocated
- For MemoryWorksaces: whenever an INDArray leaves the workspace scope (for example, when a layer finished forward pass/predictions) its memory may be reused, without deallocation and reallocation. This results in better performance for cyclical workloads.


# Configuring Memory Limits

With DL4J/ND4J, there are two types of memory limits to be aware of and configure: The on-heap JVM memory limit, and the off-heap memory limit. Both limits are controlled via Java command line arguments:

`-Xms` - this option defines how much memory JVM heap will use at application start.

`-Xmx` - this option allows you to specify JVM heap memory limit (maximum, at any point). Only allocated up to this amount (at the discretion of the JVM) if required.

`-Dorg.bytedeco.javacpp.maxbytes`  - this option allows you to specify the off-heap memory limit.

`-Dorg.bytedeco.javacpp.maxphysicalbytes`  - also for off-heap, this option usually should be set equal to `maxbytes`

Example: Configuring 1GB initial on-heap, 2GB max on-heap, 8GB off-heap:

```-Xms1G -Xmx2G -Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=8G```

**Best practice**: for many applications, you want less RAM to be used in JVM heap, and more RAM to be used in off-heap, since all NDArrays are stored there. If you allocate too much to the JVM heap, there will not be enough memory left for the off-heap memory.


**PLEASE NOTE**: If you don't specify JVM heap limit, it will use 1/4 of your total system RAM as the limit, by default.

**PLEASE NOTE**: If you don't specify off-heap memory limit, x2 of JVM heap limit (Xmx) will be used by default. i.e. `-Xmx8G` will mean that 8GB can be used by JVM heap, and 16GB can be used by ND4j in off-heap.

**PLEASE NOTE**: In limited memory environments it's usually a bad idea to use high `-Xmx` value together with `-Xms` option. Again: this won't leave enough off-heap memory. Consider a 16GB system. Suppose you set `-Xms14G`, this means 14 of 16GB will be allocated to the JVM, leaving only 2GB for the off-heap memory (and the OS and all other programs).


# Memory-mapped files

As of 0.9.2-SNAPSHOT, it's possible to use memory-mapped file instead of RAM when using `nd4j-native` backend. On one hand it's slower then RAM, but on other hand it allows you to allocate memory chunks impossible otherwise.

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
In this case, 1GB temporary file will be created and mmap'ed, and NDArray `x` will be created in that space.
Obviously, this option is mostly viable for cases when you need NDArrays that can't fit into your RAM.



# GPUs

When using GPUs, a typical case is your CPU ram being greater than GPU ram. When GPU ram is less than CPU ram, we need to monitor how much ram is being used off-heap. You can check this based on the javacpp options specified above.

Of note here is we allocate memory on the GPU equivalent to the amount of offheap memory you specify. We don't use anymore of your GPU than that. You are also allowed to (but it's not encouraged) to specify heap space greater than your gpu, but your gpu will run out of ram when trying to run jobs.
We also allocate off heap memory on the cpu ram as well. This is for efficient communicaton of CPU to GPU, and CPU accessing data from an NDArray without having to fetch data from the GPU each time you call for it.

If JavaCPP or your GPU throws an out of memory error, or even if your compute slows down (due to GPU memory being limited), then you either may want to decrease batch size or if you can increase the amount of off-heap memory javacpp is allowed to allocate.

Try to run with an off-heap memory equal to your GPU's ram. Also, always remember to set up a small JVM heap space (using `Xmx` option)

Please do note that if your gpu has < 2g of ram it's probably not really usable for deep learning. You should consider just using your CPU if this is the case. Typical deep learning workloads should *at minimum* have 4GB of ram. 4GB of ram is not recommended though. At least 8GB of ram on a GPU is recommended for deep learning workloads.

However, it IS possible to use HOST-only memory with CUDA backend, that can be done using workspaces.

Example:
```
WorkspaceConfiguration basicConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.HOST_ONLY) // <--- this option does this trick
                .policySpill(SpillPolicy.EXTERNAL)
                .build();
```

***PLEASE NOTE:*** it's not recommended to use HOST-only arrays directly, since they will dramatically reduce performance. But they might be useful as in-memory cache paires with `INDArray.unsafeDuplication()` method

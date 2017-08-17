---
title: Memory management in DL4j and ND4j
layout: default
---

# How it works?

ND4j uses off-heap memory to store NDArrays, to provide better performance while working with NDArrays from native code via JNI.
Basically, on Java side we only hold pointers to off-heap memory.

# Configuring limits

With DL4j/ND4j you can control both memory limits: JVM heap limit, and off-heap limit. Both are controlled via Java command line arguments:

`-Xmx` - this option allows you to specify JVM heap memory limit.
`-Dorg.bytedeco.javacpp.maxbytes`  - this option allows you to specify off-heap memory limit.


PLEASE NOTE: If you don't specify JVM heap limit, it will use 1/4 of your total system RAM as limit.
PLEASE NOTE: If you don't specify off-heap memory limit, x2 of JVM heap limit will be considered. i.e. `-Xmx8G` will mean that 8GB can be used by JVM heap, and 16GB can be used by ND4j in off-heap.

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
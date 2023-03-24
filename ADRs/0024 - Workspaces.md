# Workspaces

## Status

Implemented

Proposed by: Adam Gibson (14 Mar 2023)

Discussed with: Paul Dubs



## Context

Neural networks require a significant amount of memory during execution, often in the range of billions of parameters.
To improve performance and manage memory usage, we can take advantage of the fact that neural network allocations are cyclic in nature. 
Since most workloads repeatedly allocate the same ndarrays, we create a memory abstraction known as "workspaces" to avoid redundant memory allocation.
This approach helps to optimize memory usage and enhance overall performance.

## Proposal

This architecture decision record discusses the implementation of the workspaces concept using ringbuffers within a
namespace-like abstraction and Java's try/with resources for memory allocation and garbage collection. 
Workspaces require a configuration with several parameters for controlling memory allocation. (See the description section for more details.)

A MemoryManager is used to allocate an INDArray, and an operation (element-wise multiplication) is performed. 
The workspace and INDArray are automatically closed and released when the try blocks are exited.

The workspace tracks different types of memory, including:
1. allocated memory
2. external memory
3. unreferenced memory
4. workspace memory
5. gradient memory. 

We reduce memory usage by reusing the ring buffers described above. The key trick in reducing allocations is to reuse the same memory for the same operations
learned through the learning policy. This is done by using a ring buffer to store the memory for each operation.
In doing this, the user can reuse existing memory for training/inference increasing performance and reducing memory usage.



## Description 

To create a named scope that reuses memory instead of allocating it again, you can use ringbuffers within a namespace-like abstraction, 
and combine it with java's try/with resources to indicate a scope of memory as well as to automatically garbage collect relevant memory.





In order to use a workspace we need to have a configuration to determine how a workspace is created
and how it allocates memory. The following parameters are possible:

1. initialSize: The initial size of the workspace in bytes. If the workspace exceeds this size, it will be automatically expanded.

2. maxSize: The maximum size of the workspace in bytes. If the workspace tries to expand beyond this size, an exception will be thrown.

3. overallocationLimit: The amount of extra memory to allocate beyond the initial size when the workspace is created. 
This is useful for workloads that have high variability in their memory usage.

4. policyAllocation: The allocation policy for the workspace, which can be STRICT (strict allocation), 
OVERALLOCATE (overallocation), or ALWAYS (always allocate new memory).

5. policyLearning: The learning policy for the workspace, which can be NONE (no learning), 
OPTIMIZED (optimized learning), or TRAINING (full training mode).

6. policyMirroring: The mirroring policy for the workspace, which can be ENABLED (enable mirroring),
DISABLED (disable mirroring), or HOST_ONLY (mirror only to host memory).

7. policySpill: The spill policy for the workspace, which can be FAIL (fail if workspace runs out of memory),
REALLOCATE (reallocate memory on the fly), or EXTERNAL (spill to external memory).

8. overallocationLimit: The amount of extra memory to allocate beyond the initial size when the workspace is created.
This is useful for workloads that have high variability in their memory usage.

9. tempBlockSize: The size of the temporary memory blocks used by the workspace, in bytes.

10. useCycleDetector: Whether to enable the cycle detector for the workspace, which detects and prevents memory leaks.
workspaceMode: The workspace mode, which can be ENABLED (enable workspace mode), SINGLE (use a single global workspace), 
or NONE (disable workspace mode).

11. helperAllowFallback: Whether to allow fallback to the CPU when using GPU memory.

12. helperMinSize: The minimum size in bytes for workspace helper operations.



Example usage:
In this example, we use try/with blocks to automatically close the workspace and release the INDArray from the workspace memory when
the try block is exited.

We create a workspace with the specified configuration within the try block, and get the MemoryManager for the workspace.
We allocate an INDArray using the workspace memory within another try block, and perform some operation
on it (in this case, an element-wise multiplication).


```java
// create a workspace configuration with 1 GB initial size and host memory only
WorkspaceConfiguration config = WorkspaceConfiguration.builder()
    .initialSize(1024 * 1024 * 1024) // 1 GB initial size
    .policyMirroring(MirroringPolicy.HOST_ONLY) // use host memory only
    .build();

// create a workspace with the specified configuration
try (Workspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(config)) {

    // get the memory manager for the workspace
    MemoryManager memMgr = workspace.getMemoryManager();

    // allocate an INDArray using the workspace memory
    try (INDArray input = memMgr.allocate(new long[]{32, 32}, DataBuffer.Type.FLOAT)) {

        // use the INDArray for some operation, e.g. element-wise multiplication
        input.muli(2);

    } // the INDArray is automatically released from the workspace memory when the try block is exited

} // the workspace is automatically closed when the try block is exited
```

Since we used try/with blocks to create the workspace and allocate the INDArray, they will be automatically
closed and released from the workspace memory when the try blocks are exited, regardless of
whether an exception is thrown or not.


In order to create a workspace we need to track the following kinds of memory:
Allocated memory: This is memory that has been explicitly allocated by the workspace for a particular operation or computation.

External memory: This is memory that has been allocated outside of the workspace, but is being used by operations within the workspace. 
External memory can be useful when working with large datasets or models that do not fit entirely within the workspace.

Unreferenced memory: This is memory that has been allocated by the workspace, but is no longer being used by any operations or computations. 
Unreferenced memory can be automatically deallocated by the workspace to free up memory resources.

Workspace memory: This is memory that has been explicitly allocated by the workspace itself for managing memory and workspace state. 
Workspace memory can include things like memory for managing scope, tracking allocations and deallocations, and managing internal structures.

Gradient memory: This is memory that is used for storing gradients during backpropagation during training.
DL4J's workspaces can track different types of gradient memory, including standard gradients, external gradients, and deferred gradients.


Note that misuse can cause memory leaks in the following ways:
Not closing the workspace properly: If a workspace is not properly closed after use, it can cause a memory leak. 
This can happen when a user forgets to close the workspace or when an exception occurs and the workspace is not closed in the catch block.

Using a workspace for too long: If a workspace is used for too long, it can cause a memory leak. 
This can happen if the workspace is reused too many times or if it is not cleared after each use.

Holding onto references: If references to objects created within a workspace are held onto for too long, it can cause a memory leak. 
This can happen if objects are not released from the workspace after they are no longer needed.

Using too many workspaces: If too many workspaces are created, it can cause a memory leak.
This can happen if workspaces are created unnecessarily or if they are not properly managed.

Incorrect workspace configuration: If the workspace is configured incorrectly, it can cause a memory leak.
This can happen if the workspace is not allocated enough memory or if the allocation policy is not set correctly.


## Consequences

### Advantages

* Memory allocation: Workspaces allow for pre-allocation of memory to avoid the overhead associated with dynamic memory allocation during training.

* Memory reuse: By reusing allocated memory rather than allocating new memory for each operation, workspaces help to reduce memory fragmentation and improve performance.

* Scope management: Workspaces are created within a particular scope and can be closed once they are no longer needed. This allows for efficient memory management and prevents memory leaks.

* Automatic deallocation: When a workspace is closed, any memory that was allocated within the workspace is automatically deallocated, freeing up memory resources for other operations.

Multiple workspaces: DL4J allows for the creation of multiple workspaces, which can be useful when running multiple models or training processes simultaneously.

### Disadvantages

* Increased code complexity: Implementing workspaces in your code can add an additional layer of complexity and require more careful management of workspace creation and usage.

* Memory overhead: Workspaces require some overhead for workspace creation, management, and tracking, which can increase memory usage.

* Workspace size limitations: Since workspaces are pre-allocated with a fixed size, there may be cases where the allocated size is not sufficient for larger models or datasets. This can limit the performance and accuracy of the training process.

* Training slowdowns: Depending on the specific use case and how workspaces are implemented, there may be cases where using workspaces could actually slow down the training process rather than speed it up.

* Learning curve: Using workspaces effectively requires a good understanding of how they work and how to manage them properly, which may require some additional learning and training time.
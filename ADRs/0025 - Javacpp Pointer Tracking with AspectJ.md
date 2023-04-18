# JavaCPP Pointer Tracking with AspectJ

## Status
Implemented

Proposed by: Adam Gibson (18 Apr 2023)

Discussed with: Paul Dubs

Finalized by: Adam Gibson (18 Apr 2023)

## Context

Tracking memory allocations and deallocations for JavaCPP pointers is challenging. This is 
important for understanding memory usage patterns and identifying memory leaks in 
applications that use JavaCPP. Currently, developers rely on manual tracking 
and debugging techniques,
which are time-consuming and error-prone.

Aspect Oriented Programming (AOP) with AspectJ is used to intercept the allocation and
deallocation of JavaCPP pointers, allowing for automatic tracking and reporting of memory usage.
In this context, we are implementing an aspect and a memory counter for tracking JavaCPP pointer
allocations and deallocations.

AOP is only enabled when used with compile time weaving. You have to build the profiler with
the specified modules in order to enable the weaving. 

The code should not be built with AOP by default due to the overhead.

## Decision

We use AspectJ to create an aspect called `MemoryCounterAspect` that intercepts 
the allocation and deallocation of JavaCPP pointers. This aspect leverages two 
around advice methods, `allocateMemory` and `deallocate`, to track the memory usage.

The `allocateMemory` method is triggered when a new JavaCPP pointer is created. It calculates 
the difference in physical bytes before and after the pointer allocation, and then increments 
the memory counter accordingly.

The `deallocate` method is triggered when a JavaCPP pointer is deallocated. It calculates the difference 
in physical bytes before and after the pointer deallocation, and then decrements the memory counter accordingly.

The memory counter, `MemoryCounter`, maintains two counters: `allocated` for tracking the total allocated memory,
and `instanceCounts` for tracking the number of instances of each JavaCPP pointer class.

Example usage:

    ```java


    // Perform operations involving JavaCPP pointers
    // ...

    // Get memory usage information
    Map<String, Long> allocatedMemory = MemoryCounter.getAllocated().getCounts();
    Map<String, Long> instanceCounts = MemoryCounter.getInstanceCounts().getCounts();


    ```

## Consequences

### Advantages
* Simplifies memory tracking for JavaCPP pointers
* Reduces manual debugging efforts
* Provides valuable insights into memory usage patterns and potential memory leaks

### Disadvantages
* AspectJ introduces additional overhead and may slightly impact performance
* The tracking aspect may not cover all possible allocation and deallocation scenarios, depending on the JavaCPP library's 
  behavior and the application's usage patterns
* The aspect may need to be updated to stay in sync with changes in the JavaCPP library

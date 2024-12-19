# ADR 0033: Shape Buffer Trie Implementation

## Status

Implemented

Proposed by: Adam Gibson (19-12-2024)
Discussed with: Paul Dubs

## Context
The libnd4j library requires efficient storage and lookup of shape information for neural network operations. Shape information is frequently accessed during computation and needs to be managed efficiently to prevent memory leaks and optimize performance. Previously, we used a ShapeDescriptor-based cache with an unordered map, which created unnecessary object allocations. The primary challenges include:

1. Frequent shape buffer allocations and deallocations during neural network operations
2. Need for fast shape lookup during computation
3. Memory management of redundant shape information
4. Thread safety requirements for parallel execution
5. Overhead from ShapeDescriptor creation for cache lookups
6. Memory overhead from unordered map storage

## Decision
We implement a shape buffer trie data structure (`DirectShapeTrie`) to manage and cache shape information, replacing the previous unordered map implementation. The trie structure is chosen for the following characteristics:

### Key Components
- A trie node structure containing:
  - Shape buffer pointer
  - Child node pointers
  - Reference counting mechanism
- Striped thread safety using an array of mutexes
- Direct memory management of shape buffers
- Sequential shape information exploitation similar to word tries

### Implementation Details
1. The trie stores shape buffers based on their content as sequential paths
2. Each unique shape path in the trie represents a unique shape configuration
3. Reference counting is used to manage memory lifecycle
4. Thread safety is ensured through striped mutex locks
5. Direct memory allocation is used instead of standard containers
6. Removed ShapeDescriptor creation requirement for lookups
7. Shape values are stored sequentially in the trie, similar to characters in a word trie

### Visual Example
```
Root
├── 2 (rank)
│   ├── 3,4 (shape values)
│   │   └── [ptr: shape_buffer_1]
│   └── 5,6 (shape values)
│       └── [ptr: shape_buffer_2]
└── 3 (rank)
    ├── 2,3,4 (shape values)
    │   │   └── [ptr: shape_buffer_3]
    └── 4,5,6 (shape values)
        └── [ptr: shape_buffer_4]
```

In this example:
- Each level represents a component of the shape
- First level: rank of the array
- Subsequent levels: actual shape values
- Leaf nodes contain pointers to the actual shape buffers
- Multiple shapes can share common prefixes, saving memory

## Thread Safety Implementation

The shape buffer cache implements striped locking using an array of mutexes:
```cpp
mutable std::array<MUTEX_TYPE, NUM_STRIPES> _mutexes;
```

This design provides:
1. Reduced contention through multiple lock stripes
2. Better concurrency than a single global mutex
3. Lower memory overhead than per-node locking
4. Const-correctness through mutable mutex array

The striping mechanism:
1. Distributes shapes across multiple mutexes based on their characteristics
2. Allows concurrent operations on shapes in different stripes
3. Balances between fine-grained locking and implementation complexity

## Consequences

### Advantages
1. Memory Efficiency:
   - Eliminates redundant shape buffer storage
   - Automatic cleanup of unused shapes through reference counting
   - Shared shape buffers across operations
   - Removal of ShapeDescriptor allocation overhead
   - Better memory locality due to trie structure

2. Performance:
   - O(n) lookup time where n is the shape length
   - Efficient shape comparison through pointer equality
   - Reduced memory allocation overhead
   - No ShapeDescriptor creation cost for lookups
   - Sequential access patterns for shape values

3. Thread Safety:
   - Safe concurrent access through striped mutex protection
   - Atomic reference counting operations
   - Protected shape buffer lifecycle management

4. Concurrency:
   - Striped locking enables parallel access to different shape regions
   - Better scaling under high concurrency than single mutex
   - Maintains simplicity compared to per-node locking

### Disadvantages
1. Implementation Complexity:
   - Manual memory management requires careful implementation
   - Reference counting edge cases need careful handling
   - Thread synchronization adds complexity
   - More complex trie traversal logic

2. Memory Overhead:
   - Trie structure itself introduces memory overhead
   - Additional pointers for trie navigation

3. Performance Trade-offs:
   - Stripe selection adds minor overhead
   - Multiple shapes might still hash to same stripe
   - Reference counting operations add CPU overhead
   - Fixed number of stripes limits maximum parallelism

## Technical Details

### Memory Management
```cpp
sd::LongType* createBuffer(int length);
void deleteBuffer(sd::LongType* buffer);
```

### API Design
```cpp
sd::LongType* lookupBuffer(const sd::LongType* shape, int length);
void registerBuffer(const sd::LongType* shape, int length);
void decrementRef(const sd::LongType* buffer);
```

## Alternatives Considered

1. Hash Table Implementation (Previous Approach):
   - Pros: Simpler implementation, O(1) average lookup
   - Cons: More memory usage, ShapeDescriptor overhead, potential hash collisions
   - Removed due to ShapeDescriptor creation overhead and memory concerns

2. Simple Buffer Pool:
   - Pros: Simpler implementation, direct access
   - Cons: Less memory efficient, slower lookups, no shape value exploitation

3. Lock-free Data Structure:
   - Pros: Better concurrent performance
   - Cons: Much higher implementation complexity, harder to maintain

4. Retain ShapeDescriptor with Different Structure:
   - Pros: Familiar API, existing code compatibility
   - Cons: Continued object creation overhead, memory pressure

5. Alternative Locking Strategies:
   - Single Global Mutex:
     - Pros: Simplest implementation
     - Cons: High contention under load
   - Per-Node Locking:
     - Pros: Maximum concurrency
     - Cons: High memory overhead, complex synchronization
   - Lock-Free Design:
     - Pros: No lock contention
     - Cons: Extremely complex implementation, harder to verify correctness



## References
- ConstantShapeHelper.cpp implementation
- DirectShapeTrie.h interface
- Existing shape management system documentation
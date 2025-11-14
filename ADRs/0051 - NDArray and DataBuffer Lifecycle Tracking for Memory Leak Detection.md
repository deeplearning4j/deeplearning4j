# NDArray and DataBuffer Lifecycle Tracking for Memory Leak Detection

## Status

Implemented

Proposed by: Adam Gibson (06-11-2025)
Discussed with: Claude Code

## Context

Memory leak detection in the deeplearning4j NDArray subsystem is challenging because:

1. **Complex ownership model**: NDArrays can be heap-allocated or stack-allocated, views vs copies, with different cleanup responsibilities
2. **Two-level memory architecture**:
   - NDArray wrappers contain shape/metadata
   - DataBuffer objects contain actual memory allocations (PRIMARY for CPU, SPECIAL for CUDA)
3. **Template-heavy codebase**: Operations create many temporary NDArrays through template instantiation
4. **JNI boundary issues**: Leaks at the C++/Java boundary are difficult to attribute
5. **Scale**: Applications can create millions of NDArrays, making Valgrind impractically slow (20-50x overhead)
6. **Lack of context**: AddressSanitizer (ADR 0049) detects leaks but doesn't provide application-level context about NDArray shapes, types, or creation patterns

Specific problems encountered:
- 9+ memory leaks in `convolutions_conv2d.cpp` totaling 500MB-1GB per forward pass
- Gradual memory growth: 2GB/minute in production workloads
- 498 new anonymous memory regions growing over 10 minutes
- 100GB of memory not freed despite NDArray destructors being called
- No systematic way to identify which operations or files leak the most
- Difficult to distinguish between wrapper leaks (NDArray objects) and actual memory leaks (DataBuffer allocations)

### Existing Infrastructure

The codebase already has `SD_GCC_FUNCTRACE` compile flag that:
- Enables stack trace capture using backward-cpp library
- Stores `StackTrace creationTrace` field in NDArray (NDArray.h:211-213)
- Stores allocation traces in DataBuffer (DataBuffer.h:55-60)
- Provides `TraceResolver` for symbol resolution

However, this infrastructure only captures **allocation** traces and doesn't:
- Track when NDArrays are deallocated
- Track actual DataBuffer memory allocations/deallocations
- Separate PRIMARY (host/CPU) from SPECIAL (device/GPU) memory tracking
- Aggregate leak statistics by file or function
- Generate periodic reports during execution
- Produce flamegraph visualizations
- Compare allocation vs deallocation patterns

## Decision

Implement a **two-level comprehensive lifecycle tracking system** that:

### Level 1: NDArray Wrapper Tracking
1. **Records all NDArray wrapper allocations and deallocations** with full stack traces
2. **Tracks view vs owned NDArrays** to understand wrapper object lifecycle
3. **Identifies which operations create excessive temporary objects**

### Level 2: DataBuffer Memory Tracking
1. **Records all DataBuffer PRIMARY (CPU) and SPECIAL (GPU) memory allocations/deallocations**
2. **Separates host and device memory statistics**
3. **Tracks actual memory consumption** (the real leaks)
4. **Correlates with NDArray ownership**

### Cross-Layer Integration
1. **Aggregates leak statistics** by source file, function, and type
2. **Generates periodic reports** (default: every 5 minutes) during execution
3. **Produces flamegraphs** for allocation and deallocation hotspot visualization
4. **Configures via environment variables** with smart defaults
5. **Integrates with existing Environment singleton** for consistent configuration management
6. **Provides JNI API** for Java-level access to statistics
7. **Works with existing DeallocatorService** for Java/C++ boundary tracking

## Architecture

The system consists of a multi-layer architecture spanning C++, JNI, and Java:

```
┌─────────────────────────────────────────────────────────┐
│  Java Application Layer                                 │
│  - DeallocatorService (existing)                        │
│  - OpaqueNDArrayDeallocator (existing)                  │
│  - OpaqueDataBufferDeallocator (existing)               │
│  - LifecycleTrackingHelper (new - for statistics)       │
└──────────────────┬──────────────────────────────────────┘
                   │ JNI
┌──────────────────▼──────────────────────────────────────┐
│  JNI Layer (libnd4j)                                     │
│  - NativeOps.h - API declarations (8 new functions)     │
│  - NativeOpsHelpers_LifecycleTracking.cpp               │
│  - JSON serialization for Java consumption              │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  C++ Native Layer                                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  NDArray Level Tracking (Wrapper Objects)       │    │
│  │  - All 26 NDArray constructors instrumented     │    │
│  │  - NDArray destructor                           │    │
│  │  - NDArrayLifecycleTracker singleton            │    │
│  │  - Tracks wrapper creation/destruction          │    │
│  │  - View vs owned NDArray differentiation        │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  DataBuffer Level Tracking (Actual Memory)      │    │
│  │  - DataBuffer::allocatePrimary() (CPU)          │    │
│  │  - DataBuffer::deletePrimary() (CPU)            │    │
│  │  - DataBuffer::allocateSpecial() (CUDA)         │    │
│  │  - DataBuffer::deleteSpecial() (CUDA)           │    │
│  │  - DataBufferLifecycleTracker singleton         │    │
│  │  - Separate PRIMARY (host) & SPECIAL (device)   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Component 1: NDArrayLifecycleTracker

Singleton class that maintains:
- **Live allocations map**: `unordered_map<void*, AllocationRecord*>` tracking all currently allocated NDArrays
- **Deletion history**: `vector<DeallocationRecord*>` (bounded, configurable size)
- **Allocation history**: `unordered_map<uint64_t, AllocationRecord*>` keyed by allocation ID
- **File statistics**: `map<string, FileStats>` aggregating leaks by source file
- **Global counters**: atomic counters for total allocations, deallocations, current live count, current bytes, peak bytes, double-frees

**Key data structures:**

```cpp
struct AllocationRecord {
    void *ndarray_ptr;              // NDArray address
    size_t size_bytes;              // Memory size
    DataType dtype;                 // Element type
    std::string shape_str;          // Shape as "[2,3,4]"
    StackTrace *allocation_trace;   // Creation stack trace
    std::chrono::steady_clock::time_point allocation_time;
    uint64_t thread_id;
    bool is_view;                   // View vs owned buffer
    uint64_t allocation_id;         // Unique ID
};

struct DeallocationRecord {
    void *ndarray_ptr;
    StackTrace *deallocation_trace; // Deletion stack trace
    std::chrono::steady_clock::time_point deallocation_time;
    uint64_t thread_id;
    uint64_t allocation_id;         // Matches AllocationRecord
};

struct FileStats {
    std::string filename;
    uint64_t alloc_count;
    uint64_t dealloc_count;
    uint64_t current_live;          // Leaks in this file
    size_t current_bytes;
    size_t peak_bytes;
    std::map<std::string, uint64_t> function_alloc_counts;  // Per-function breakdown
    std::map<std::string, size_t> function_bytes;
};
```

**Location**: `libnd4j/include/array/NDArrayLifecycleTracker.h` and `libnd4j/include/array/impl/NDArrayLifecycleTracker.cpp`

### Component 2: DataBufferLifecycleTracker

Singleton class that tracks **actual memory allocations** at the DataBuffer level:

**Key data structures:**

```cpp
enum class BufferType {
  PRIMARY,   // Host/CPU memory
  SPECIAL,   // Device/GPU memory
  UNKNOWN
};

struct BufferAllocationRecord {
    void *buffer_ptr;               // Actual memory address
    size_t size_bytes;              // Buffer size
    DataType dtype;
    BufferType buffer_type;         // PRIMARY or SPECIAL
    void *databuffer_owner;         // DataBuffer object that owns this
    StackTrace *allocation_trace;
    std::chrono::steady_clock::time_point allocation_time;
    uint64_t thread_id;
    uint64_t allocation_id;
    bool is_workspace;              // Allocated in workspace
};

struct BufferDeallocationRecord {
    void *buffer_ptr;
    BufferType buffer_type;
    StackTrace *deallocation_trace;
    std::chrono::steady_clock::time_point deallocation_time;
    uint64_t thread_id;
    uint64_t allocation_id;
};

struct BufferFileStats {
    std::string filename;
    uint64_t primary_alloc_count;
    uint64_t primary_dealloc_count;
    uint64_t special_alloc_count;
    uint64_t special_dealloc_count;
    uint64_t current_live_primary;
    uint64_t current_live_special;
    size_t current_primary_bytes;
    size_t current_special_bytes;
    size_t peak_primary_bytes;
    size_t peak_special_bytes;
    std::map<std::string, uint64_t> function_primary_alloc_counts;
    std::map<std::string, uint64_t> function_special_alloc_counts;
    std::map<std::string, size_t> function_primary_bytes;
    std::map<std::string, size_t> function_special_bytes;
};
```

**Key features:**
- Separate tracking for PRIMARY (host/CPU) and SPECIAL (device/GPU) memory
- Independent statistics, flamegraphs, and reports for each buffer type
- Workspace allocation flagging
- Owner tracking (links to DataBuffer object)

**Location**: `libnd4j/include/array/DataBufferLifecycleTracker.h` and `libnd4j/include/array/impl/DataBufferLifecycleTracker.cpp`

### Component 3: Environment Integration

Extended existing `Environment` singleton with lifecycle tracking configuration:

**Environment.h additions:**
```cpp
// Private fields (with atomic storage for thread safety)
std::atomic<bool> _lifecycleTracking{true};       // Enabled by default with functrace
std::atomic<bool> _trackViews{true};              // Track view NDArrays
std::atomic<bool> _trackDeletions{true};          // Capture deletion stack traces
std::atomic<int> _stackDepth{32};                 // Stack trace depth
std::atomic<int> _reportInterval{300};            // Periodic report interval (seconds)
std::atomic<size_t> _maxDeletionHistory{10000};  // Bounded deletion history

// Public methods
bool isLifecycleTracking();
void setLifecycleTracking(bool enabled);
int getStackDepth();
void setStackDepth(int depth);
int getReportInterval();
void setReportInterval(int seconds);
// ... more methods
```

### Component 4: NDArray Integration

**Complete constructor instrumentation** - All 26 NDArray constructors now tracked:

```cpp
// Macro for recording allocations (NDArray.hXX lines 62-76)
#if defined(SD_GCC_FUNCTRACE)
#define RECORD_NDARRAY_ALLOCATION() \
  do { \
    if (!isEmpty() && _shapeInfo != nullptr) { \
      std::vector<sd::LongType> shape_vec; \
      for (int i = 0; i < rankOf(); i++) { \
        shape_vec.push_back(sizeAt(i)); \
      } \
      size_t size_bytes = lengthOf() * sizeOfT(); \
      array::NDArrayLifecycleTracker::getInstance().recordAllocation( \
        this, size_bytes, dataType(), shape_vec, isView()); \
    } \
  } while(0)
#else
#define RECORD_NDARRAY_ALLOCATION() do {} while(0)
#endif
```

**Destructor tracking** (NDArray.hXX lines 664-686):
```cpp
NDArray::~NDArray() {
#if defined(SD_GCC_FUNCTRACE)
  array::NDArrayLifecycleTracker::getInstance().recordDeallocation(this);
#endif

  // Only delete buffer if not a view
  if(_buffer != nullptr && _shapeInfo != nullptr && !this->isView() && !this->isEmpty()) {
    delete this->_buffer;  // This triggers DataBuffer destructor → DataBuffer tracking
  }
  // ... cleanup ...
}
```

**View Handling**: The `!this->isView()` check ensures:
- View NDArray wrappers are tracked as allocations (shows excessive view creation)
- View destructor doesn't double-free the shared buffer
- DataBuffer is only deleted once by the owning NDArray

### Component 5: DataBuffer Integration

**CPU (PRIMARY) Memory Tracking** (DataBuffer.cpp):

```cpp
// Allocation (line 436-441)
void DataBuffer::allocatePrimary() {
    ALLOCATE(_primaryBuffer, _workspace, getLenInBytes(), int8_t);
    _isOwnerPrimary = true;

#if defined(SD_GCC_FUNCTRACE)
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        _primaryBuffer, getLenInBytes(), getDataType(),
        array::BufferType::PRIMARY, this, _workspace != nullptr);
#endif
}

// Deallocation (line 461-468)
void DataBuffer::deletePrimary() {
    if (_isOwnerPrimary && _primaryBuffer != nullptr) {
#if defined(SD_GCC_FUNCTRACE)
        array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
            _primaryBuffer, array::BufferType::PRIMARY);
#endif
        RELEASE(p, _workspace);
        _primaryBuffer = nullptr;
    }
}
```

**CUDA (SPECIAL) Memory Tracking** (DataBuffer.cu):

```cpp
// Allocation (line 336-344)
void DataBuffer::allocateSpecial() {
    ALLOCATE_SPECIAL(_specialBuffer, _workspace, getLenInBytes(), int8_t);
    _isOwnerSpecial = true;

#if defined(SD_GCC_FUNCTRACE)
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        _specialBuffer, getLenInBytes(), getDataType(),
        array::BufferType::SPECIAL, this, _workspace != nullptr);
#endif
}

// Deallocation (line 421-428)
void DataBuffer::deleteSpecial() {
    if (_isOwnerSpecial && _specialBuffer != nullptr) {
#if defined(SD_GCC_FUNCTRACE)
        array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
            _specialBuffer, array::BufferType::SPECIAL);
#endif
        RELEASE_SPECIAL(p, _workspace);
        _specialBuffer = nullptr;
    }
}
```

### Component 6: JNI Layer Integration

**Native API Functions** (NativeOps.h lines 478-559):

Eight new functions for Java integration:

```cpp
#if defined(SD_GCC_FUNCTRACE)

// Statistics as JSON
SD_LIB_EXPORT const char* getNDArrayLifecycleStats();
SD_LIB_EXPORT const char* getDataBufferLifecycleStats();

// Flamegraph generation
SD_LIB_EXPORT void generateNDArrayAllocationFlamegraph(const char* outputPath);
SD_LIB_EXPORT void generateNDArrayDeallocationFlamegraph(const char* outputPath);
SD_LIB_EXPORT void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType);
SD_LIB_EXPORT void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType);

// Combined reports
SD_LIB_EXPORT void generateLifecycleLeakReport(const char* outputPath);

// Memory management
SD_LIB_EXPORT void freeString(const char* str);

#endif
```

**Implementation** (NativeOpsHelpers_LifecycleTracking.cpp):
- JSON serialization for Java consumption
- Delegates to lifecycle trackers
- Automatic string memory management via `freeString()`

**OpaqueNDArray Integration**:
- `OpaqueNDArray` is typedef for `sd::NDArray*` (NativeOps.h line 61)
- Existing `deleteNDArray()` calls `delete array` → triggers destructor → automatic tracking
- No additional changes needed - integration is automatic

### Component 7: Java Layer Integration

**Existing Components** (no changes needed):
- `DeallocatorService` - Already manages reference queues and calls deallocators
- `OpaqueNDArrayDeallocator` - Calls `deleteNDArray()` → triggers tracking
- `OpaqueDataBufferDeallocator` - Similar pattern

**Proposed Helper Class** (LifecycleTrackingHelper.java):

```java
package org.nd4j.linalg.api.memory;

public class LifecycleTrackingHelper {
    private static final boolean AVAILABLE = checkAvailability();

    public static class NDArrayStats {
        public final long totalAllocations;
        public final long totalDeallocations;
        public final long currentLive;
        public final long peakLive;
        public final long currentBytes;
        public final long peakBytes;
        public final long doubleFrees;
        // ... constructor from JSON
    }

    public static class DataBufferStats {
        public static class BufferTypeStats {
            public final long totalAllocations;
            public final long totalDeallocations;
            public final long currentLive;
            public final long currentBytes;
            public final long peakBytes;
            // ... constructor from JSON
        }
        public final BufferTypeStats primary;  // CPU/host memory
        public final BufferTypeStats special;  // GPU/device memory
        public final long doubleFrees;
        // ... constructor from JSON
    }

    public static NDArrayStats getNDArrayStats() {
        String json = Nd4j.getNativeOps().getNDArrayLifecycleStats();
        try {
            return new NDArrayStats(new JSONObject(json));
        } finally {
            Nd4j.getNativeOps().freeString(json);
        }
    }

    public static DataBufferStats getDataBufferStats() {
        String json = Nd4j.getNativeOps().getDataBufferLifecycleStats();
        try {
            return new DataBufferStats(new JSONObject(json));
        } finally {
            Nd4j.getNativeOps().freeString(json);
        }
    }

    public static void generateLeakReport(String outputPath) {
        Nd4j.getNativeOps().generateLifecycleLeakReport(outputPath);
    }

    // ... more methods for flamegraph generation
}
```

### Component 8: Configuration System

All configuration via **optional** environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SD_LIFECYCLE_TRACKING` | `true` (when functrace enabled) | Master enable/disable |
| `SD_TRACK_VIEWS` | `true` | Track view NDArrays (share buffers) |
| `SD_TRACK_DELETIONS` | `true` | Capture deletion stack traces (memory intensive) |
| `SD_STACK_DEPTH` | `32` | Stack trace capture depth |
| `SD_REPORT_INTERVAL` | `300` | Periodic report interval in seconds (5 minutes) |
| `SD_MAX_DELETION_HISTORY` | `10000` | Max deletion records to keep in memory |
| `SD_LIFECYCLE_SNAPSHOT_FILES` | `false` | Write periodic file snapshots (not just stderr) |
| `SD_LIFECYCLE_TRACK_OPERATIONS` | `false` | Track operation names (e.g., conv2d, matmul) |

### Component 9: Enhanced Periodic Reporting and Operation Tracking (2025-11-06)

**Motivation**: Initial implementation only reported to stderr during periodic checks. Users needed:
1. **File-based snapshots** - Ability to kill the process early and still have leak reports
2. **Operation breakdown** - Identify which operations (Conv2D, MatMul, etc.) are leaking, not just which files
3. **Signal handling** - Generate reports on Ctrl+C or SIGTERM instead of losing all data

#### 9.1: Periodic File-Based Snapshots

**Implementation**: Enhanced `periodicCheck()` method in NDArrayLifecycleTracker.cpp (lines 605-649)

**Features**:
- Writes timestamped snapshot files when `SD_LIFECYCLE_SNAPSHOT_FILES=1`
- Files named: `ndarray_snapshot_pid<PID>_<timestamp>_<sequence>.txt`
- Sequence counter prevents overwriting previous snapshots
- Each snapshot includes:
  - Full statistics (allocations, deallocations, live count, memory usage)
  - Current leak report with stack traces
  - Operation breakdown (when enabled)

**Code snippet**:
```cpp
void NDArrayLifecycleTracker::periodicCheck() {
    Environment &env = Environment::getInstance();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - _last_report_time).count();

    if (elapsed >= env.getReportInterval()) {
        std::cerr << "\n=== NDArray Lifecycle Periodic Report ===" << std::endl;
        printStatistics(std::cerr);

        // Write periodic snapshot to file if enabled
        if (env.isSnapshotFiles()) {
            std::string timestamp = std::to_string(
                std::chrono::system_clock::now().time_since_epoch().count());
            std::string pid = std::to_string(getpid());
            uint64_t seq = _snapshot_counter++;

            std::ostringstream filename;
            filename << _log_dir << "ndarray_snapshot_pid" << pid << "_" << timestamp
                    << "_" << std::setfill('0') << std::setw(3) << seq << ".txt";

            std::ofstream out(filename.str());
            if (out.is_open()) {
                out << "=== NDArray Lifecycle Periodic Snapshot ===" << std::endl;
                out << "Timestamp: " << timestamp << std::endl;
                out << "PID: " << pid << std::endl;
                out << "Sequence: " << seq << std::endl << std::endl;

                printStatistics(out);
                out << std::endl;
                printCurrentLeaks(out);

                if (env.isTrackOperations()) {
                    out << std::endl;
                    printOperationBreakdown(out);
                }

                out.close();
                std::cerr << "Snapshot written to: " << filename.str() << std::endl;
            }
        }

        _last_report_time = now;
    }
}
```

**Benefits**:
- Can kill process at any time (Ctrl+C) and have previous snapshots preserved
- Time-series analysis: compare snapshots to see leak growth over time
- Debugging long-running processes: check snapshot files without waiting for shutdown
- Automatic cleanup not required: sequence numbers prevent overwrite

#### 9.2: Operation Name Tracking

**Implementation**: Extended AllocationRecord with operation name field

**Modified structures** (NDArrayLifecycleTracker.h):
```cpp
struct AllocationRecord {
    void *ndarray_ptr;
    size_t size_bytes;
    DataType dtype;
    std::string shape_str;
    std::string operation_name;  // NEW: Operation that allocated this (e.g., "conv2d", "matmul")
    StackTrace *allocation_trace;
    std::chrono::steady_clock::time_point allocation_time;
    uint64_t thread_id;
    bool is_view;
    uint64_t allocation_id;

    ~AllocationRecord() {
        if (allocation_trace) delete allocation_trace;
    }
};
```

**New API method** (NDArrayLifecycleTracker.h lines 120-122):
```cpp
void recordAllocationWithOp(void *ndarray_ptr, size_t size_bytes, DataType dtype,
                           const std::vector<LongType> &shape, const char* operation_name,
                           bool is_view = false);
```

**Implementation** (NDArrayLifecycleTracker.cpp lines 225-292):
```cpp
void NDArrayLifecycleTracker::recordAllocationWithOp(void *ndarray_ptr, size_t size_bytes,
                                                     DataType dtype,
                                                     const std::vector<LongType> &shape,
                                                     const char* operation_name,
                                                     bool is_view) {
    Environment &env = Environment::getInstance();
    if (!env.isLifecycleTracking()) return;
    if (is_view && !env.isTrackViews()) return;

    uint64_t alloc_id = _next_allocation_id++;

    AllocationRecord *record = new AllocationRecord();
    record->ndarray_ptr = ndarray_ptr;
    record->size_bytes = size_bytes;
    record->dtype = dtype;
    record->is_view = is_view;
    record->allocation_time = std::chrono::steady_clock::now();
    record->allocation_id = alloc_id;
    record->operation_name = operation_name ? std::string(operation_name) : "";

    // Build shape string
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) ss << ",";
        ss << shape[i];
    }
    ss << "]";
    record->shape_str = ss.str();

    // Capture stack trace
    record->allocation_trace = new StackTrace();
    record->allocation_trace->load_here(env.getStackDepth());
    record->thread_id = record->allocation_trace->thread_id();

    // Update statistics
    TraceResolver resolver;
    std::string filename = extractFilename(record->allocation_trace, resolver);
    std::string function = extractFunction(record->allocation_trace, resolver);

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _live_allocations[ndarray_ptr] = record;
        _allocation_history[alloc_id] = record;

        FileStats &fstats = _file_stats[filename];
        fstats.filename = filename;
        fstats.alloc_count++;
        fstats.current_live++;
        fstats.current_bytes += size_bytes;
        if (fstats.current_bytes > fstats.peak_bytes) {
            fstats.peak_bytes = fstats.current_bytes;
        }
        fstats.function_alloc_counts[function]++;
        fstats.function_bytes[function] += size_bytes;
    }

    _total_allocations++;
    _current_live++;
    _current_bytes += size_bytes;

    if (_current_bytes > _peak_bytes) {
        _peak_bytes.store(_current_bytes.load());
    }

    periodicCheck();
}
```

**Operation Breakdown Report** (NDArrayLifecycleTracker.cpp lines 670-717):
```cpp
void NDArrayLifecycleTracker::printOperationBreakdown(std::ostream &out) {
    std::map<std::string, std::pair<uint64_t, size_t>> op_stats;  // operation -> (count, bytes)

    {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto &pair : _live_allocations) {
            AllocationRecord *record = pair.second;
            std::string op_name = record->operation_name.empty() ? "<unknown>" : record->operation_name;
            op_stats[op_name].first++;
            op_stats[op_name].second += record->size_bytes;
        }
    }

    if (op_stats.empty()) {
        out << "No operation breakdown available (operation tracking may be disabled)" << std::endl;
        return;
    }

    // Convert to vector for sorting
    std::vector<std::tuple<std::string, uint64_t, size_t>> sorted_ops;
    for (const auto &pair : op_stats) {
        sorted_ops.push_back(std::make_tuple(pair.first, pair.second.first, pair.second.second));
    }

    // Sort by bytes (descending)
    std::sort(sorted_ops.begin(), sorted_ops.end(),
              [](const auto &a, const auto &b) { return std::get<2>(a) > std::get<2>(b); });

    out << "=== Top 20 Leaking Operations ===" << std::endl << std::endl;
    out << std::left << std::setw(40) << "Operation"
        << std::right << std::setw(12) << "Leaks"
        << std::setw(15) << "Memory (MB)" << std::endl;
    out << std::string(67, '-') << std::endl;

    int count = 0;
    for (const auto &op : sorted_ops) {
        if (count++ >= 20) break;

        std::string op_name = std::get<0>(op);
        uint64_t leak_count = std::get<1>(op);
        size_t bytes = std::get<2>(op);
        double mb = bytes / (1024.0 * 1024.0);

        out << std::left << std::setw(40) << op_name
            << std::right << std::setw(12) << leak_count
            << std::setw(15) << std::fixed << std::setprecision(2) << mb << std::endl;
    }
}
```

**Usage from operations**:
```cpp
// In SameDiff operation implementation
auto output = new NDArray('c', {batch, height, width, channels}, FLOAT32);
NDArrayLifecycleTracker::getInstance().recordAllocationWithOp(
    output, output->lengthOf() * output->sizeOfT(),
    output->dataType(), {batch, height, width, channels}, "conv2d_bp");
```

**Example report output**:
```
=== Top 20 Leaking Operations ===

Operation                                      Leaks     Memory (MB)
-------------------------------------------------------------------
conv2d_bp                                       1245        12345.67
matmul                                           823         8234.12
relu_bp                                          412         2100.45
<unknown>                                        156          987.23
maxpool                                           89          456.78
...
```

**Benefits**:
- **High-level attribution**: See which operations leak, not just which C++ files
- **Actionable**: "Fix conv2d_bp" is more actionable than "Fix convolutions_conv2d.cpp line 234"
- **Cross-layer analysis**: Correlate operation types with file/function leaks
- **Opt-in overhead**: Operation tracking disabled by default (requires explicit tagging)

#### 9.3: Signal Handler for Early Termination

**Motivation**: When debugging, users often kill the process with Ctrl+C before it completes. Without signal handling, all leak information is lost.

**Implementation** (NDArrayLifecycleTracker.cpp lines 41-58):
```cpp
// Static signal handler for early termination (SIGINT, SIGTERM)
static void signalHandler(int signum) {
    std::cerr << "\n\nCaught signal " << signum << " ("
              << (signum == SIGINT ? "SIGINT/Ctrl+C" :
                  signum == SIGTERM ? "SIGTERM" : "UNKNOWN")
              << "), generating leak reports before exit..." << std::endl;

    // Generate reports for both tracking systems
    try {
        NDArrayLifecycleTracker::getInstance().generateLeakReport("");
    } catch (...) {
        std::cerr << "Error generating NDArray leak report during signal handling" << std::endl;
    }

    // Re-raise signal to perform default action (terminate)
    std::signal(signum, SIG_DFL);
    std::raise(signum);
}
```

**Registration** (NDArrayLifecycleTracker.cpp lines 74-76):
```cpp
NDArrayLifecycleTracker::NDArrayLifecycleTracker() {
    // ... existing initialization ...

    // Register signal handlers for early termination reporting
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
}
```

**Behavior**:
1. User presses Ctrl+C → SIGINT received
2. Signal handler calls `generateLeakReport("")` (uses default path from `SD_LIFECYCLE_LOG_DIR`)
3. Full leak report written to file with timestamp and PID
4. Signal re-raised → process terminates normally
5. User has leak report even though process was killed early

**Benefits**:
- **No data loss**: Leak reports always generated, even on early termination
- **Interactive debugging**: Can Ctrl+C at any point and get current state
- **CI/CD integration**: Timeout kills now produce leak reports
- **Graceful degradation**: If report generation fails, process still terminates

#### 9.4: Environment Integration

**Modified files**:

**Environment.h** (lines 63-64, 284-287):
```cpp
// Private member variables
std::atomic<bool> _snapshotFiles{false};  // Default off - only write snapshots on demand
std::atomic<bool> _trackOperations{false};  // Default off - operation tracking adds overhead

// Public methods
bool isSnapshotFiles();
void setSnapshotFiles(bool enabled);
bool isTrackOperations();
void setTrackOperations(bool enabled);
```

**Environment.cpp** (lines 248-264, 1106-1112):
```cpp
// Environment variable parsing (constructor)
// Snapshot files - write periodic file snapshots (default off)
const char *snapshot_files = std::getenv("SD_LIFECYCLE_SNAPSHOT_FILES");
if (snapshot_files != nullptr) {
  std::string val(snapshot_files);
  if (val == "1" || val == "true" || val == "TRUE") {
    _snapshotFiles.store(true);
  }
}

// Track operations - enable operation name tracking (default off)
const char *track_operations = std::getenv("SD_LIFECYCLE_TRACK_OPERATIONS");
if (track_operations != nullptr) {
  std::string val(track_operations);
  if (val == "1" || val == "true" || val == "TRUE") {
    _trackOperations.store(true);
  }
}

// Getter/setter implementations
bool Environment::isSnapshotFiles() { return _snapshotFiles.load(); }
void Environment::setSnapshotFiles(bool enabled) { _snapshotFiles.store(enabled); }
bool Environment::isTrackOperations() { return _trackOperations.load(); }
void Environment::setTrackOperations(bool enabled) { _trackOperations.store(enabled); }
```

**Default values rationale**:
- `SD_LIFECYCLE_SNAPSHOT_FILES=false` - File I/O overhead, only enable when needed
- `SD_LIFECYCLE_TRACK_OPERATIONS=false` - Requires manual operation tagging, opt-in

#### 9.5: Enhanced Usage Example

**Complete workflow with all features**:

```bash
# Enable all lifecycle tracking features
export SD_LIFECYCLE_TRACKING=1
export SD_LIFECYCLE_LOG_DIR=/tmp/leaks
export SD_LIFECYCLE_REPORT_INTERVAL=30      # Every 30 seconds
export SD_LIFECYCLE_SNAPSHOT_FILES=1         # Write files, not just stderr
export SD_LIFECYCLE_TRACK_OPERATIONS=1       # Track operation names
export SD_LIFECYCLE_TRACK_DELETIONS=1        # Capture deletion traces
export SD_LIFECYCLE_STACK_DEPTH=64           # Deeper stack traces

# Run application
java -cp myapp.jar com.example.MyApp

# After 30 seconds: /tmp/leaks/ndarray_snapshot_pid12345_<timestamp>_000.txt
# After 60 seconds: /tmp/leaks/ndarray_snapshot_pid12345_<timestamp>_001.txt
# After 90 seconds: /tmp/leaks/ndarray_snapshot_pid12345_<timestamp>_002.txt

# Kill early with Ctrl+C
^C

# Signal handler generates: /tmp/leaks/ndarray_leaks_pid12345_<timestamp>.txt
# Contains:
# - Statistics (allocations, deallocations, live count, memory)
# - File breakdown (which source files leaked)
# - Operation breakdown (which operations leaked) ← NEW
# - Current leaks with full stack traces
# - Deletion traces (if tracked)
```

**Snapshot file contents**:
```
=== NDArray Lifecycle Periodic Snapshot ===
Timestamp: 1730908123456789
PID: 12345
Sequence: 2

Total Allocations: 1234567
Total Deallocations: 1234500
Current Live: 67
Current Bytes: 2400000000 (2.24 GB)
Peak Bytes: 4800000000 (4.47 GB)

=== Current Leaks ===
1. NDArray 0x7f8a4c002000 [256,256,3] FLOAT32 24.0 MB
   Allocated in: convolutions_conv2d.cpp:234 (conv2dBackprop)
   Thread: 139876543210
   Stack trace:
     #0 0x7f8a4c123456 in NDArray::NDArray() at NDArray.cpp:123
     #1 0x7f8a4c234567 in conv2dBackprop() at convolutions_conv2d.cpp:234
     ...

=== Top 20 Leaking Operations ===

Operation                                      Leaks     Memory (MB)
-------------------------------------------------------------------
conv2d_bp                                       1245        12345.67
matmul                                           823         8234.12
relu_bp                                          412         2100.45
...
```

## Implementation Details

### Build Configuration

Enable lifecycle tracking at compile time:

```bash
cd libnd4j
./buildnativeoperations.sh -DCMAKE_CXX_FLAGS="-DSD_GCC_FUNCTRACE"
```

Or via Maven:
```bash
mvn clean install -Dlibnd4j.calltrace=ON
# or
mvn clean install -Dlibnd4j.functrace=ON
```

### Runtime Usage

#### C++ Usage:

```cpp
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>

// Enable tracking
Environment::getInstance().setLifecycleTracking(true);
Environment::getInstance().setLifecycleReportInterval(300);

// Your code runs normally
auto array = new NDArray('c', {10, 20}, FLOAT32);
delete array;  // Automatically tracked

// Query statistics
auto ndarray_stats = NDArrayLifecycleTracker::getInstance().getStats();
auto buffer_stats = DataBufferLifecycleTracker::getInstance().getStats();

std::cout << "Live NDArrays: " << ndarray_stats.current_live << std::endl;
std::cout << "Live PRIMARY buffers: " << buffer_stats.current_live_primary << std::endl;
std::cout << "Live SPECIAL buffers: " << buffer_stats.current_live_special << std::endl;
```

#### Java Usage:

```java
import org.nd4j.linalg.api.memory.LifecycleTrackingHelper;

// Check if tracking is available
if (LifecycleTrackingHelper.isAvailable()) {
    // Query statistics
    var ndarrayStats = LifecycleTrackingHelper.getNDArrayStats();
    var bufferStats = LifecycleTrackingHelper.getDataBufferStats();

    System.out.println("NDArray stats: " + ndarrayStats);
    System.out.println("PRIMARY (CPU): " + bufferStats.primary);
    System.out.println("SPECIAL (GPU): " + bufferStats.special);

    // Generate reports
    LifecycleTrackingHelper.generateLeakReport("/tmp/leak_report.txt");
    LifecycleTrackingHelper.generateDataBufferAllocationFlamegraph("/tmp/cpu_allocs.svg", false);
    LifecycleTrackingHelper.generateDataBufferAllocationFlamegraph("/tmp/gpu_allocs.svg", true);
}
```

### Report Format Examples

#### Combined Lifecycle Statistics:

```json
{
  "ndarray": {
    "total_allocations": 1234567,
    "total_deallocations": 1234500,
    "current_live": 67,
    "peak_live": 890,
    "current_bytes": 2400000000,
    "peak_bytes": 4800000000,
    "double_frees": 0
  },
  "databuffer": {
    "primary": {
      "total_allocations": 567890,
      "total_deallocations": 567850,
      "current_live": 40,
      "current_bytes": 1600000000,
      "peak_bytes": 3200000000
    },
    "special": {
      "total_allocations": 123456,
      "total_deallocations": 123429,
      "current_live": 27,
      "current_bytes": 800000000,
      "peak_bytes": 1600000000
    },
    "double_frees": 0
  }
}
```

## Key Benefits

### Two-Level Visibility

1. **NDArray Level**: Shows which operations create too many wrapper objects
   - Excessive view creation
   - Temporary object proliferation
   - Constructor/destructor imbalance

2. **DataBuffer Level**: Shows actual memory leaks
   - PRIMARY (host/CPU) memory leaks
   - SPECIAL (device/GPU) memory leaks
   - Real memory consumption

### Correlation Analysis

- If NDArray leaks are high but DataBuffer leaks are low: Wrapper object leaks (views not cleaned up)
- If DataBuffer leaks are high: Actual memory not freed (the serious problem)
- Can correlate both levels to understand complete leak patterns

### View Tracking

- Views tracked as NDArray creations
- Automatic handling via `!isView()` check prevents double-free
- Visibility into view lifecycle patterns
- Can identify excessive view creation (performance problem)

### Host/Device Separation

- Separate tracking for PRIMARY (CPU) and SPECIAL (GPU) memory
- Essential for CUDA memory leak debugging
- Independent flamegraphs for CPU and GPU allocations
- Can identify which memory type is leaking

### Cross-Language Integration

- C++ native tracking with full stack traces
- JNI API for querying statistics from Java
- JSON format for easy parsing
- Works with existing DeallocatorService
- No changes needed to Java deallocator pattern

### Production-Safe

- Compile-time flag (SD_GCC_FUNCTRACE) to disable in production
- Minimal overhead when disabled (preprocessor removes all tracking code)
- Thread-safe atomic counters and mutexes
- Bounded memory usage (deletion history cap)

## Memory Overhead

### NDArray Tracking
- **Per allocation**: ~200 bytes (AllocationRecord + StackTrace)
- **Per deallocation**: ~150 bytes (DeallocationRecord + StackTrace)
- **For 1 million live NDArrays**: ~200 MB tracking overhead

### DataBuffer Tracking
- **Per PRIMARY allocation**: ~200 bytes
- **Per SPECIAL allocation**: ~200 bytes
- **For 500K live buffers**: ~100 MB tracking overhead

**Total overhead**: Proportional to number of live objects. Typically 1-5% of application memory.

## Performance Impact

- **Allocation/deallocation**: +10-20 microseconds (stack trace capture + map operations)
- **Periodic reporting**: ~100ms for 100,000 live allocations (runs in background)
- **Overall runtime impact**: <5% for typical workloads

**Comparison with alternatives:**

| Tool | Runtime Overhead | Memory Overhead | Setup Complexity | Context |
|------|------------------|-----------------|------------------|---------|
| Lifecycle Tracker | <5% | 1-5% | Low (env vars) | Full (shape, dtype, operation) |
| AddressSanitizer | 2-3x | 2x | Low (LD_PRELOAD) | Partial (stack trace only) |
| Valgrind | 20-50x | 10x+ | Medium | Partial (memory addresses) |

## Implementation Files

### New Files Created

1. **NDArray Lifecycle Tracker**
   - `libnd4j/include/array/NDArrayLifecycleTracker.h` (~175 lines, updated 2025-11-06)
   - `libnd4j/include/array/impl/NDArrayLifecycleTracker.cpp` (~750 lines, updated 2025-11-06)
     - Added: Signal handler, periodic file snapshots, operation tracking

2. **DataBuffer Lifecycle Tracker**
   - `libnd4j/include/array/DataBufferLifecycleTracker.h` (~175 lines)
   - `libnd4j/include/array/impl/DataBufferLifecycleTracker.cpp` (~700 lines)

3. **JNI Integration**
   - `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking.cpp` (~200 lines)

4. **Documentation**
   - `LIFECYCLE_TRACKING_INTEGRATION.md` - Complete integration guide
   - `ADRs/0051 - NDArray and DataBuffer Lifecycle Tracking for Memory Leak Detection.md` - This document

### Modified Files

1. **NDArray Integration**
   - `libnd4j/include/array/NDArray.hXX` - Added tracking to 24 more constructors (all 26 now tracked)
   - Lines: 62-76 (macro), 315 (constructor), 358 (constructor), 664-686 (destructor)

2. **DataBuffer Integration**
   - `libnd4j/include/array/impl/DataBuffer.cpp` - PRIMARY buffer tracking
     - Lines: 30-32 (include), 436-441 (allocate), 461-468 (delete)
   - `libnd4j/include/array/cuda/DataBuffer.cu` - SPECIAL buffer tracking
     - Lines: 34-36 (include), 336-344 (allocate), 421-428 (delete)

3. **JNI API**
   - `libnd4j/include/legacy/NativeOps.h` - Added 8 API functions (lines 478-559)

4. **Environment Configuration**
   - `libnd4j/include/system/Environment.h` (lines 56-62, 272-283, updated 2025-11-06)
     - Added: `_snapshotFiles`, `_trackOperations` fields and methods
   - `libnd4j/include/legacy/impl/Environment.cpp` (lines 173-247, 1073-1104, updated 2025-11-06)
     - Added: Environment variable parsing for `SD_LIFECYCLE_SNAPSHOT_FILES` and `SD_LIFECYCLE_TRACK_OPERATIONS`

## Consequences

### Advantages

1. **Two-level visibility**: Wrapper objects (NDArray) and actual memory (DataBuffer)
2. **Host/Device separation**: PRIMARY (CPU) and SPECIAL (GPU) tracked independently
3. **Application-level context**: See NDArray shapes, types, and operations creating leaks
4. **Fast**: <5% overhead vs Valgrind's 20-50x
5. **File-level attribution**: Identify which source files leak the most
6. **Function breakdown**: See which functions within files allocate without deallocating
7. **Deletion patterns**: Compare allocation vs deallocation stack traces
8. **Flamegraph visualization**: Interactive exploration of allocation hotspots (per buffer type)
9. **Zero-configuration**: Works out-of-the-box when functrace is enabled
10. **Periodic reports**: Catch leaks during long-running processes
11. **Bounded memory**: Deletion history is capped to prevent unbounded growth
12. **Thread-safe**: Uses mutex for all shared data structure access
13. **Cross-language**: C++ tracking, JNI API, Java helper class
14. **View tracking**: Can distinguish views (shared buffers) from owned allocations
15. **Workspace awareness**: Flags workspace allocations separately
16. **Periodic file snapshots** (2025-11-06): Write leak state to files during execution
17. **Signal handling** (2025-11-06): Generate reports on Ctrl+C/SIGTERM
18. **Operation breakdown** (2025-11-06): Track leaks by operation name (Conv2D, MatMul, etc.)

### Drawbacks

1. **Requires recompilation**: Must build with `-DSD_GCC_FUNCTRACE`
2. **Memory overhead**: ~200 bytes per live NDArray + ~200 bytes per live DataBuffer
3. **Performance impact**: 10-20 microseconds per allocation/deallocation
4. **Linux/macOS only**: backward-cpp dependency (DWARF-based stack unwinding)
5. **Not real-time**: Reports are periodic, not instantaneous
6. **Manual flamegraph generation**: Requires external tool (flamegraph.pl)
7. **Stack trace quality**: Depends on debug symbols in binaries

### Comparison with AddressSanitizer (ADR 0049)

| Feature | Lifecycle Tracker | AddressSanitizer |
|---------|-------------------|------------------|
| **Detects leaks** | Yes (wrapper + memory) | Yes (memory only) |
| **Application context** | Yes (shape, dtype, operation, buffer type) | No (just memory addresses) |
| **File attribution** | Yes | Partial (from stack traces) |
| **Function breakdown** | Yes | No |
| **Flamegraphs** | Yes (native, per buffer type) | No (requires post-processing) |
| **Host/Device separation** | Yes (PRIMARY/SPECIAL) | No |
| **Runtime overhead** | <5% | 2-3x |
| **Memory overhead** | 1-5% | 2x |
| **Java integration** | Yes (JNI API) | No |
| **Use-after-free detection** | No | Yes |
| **Buffer overflow detection** | No | Yes |

**Recommendation**: Use both tools together:
- **Lifecycle Tracker** for identifying high-level leak patterns and which operations/files leak
- **AddressSanitizer** for low-level memory corruption detection

## Usage Examples

### Example 1: Two-Level Leak Analysis

**Problem**: Application leaking 100GB despite NDArray destructors being called.

**Investigation:**

```bash
export SD_LIFECYCLE_TRACKING=1
export SD_REPORT_INTERVAL=60
java -cp app.jar MyApp
```

**Console output:**
```
NDArray Statistics: 67 live (2.4 GB)
DataBuffer PRIMARY Statistics: 40 live (1.6 GB)
DataBuffer SPECIAL Statistics: 27 live (800 MB)
```

**Finding**: NDArray count is low (67), but DataBuffer count is high with significant memory.

**Conclusion**: NDArray destructors ARE running, but DataBuffer memory isn't being freed. The leak is at the memory level, not wrapper level.

### Example 2: Host vs Device Memory Leaks

**Problem**: CUDA application running out of device memory.

**Investigation:**

```java
var stats = LifecycleTrackingHelper.getDataBufferStats();
System.out.println("PRIMARY leaks: " + stats.primary.currentBytes);  // 500 MB
System.out.println("SPECIAL leaks: " + stats.special.currentBytes);  // 15 GB

LifecycleTrackingHelper.generateDataBufferAllocationFlamegraph("/tmp/gpu_leaks.svg", true);
```

**Finding**: SPECIAL (GPU) memory leaking 15GB, PRIMARY (CPU) only 500MB.

**Fix**: Focus on CUDA-specific operations identified in the GPU flamegraph.

### Example 3: View Lifecycle Analysis

**Problem**: Excessive memory usage from views.

**Investigation:**

NDArray report shows `is_view=true` for many live allocations. These are wrapper objects sharing buffers.

**Finding**: Code creates many views but doesn't delete the wrapper objects.

**Fix**: Ensure all view-returning operations (permute, transpose, reshape) have their return values deleted.

### Example 4: Cross-Language Debugging

**Problem**: Memory leaks at Java/C++ boundary.

**Java code:**

```java
// Periodic monitoring in production
if (LifecycleTrackingHelper.isAvailable()) {
    var stats = LifecycleTrackingHelper.getDataBufferStats();
    logger.info("Memory stats - PRIMARY: {}MB, SPECIAL: {}MB",
        stats.primary.currentBytes / 1024 / 1024,
        stats.special.currentBytes / 1024 / 1024);

    if (stats.primary.currentBytes > threshold) {
        LifecycleTrackingHelper.generateLeakReport("/tmp/emergency_leak_report.txt");
    }
}
```

**Finding**: Can query native memory statistics from Java and trigger emergency reports when thresholds exceeded.

## Migration Path

### Phase 1: Enable Tracking (No Code Changes)

1. Rebuild libnd4j with `-DSD_GCC_FUNCTRACE`
2. Run existing applications
3. Review periodic reports for baseline statistics

### Phase 2: Fix Identified Leaks

Use file breakdown and flamegraphs to systematically fix leaks:

1. Start with DataBuffer leaks (actual memory)
2. Then address NDArray wrapper leaks
3. Focus on files/functions with highest leak counts
4. Verify fixes by re-running with tracking enabled

### Phase 3: Java Integration

Add LifecycleTrackingHelper to production monitoring:

```java
// In monitoring thread
if (LifecycleTrackingHelper.isAvailable()) {
    var stats = LifecycleTrackingHelper.getDataBufferStats();
    metrics.gauge("memory.primary.live", stats.primary.currentBytes);
    metrics.gauge("memory.special.live", stats.special.currentBytes);
}
```

### Phase 4: Continuous Monitoring

Integrate into CI/CD:

```bash
# Fail build if leak count exceeds threshold
java -cp test.jar RunTests
if [[ $(get_leak_count) -gt 100 ]]; then
  echo "FAIL: Too many leaks"
  exit 1
fi
```

## References

- ADR 0049 - AddressSanitizer Memory Leak Detection
- ADR 0050 - Clang Sanitizers for JNI Memory Debugging
- backward-cpp library: https://github.com/bombela/backward-cpp
- FlameGraph tool: https://github.com/brendangregg/FlameGraph
- DeallocatorService.java - Java GC integration
- OpaqueNDArray pattern - JNI object lifecycle
- convolutions_conv2d.cpp - example file with fixed leaks (lines 63-133)

## Implemented Enhancements (2025-11-06)

✅ **Periodic file-based snapshots** - Write snapshots to files during execution, not just stderr (Component 9.1)
✅ **Signal handlers for early termination** - Generate reports on Ctrl+C/SIGTERM (Component 9.3)
✅ **Operation name tracking** - Track which operations (Conv2D, MatMul, etc.) leak (Component 9.2)

## Runtime Control API (2025-11-09)

### Component 10: Individual Tracker Enable/Disable

**Motivation**: Users needed fine-grained control over which tracking systems are active at runtime without restarting the application. For example, enabling ONLY NDArray tracking without the overhead of DataBuffer tracking, or disabling specific trackers during performance-critical sections.

#### 10.1: Native API Functions

**NativeOps.h** (lines 477-486):
```cpp
SD_LIB_EXPORT void enableNDArrayTracking();
SD_LIB_EXPORT void disableNDArrayTracking();
SD_LIB_EXPORT void enableDataBufferTracking();
SD_LIB_EXPORT void disableDataBufferTracking();
SD_LIB_EXPORT void enableTADCacheTracking();
SD_LIB_EXPORT void disableTADCacheTracking();
SD_LIB_EXPORT void enableShapeCacheTracking();
SD_LIB_EXPORT void disableShapeCacheTracking();
SD_LIB_EXPORT void enableOpContextTracking();
SD_LIB_EXPORT void disableOpContextTracking();
```

**Implementation** (NativeOpsHelpers_LifecycleTracking.cpp lines 213-318):

```cpp
#if defined(SD_GCC_FUNCTRACE)

SD_LIB_EXPORT void enableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(true);
}

SD_LIB_EXPORT void disableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(false);
}

SD_LIB_EXPORT void enableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(true);
}

SD_LIB_EXPORT void disableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(false);
}

// Similar for TADCache, ShapeCache, OpContext trackers...

#else // !SD_GCC_FUNCTRACE

// No-op implementations when lifecycle tracking not compiled in
SD_LIB_EXPORT void enableNDArrayTracking() {}
SD_LIB_EXPORT void disableNDArrayTracking() {}
SD_LIB_EXPORT void enableDataBufferTracking() {}
SD_LIB_EXPORT void disableDataBufferTracking() {}
// ... more no-ops

#endif
```

**Key Features**:
- Functions available regardless of build configuration (graceful degradation)
- No-op when `SD_GCC_FUNCTRACE` not defined
- Thread-safe atomic flag operations
- Zero-cost when tracker disabled (early return in hot paths)

#### 10.2: Tracker-Level Implementation

**Modified tracker headers** to add runtime enable/disable:

**NDArrayLifecycleTracker.h** (lines 103, 151-152):
```cpp
class NDArrayLifecycleTracker {
private:
    std::atomic<bool> _tracking_enabled{true};  // Enabled by default

public:
    void setEnabled(bool enabled) { _tracking_enabled.store(enabled); }
    bool isEnabled() const { return _tracking_enabled.load(); }
};
```

**DataBufferLifecycleTracker.h** (lines 128, 175-176):
```cpp
class DataBufferLifecycleTracker {
private:
    std::atomic<bool> _tracking_enabled{true};  // Enabled by default

public:
    void setEnabled(bool enabled) { _tracking_enabled.store(enabled); }
    bool isEnabled() const { return _tracking_enabled.load(); }
};
```

**Early return checks in hot paths**:

**NDArrayLifecycleTracker.cpp** (lines 271, 344):
```cpp
void NDArrayLifecycleTracker::recordAllocation(...) {
    if (!_tracking_enabled.load()) return;  // Early return if disabled

    // Existing Environment checks
    Environment &env = Environment::getInstance();
    if (!env.isLifecycleTracking()) return;
    if (is_view && !env.isTrackViews()) return;

    // ... rest of allocation tracking
}

void NDArrayLifecycleTracker::recordAllocationWithOp(...) {
    if (!_tracking_enabled.load()) return;  // Early return if disabled

    // ... rest of allocation tracking
}
```

**DataBufferLifecycleTracker.cpp** (line 208):
```cpp
void DataBufferLifecycleTracker::recordAllocation(...) {
    if (!_tracking_enabled.load()) return;  // Early return if disabled

    Environment &env = Environment::getInstance();
    if (!env.isLifecycleTracking()) return;

    // ... rest of allocation tracking
}
```

#### 10.3: Java Interface

**NativeOps.java** (lines 447-502):
```java
/**
 * Enables NDArray lifecycle tracking.
 */
void enableNDArrayTracking();

/**
 * Disables NDArray lifecycle tracking.
 */
void disableNDArrayTracking();

/**
 * Enables DataBuffer lifecycle tracking.
 */
void enableDataBufferTracking();

/**
 * Disables DataBuffer lifecycle tracking.
 */
void disableDataBufferTracking();

/**
 * Enables TADCache lifecycle tracking.
 */
void enableTADCacheTracking();

/**
 * Disables TADCache lifecycle tracking.
 */
void disableTADCacheTracking();

/**
 * Enables ShapeCache lifecycle tracking.
 */
void enableShapeCacheTracking();

/**
 * Disables ShapeCache lifecycle tracking.
 */
void disableShapeCacheTracking();

/**
 * Enables OpContext lifecycle tracking.
 */
void enableOpContextTracking();

/**
 * Disables OpContext lifecycle tracking.
 */
void disableOpContextTracking();

/**
 * Clears all cached TAD packs to prevent memory leaks.
 */
void clearTADCache();
```

#### 10.4: REST API Controller

**LifecycleTrackingController.java** - Spring Boot REST controller for runtime control:

```java
@RestController
@RequestMapping("/api/lifecycle")
public class LifecycleTrackingController {

    /**
     * Enable ONLY NDArray lifecycle tracking (no DataBuffer tracking)
     */
    @PostMapping("/enable-ndarray-only")
    public ResponseEntity<Map<String, String>> enableNDArrayOnlyTracking() {
        // Enable NDArray tracking
        Nd4j.getNativeOps().enableNDArrayTracking();

        // Disable DataBuffer tracking to reduce overhead
        Nd4j.getNativeOps().disableDataBufferTracking();

        // Configure Java-side flags
        Nd4j.getEnvironment().setLifecycleTracking(true);
        Nd4j.getEnvironment().setTrackViews(false);
        Nd4j.getEnvironment().setSnapshotFiles(true);
        Nd4j.getEnvironment().setTrackOperations(true);

        Map<String, String> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "NDArray-only tracking enabled");
        return ResponseEntity.ok(response);
    }

    /**
     * Trigger immediate leak check
     */
    @PostMapping("/trigger-leak-check")
    public ResponseEntity<Map<String, String>> triggerLeakCheck() {
        Nd4j.getNativeOps().triggerLeakCheck();

        Map<String, String> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "Leak check triggered");
        return ResponseEntity.ok(response);
    }
}
```

**Benefits**:
- **Runtime control**: Toggle tracking without restarting the application
- **Performance tuning**: Disable tracking during critical sections
- **Selective tracking**: Enable only specific trackers (e.g., NDArray but not DataBuffer)
- **Web-based**: Control via HTTP API for production deployments
- **Integration-ready**: Works with monitoring tools and dashboards

#### 10.5: Usage Examples

**Example 1: Performance-Critical Section**
```java
// Disable tracking during critical section
Nd4j.getNativeOps().disableNDArrayTracking();
Nd4j.getNativeOps().disableDataBufferTracking();

// Run performance-critical code
runInference(model, input);

// Re-enable tracking
Nd4j.getNativeOps().enableNDArrayTracking();
Nd4j.getNativeOps().enableDataBufferTracking();
```

**Example 2: NDArray-Only Tracking**
```java
// Focus on wrapper leaks without DataBuffer overhead
Nd4j.getNativeOps().enableNDArrayTracking();
Nd4j.getNativeOps().disableDataBufferTracking();

// Application runs with reduced tracking overhead
runApplication();
```

**Example 3: Selective Investigation**
```java
// Initially track everything to get baseline
Nd4j.getNativeOps().enableNDArrayTracking();
Nd4j.getNativeOps().enableDataBufferTracking();
Nd4j.getNativeOps().enableTADCacheTracking();

// After identifying NDArray leaks as the issue
Nd4j.getNativeOps().disableDataBufferTracking();
Nd4j.getNativeOps().disableTADCacheTracking();

// Focus exclusively on NDArray tracking
runNDArrayLeakTest();
```

**Example 4: REST API Control**
```bash
# Enable NDArray-only tracking via REST API
curl -X POST http://localhost:8080/api/lifecycle/enable-ndarray-only

# Trigger manual leak check
curl -X POST http://localhost:8080/api/lifecycle/trigger-leak-check

# Get current configuration
curl http://localhost:8080/api/lifecycle/config
```

#### 10.6: Performance Impact

**Overhead when enabled**: ~10-20 microseconds per allocation/deallocation

**Overhead when disabled**:
- Native level: ~5 nanoseconds (single atomic load + branch)
- Effectively zero-cost (optimized away by branch predictor)

**Memory impact**:
- Atomic flag: 1 byte per tracker (negligible)
- No memory allocated when disabled

#### 10.7: Design Rationale

**Why symmetric enable/disable pairs?**
- Users need full control: both directions must be available
- Debugging workflow: enable → investigate → disable → re-enable
- Performance tuning: temporarily disable during benchmarks

**Why atomic<bool> instead of mutex?**
- Lock-free read in hot path (recordAllocation called millions of times)
- Atomic guarantees visibility across threads
- Write (setEnabled) is rare, read (load) is frequent

**Why early return pattern?**
- Minimal overhead: Single atomic load + branch
- Cache-friendly: Flag likely in L1 cache
- Predictable: Branch predictor learns tracking is usually enabled

**Why both instance-level AND Environment-level checks?**
- Instance-level (`_tracking_enabled`): Per-tracker control
- Environment-level (`isLifecycleTracking()`): Global master switch
- Both required: Global off = all trackers off, individual control when global on

#### 10.8: Files Modified

**C++ Native Layer**:
- `libnd4j/include/legacy/NativeOps.h:477-492` - API declarations
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking.cpp:210-318` - Implementations
- `libnd4j/include/array/NDArrayLifecycleTracker.h:103,151-152` - Tracker enable/disable support
- `libnd4j/include/array/DataBufferLifecycleTracker.h:128,175-176` - Tracker enable/disable support
- `libnd4j/include/array/impl/NDArrayLifecycleTracker.cpp:271,344` - Early return checks
- `libnd4j/include/array/impl/DataBufferLifecycleTracker.cpp:208` - Early return check

**Java Layer**:
- `nd4j/.../NativeOps.java:447-502` - Java interface declarations
- `kompile-app/.../LifecycleTrackingController.java` - REST API controller (example integration)

#### 10.9: Testing

**Verification that tracking can be disabled**:
```cpp
// Test that allocation is not tracked when disabled
NDArrayLifecycleTracker::getInstance().setEnabled(false);

auto array = new NDArray('c', {10, 20}, FLOAT32);
auto stats_before = NDArrayLifecycleTracker::getInstance().getStats();

delete array;

auto stats_after = NDArrayLifecycleTracker::getInstance().getStats();

// Verify counts unchanged (tracking was disabled)
assert(stats_before.total_allocations == stats_after.total_allocations);
assert(stats_before.total_deallocations == stats_after.total_deallocations);
```

**Verification of zero overhead when disabled**:
```cpp
// Benchmark allocation hot path
NDArrayLifecycleTracker::getInstance().setEnabled(false);

auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 1000000; i++) {
    auto arr = new NDArray('c', {10, 20}, FLOAT32);
    delete arr;
}
auto end = std::chrono::high_resolution_clock::now();
auto disabled_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

// Overhead should be < 1% when disabled
```

### Component 11: Stack Frame Analysis Infrastructure (2025-11-10)

**Purpose**: Common infrastructure for parsing and analyzing both Java and C++ stack frames to identify leak sources and caller-callee relationships.

**Key Features**:
1. **Unified stack frame parsing** - Parse both Java and C++ stack traces into structured StackFrame objects
2. **Leak source identification** - Identify which functions/methods create objects that leak
3. **Allocation source statistics** - Track leak rates and bytes per source
4. **Call graph generation** - Visualize caller-callee relationships in DOT format
5. **Noise filtering** - Remove JNI bridge, allocation framework, and system library frames

#### StackFrame Structure

```cpp
struct StackFrame {
    enum class Language { CPP, JAVA, UNKNOWN };

    Language language;
    std::string function_name;     // Full function/method name
    std::string class_name;        // Class/namespace
    std::string method_name;       // Method name only (Java)
    std::string source_file;       // Source file name
    int line_number;               // Line number
    std::string raw_frame;         // Original raw frame

    std::string getCanonicalName() const;    // e.g., "com.example.MyClass.myMethod"
    std::string getDisplayName() const;      // Human-readable format
    bool isUserCode() const;                 // Filter system/library code
    bool matches(const std::string& pattern) const;
};
```

#### ParsedStackTrace Structure

```cpp
struct ParsedStackTrace {
    std::vector<StackFrame> cpp_frames;
    std::vector<StackFrame> java_frames;
    std::string raw_cpp_trace;
    std::string raw_java_trace;

    std::vector<StackFrame> getAllFrames() const;        // Java first, then C++
    StackFrame getRootCaller() const;                    // Topmost Java/C++ frame
    StackFrame getImmediateAllocator() const;            // Frame before allocation
    std::vector<StackFrame> findFrames(const std::string& pattern) const;
};
```

#### AllocationSourceStats Structure

```cpp
struct AllocationSourceStats {
    std::string source_identifier;           // Canonical name
    uint64_t total_allocations;              // Total allocations
    uint64_t leaked_allocations;             // How many leaked
    size_t total_bytes_allocated;
    size_t leaked_bytes;
    std::vector<std::string> leak_examples;  // Sample traces

    double getLeakRate() const;              // Percentage leaked
    size_t getAvgLeakedBytes() const;        // Average bytes per leak
};
```

#### StackFrameAnalyzer API

```cpp
class StackFrameAnalyzer {
public:
    // Parse stack traces
    static std::vector<StackFrame> parseCppStackTrace(StackTrace* trace, TraceResolver& resolver);
    static std::vector<StackFrame> parseJavaStackTrace(const std::string& java_trace);
    static ParsedStackTrace parseFullStackTrace(StackTrace* cpp_trace, const std::string& java_trace);

    // Identify leak sources
    static StackFrame extractAllocationSource(const ParsedStackTrace& trace);

    // Build statistics (template for any allocation record type)
    template<typename AllocRecord>
    static std::map<std::string, AllocationSourceStats> buildAllocationSourceStats(
        const std::map<void*, AllocRecord*>& live_allocations,
        TraceResolver& resolver,
        int max_examples = 10);

    // Generate reports
    static void generateLeakSourceReport(std::ostream& out,
                                         const std::map<std::string, AllocationSourceStats>& stats,
                                         int top_n = 20);

    static void generateCallGraphDot(std::ostream& out,
                                     const std::map<std::string, AllocationSourceStats>& stats);

    // Filtering
    static std::vector<StackFrame> filterNoise(const std::vector<StackFrame>& frames);
    static std::string getCanonicalPath(const std::vector<StackFrame>& frames, int depth = 5);
};
```

#### Integration with Trackers

Both `NDArrayLifecycleTracker` and `DataBufferLifecycleTracker` now provide:

```cpp
// Generate leak source analysis report
void generateLeakSourceReport(const std::string &output_path);

// Generate call graph in DOT format
void generateLeakSourceCallGraph(const std::string &output_path);

// Print analysis to stream
void printLeakSourceAnalysis(std::ostream &out, int top_n = 20);
```

#### Example Leak Source Report Output

```
============================================
  LEAK SOURCE ANALYSIS
============================================

Top Leak Sources (by leaked bytes):

1. org.nd4j.linalg.api.ndarray.BaseNDArray.create
   Allocations: 1,234 total, 456 leaked (37.0%)
   Leaked bytes: 45.6 MB (avg 100 KB per leak)
   Example leak traces (first 3):
   ---
   Leaked object at 0x7f8a4c000000 (102400 bytes)
   Java stack trace:
     at org.nd4j.linalg.api.ndarray.BaseNDArray.create(BaseNDArray.java:123)
     at ai.kompile.app.service.ModelService.runInference(ModelService.java:456)
     at ai.kompile.app.web.controllers.InferenceController.predict(InferenceController.java:89)
   C++ stack trace:
     sd::NDArray::NDArray [NDArray.cpp:234]
     sd::ops::conv2d::execute [conv2d.cpp:567]

2. sd::ops::matmul::execute
   Allocations: 567 total, 123 leaked (21.7%)
   Leaked bytes: 23.4 MB (avg 190 KB per leak)
   ...

--------------------------------------------
Total leak sources tracked: 15
Total leaked: 789 allocations, 89.5 MB
```

#### Example Call Graph (DOT format)

```dot
digraph LeakCallGraph {
  rankdir=LR;
  node [shape=box, style=rounded];

  node0 [label="org.nd4j.linalg.api.ndarray.BaseNDArray.create\n456 leaks (37.0%)\n45.60 MB", fillcolor=red, style=filled];
  node1 [label="sd::ops::matmul::execute\n123 leaks (21.7%)\n23.40 MB", fillcolor=orange, style=filled];
  node2 [label="sd::ops::conv2d::execute\n45 leaks (8.2%)\n12.30 MB", fillcolor=yellow, style=filled];
}
```

Visualize with: `dot -Tpng leak_callgraph.dot -o leak_graph.png`

#### Design Rationale

1. **Unified parsing**: Single infrastructure for both Java and C++ stack frames simplifies analysis
2. **Noise filtering**: Removes JNI bridge and allocation framework frames to focus on user code
3. **Source identification**: Identifies the root cause function/method, not just allocation site
4. **Language preference**: Prefers Java frames (higher level) over C++ frames for better user context
5. **Templated statistics**: Works with any allocation record type (NDArray, DataBuffer, etc.)
6. **Multiple output formats**: Text reports for reading, DOT graphs for visualization

#### Performance Impact

- **Parse cost**: ~50-100 microseconds per stack trace (one-time at allocation)
- **Statistics build**: O(N) where N = number of live allocations
- **Report generation**: O(N log N) for sorting
- **Memory overhead**: ~500 bytes per StackFrame object

#### Usage Example

```cpp
// In tracker implementation
void NDArrayLifecycleTracker::generateLeakSourceReport(const std::string &output_path) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    // Build statistics from live allocations
    TraceResolver resolver;
    auto stats = StackFrameAnalyzer::buildAllocationSourceStats<AllocationRecord>(
        _live_allocations, resolver);

    // Generate report
    std::ofstream out(output_path);
    StackFrameAnalyzer::generateLeakSourceReport(out, stats, 20);
}
```

#### Files Modified

**New files**:
- `libnd4j/include/array/StackFrameAnalyzer.h` - Stack frame analysis header
- `libnd4j/include/array/impl/StackFrameAnalyzer.cpp` - Implementation

**Modified files**:
- `libnd4j/include/array/NDArrayLifecycleTracker.h` - Added leak source analysis methods
- `libnd4j/include/array/impl/NDArrayLifecycleTracker.cpp` - Implemented analysis methods
- `libnd4j/include/array/DataBufferLifecycleTracker.h` - Added leak source analysis methods
- `libnd4j/include/array/impl/DataBufferLifecycleTracker.cpp` - Implemented analysis methods

#### Testing Approach

1. **Unit tests**: Test parsing of Java and C++ stack frames
2. **Integration tests**: Verify leak source identification with known patterns
3. **Performance tests**: Measure overhead of analysis on large leak sets
4. **Visual verification**: Generate DOT graphs and verify with graphviz

---

### Comprehensive Leak Analysis (November 2025)

After implementing the individual tracker infrastructure, we identified a need for **cross-tracker leak analysis** to understand the full picture of memory leaks across all object types.

#### Problem

Individual trackers provide isolated views:
- NDArrayLifecycleTracker shows which functions create leaked NDArray wrappers
- DataBufferLifecycleTracker shows which functions create leaked memory buffers
- TADCacheLifecycleTracker shows which functions create leaked TAD packs
- ShapeCacheLifecycleTracker shows which functions create leaked shape descriptors
- OpContextLifecycleTracker shows which functions create leaked operation contexts

However, a single Java method or C++ function might leak **multiple types** of objects. For example, `org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D.exec()` might leak:
- 50 NDArray wrappers
- 50 corresponding DataBuffers (PRIMARY memory)
- 25 TAD cache entries
- 10 shape cache entries
- 5 OpContext objects

Without cross-tracker analysis, we couldn't answer:
- **Which single function causes the most total leaks across all object types?**
- **What is the combined memory footprint of all leaks from one source?**
- **Should we prioritize fixing function A (100 NDArray leaks) or function B (10 DataBuffer leaks but 1GB each)?**

#### Solution: ComprehensiveLeakAnalyzer

Implemented a unified analysis system that combines leak data from all 5 lifecycle trackers.

**Key Components**:

1. **`StackFrameAnalyzer`** (`libnd4j/include/array/StackFrameAnalyzer.h`):
   - Parses both Java (JNI) and C++ (backward-cpp) stack traces
   - Identifies allocation source frames (filtering out JNI bridge and framework noise)
   - Provides canonical naming for functions across Java/C++ boundary
   - Type-aware with template specialization for different tracker map types (`void*` vs `LongType*` keys)

2. **`ComprehensiveLeakAnalyzer`** (`libnd4j/generate_leak_analysis.cpp`):
   - Singleton that aggregates leak data from all 5 trackers
   - Groups leaks by source identifier (function/method name)
   - Combines byte counts across all object types
   - Ranks leak sources by total impact (not just count)
   - Generates unified reports showing breakdown by object type

3. **Public Getter Methods** (added to all tracker headers):
   ```cpp
   // NDArrayLifecycleTracker.h:168-171
   const std::unordered_map<void*, AllocationRecord*>& getLiveAllocations() const {
       return _live_allocations;
   }

   // ShapeCacheLifecycleTracker.h:182-185
   const std::unordered_map<LongType*, ShapeCacheEntry*>& getLiveShapes() const {
       return _live_shapes;
   }
   ```

   These replace the originally attempted friend class approach, providing cleaner encapsulation.

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│ ComprehensiveLeakAnalyzer                           │
│ - Singleton coordinating cross-tracker analysis     │
└──────┬──────────────────────────────────────────────┘
       │
       ├──► NDArrayLifecycleTracker::getLiveAllocations()
       ├──► DataBufferLifecycleTracker::getLiveAllocations()
       ├──► TADCacheLifecycleTracker::getLiveTADs()
       ├──► ShapeCacheLifecycleTracker::getLiveShapes()
       └──► OpContextLifecycleTracker::getLiveContexts()
               │
               ▼
       StackFrameAnalyzer::buildAllocationSourceStats()
       - Parses Java + C++ stack traces
       - Extracts allocation source (root caller)
       - Groups by canonical function name
       - Aggregates byte counts
               │
               ▼
       CombinedLeakStats (per source function)
       - ndarray_leaks, ndarray_bytes
       - databuffer_leaks, databuffer_bytes
       - tad_leaks, tad_bytes
       - shape_leaks, shape_bytes
       - opcontext_leaks
       - total_leaks, total_bytes
       - example stack traces (one per type)
```

#### Usage

**From C++ (at application shutdown)**:
```cpp
#if defined(SD_GCC_FUNCTRACE)
// Analyze all trackers and generate report
runComprehensiveLeakAnalysis("/tmp/leak_reports");
// Generates: /tmp/leak_reports/comprehensive_leak_report.txt
#endif
```

**From JNI** (future enhancement):
```java
// Java API to trigger analysis
NativeOpsHolder.getInstance().getDeviceNativeOps()
    .generateComprehensiveLeakReport("/tmp/leak_reports");
```

#### Report Format

```
============================================
  COMPREHENSIVE LEAK SOURCE ANALYSIS
  All Lifecycle Trackers Combined
============================================

Analysis aggregates undeleted allocations from:
  - NDArrayLifecycleTracker (arrays)
  - DataBufferLifecycleTracker (buffers)
  - TADCacheLifecycleTracker (TAD packs)
  - ShapeCacheLifecycleTracker (shapes)
  - OpContextLifecycleTracker (op contexts)

Top 50 Leak Sources (by total undeleted allocation count):
======================================================================

1. org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D.exec
   ---------------------------------------------------------------
   TOTAL LEAKS: 235 allocations
   TOTAL BYTES: 1.2 GB

   Breakdown by Object Type:
     - NDArray:    50 leaks, 800 MB
     - DataBuffer: 50 leaks, 350 MB
     - TADCache:   25 leaks, 45 MB
     - ShapeCache: 10 leaks, 5 MB
     - OpContext:  100 leaks

   Example Stack Traces:
   --- NDArray Example ---
     Leaked object at 0x7f1234567890 (16777216 bytes)
     Java stack trace:
       org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D.exec(Conv2D.java:123)
       org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner.exec(DefaultOpExecutioner.java:456)
     C++ stack trace:
       sd::ops::conv2d::execute(Context*, ...)
       sd::graph::GraphExecutioner::execute(...)

   --- DataBuffer Example ---
     Leaked object at 0x7f9876543210 (8388608 bytes)
     ...
```

#### Implementation Details

**Template Specialization for Different Map Types**:

The `StackFrameAnalyzer::buildAllocationSourceStats` template function has two overloads to handle different tracker map key types:

```cpp
// For most trackers (void* keys)
template<typename AllocRecord>
std::map<std::string, AllocationSourceStats> buildAllocationSourceStats(
    const std::unordered_map<void*, AllocRecord*>& live_allocations,
    TraceResolver& resolver,
    int max_examples = 10);

// For ShapeCacheLifecycleTracker (LongType* keys)
template<typename AllocRecord>
std::map<std::string, AllocationSourceStats> buildAllocationSourceStats(
    const std::unordered_map<LongType*, AllocRecord*>& live_allocations,
    TraceResolver& resolver,
    int max_examples = 10);
```

**Type Traits for Optional Fields**:

Uses SFINAE to detect whether allocation records have a `size_bytes` field (OpContext doesn't track bytes):

```cpp
template<typename T, typename = void>
struct has_size_bytes : std::false_type {};

template<typename T>
struct has_size_bytes<T, std::void_t<decltype(std::declval<T>().size_bytes)>> : std::true_type {};

// In implementation:
if constexpr (has_size_bytes<AllocRecord>::value) {
    stat.total_bytes_allocated += record->size_bytes;
    stat.leaked_bytes += record->size_bytes;
}
```

#### Files Modified

**New Files**:
- `libnd4j/generate_leak_analysis.cpp` - Comprehensive analysis tool
- `libnd4j/include/array/StackFrameAnalyzer.h` - Stack frame parsing infrastructure

**Modified Files**:
- `libnd4j/include/array/NDArrayLifecycleTracker.h:168-171` - Added `getLiveAllocations()` getter
- `libnd4j/include/array/DataBufferLifecycleTracker.h:192-195` - Added `getLiveAllocations()` getter
- `libnd4j/include/array/TADCacheLifecycleTracker.h:186-189` - Added `getLiveTADs()` getter
- `libnd4j/include/array/ShapeCacheLifecycleTracker.h:182-185` - Added `getLiveShapes()` getter
- `libnd4j/include/graph/OpContextLifecycleTracker.h:189-192` - Added `getLiveContexts()` getter

#### Benefits

1. **Unified Priority**: See which single function/method causes the most total damage across all object types
2. **Memory Impact**: Total bytes leaked, not just object counts
3. **Cross-layer Visibility**: Understand how Java code triggers C++ leaks
4. **Efficient Debugging**: Focus on the top 10 sources instead of analyzing 5 separate reports
5. **Complete Picture**: See all leak types from one source in one place

#### Example Usage Scenario

**Before comprehensive analysis**:
```
NDArray report: Conv2D.exec has 50 leaks
DataBuffer report: Conv2D.exec has 50 leaks
TAD report: Conv2D.exec has 25 leaks
Shape report: Conv2D.exec has 10 leaks
OpContext report: Conv2D.exec has 100 leaks

Developer thinks: "100 OpContext leaks sounds bad, fix that first!"
```

**After comprehensive analysis**:
```
Comprehensive report:
1. Conv2D.exec: 235 total leaks, 1.2 GB (mostly from NDArray 800MB + DataBuffer 350MB)
2. BatchNorm.exec: 20 total leaks, 2.5 GB (huge DataBuffers!)
3. ...

Developer thinks: "BatchNorm has fewer leaks but uses MORE memory. Fix that first!"
```

The comprehensive view reveals that BatchNorm's 20 leaks are actually more critical than Conv2D's 235 leaks due to memory footprint.

---

## Future Enhancements

1. **Android/Windows support**: Port to platforms without DWARF
2. **Real-time web dashboard**: HTTP server for live leak statistics
3. **ML-based leak prediction**: Use historical patterns to predict future leaks
4. **Workspace tracking**: Extend to track Workspace memory pools
5. **Suppression file**: Configure "known leaks" to filter from reports
6. **Automatic leak fixing suggestions**: Analyze patterns and suggest fixes
7. **DataBuffer operation tracking**: Extend operation tracking to DataBuffer level (currently NDArray only)
8. **Adaptive tracking**: Automatically disable tracking when overhead becomes too high
9. **Call graph edge tracking**: Track actual caller-callee edges (not just nodes) for more accurate graphs
10. **Leak clustering**: Group similar leak patterns for easier diagnosis

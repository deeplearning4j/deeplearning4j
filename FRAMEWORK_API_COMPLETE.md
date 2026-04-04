# ND4J Framework API - Complete Implementation

## Overview

A unified hierarchical API surface for handling ND4J framework internals including:
- Memory management (MemoryManager, DeallocatorService, MemoryTracker)
- Execution (DSP diagnostics, OpExecutioner, Kernel selection)
- Device management (AffinityManager, device info)
- CUDA memory pool (via native bindings - pending)
- Workspace management
- Op profiling and tracking
- Array lifecycle tracking

**Access Point**: `Nd4j.framework`

---

## Complete API Hierarchy

```
Nd4j.framework
│
├── execution()                      [ExecutionSubsystem]
│   ├── dsp()                        DSP diagnostics and plan tracking
│   │   ├── initialize()             Initialize from system properties
│   │   ├── enableCategories(mask)   Enable diagnostic categories
│   │   ├── enableAll()              Enable all categories
│   │   ├── disableCategories(mask)  Disable categories
│   │   ├── isEnabled(category)      Check if category enabled
│   │   ├── setLevel(level)          Set detail level (0-2)
│   │   ├── setJsonOutputPath(path)  Set JSON output file
│   │   ├── record(cat, message)     Record diagnostic event
│   │   ├── recordSlot(...)          Record slot-specific event
│   │   ├── getPlanReport()          Get structured text report
│   │   ├── getJsonReport()          Get JSON report
│   │   ├── clear()                  Clear all events
│   │   ├── stats()                  Get DSP statistics
│   │   └── introspection()          DSP plan introspection (NEW)
│   │       ├── getSummary(plan)              Get plan summary
│   │       ├── getPlanInfo(plan)             Get plan details as map
│   │       ├── getSlotInfo(plan, slotIndex)  Get slot information
│   │       ├── getDecodedInputs(plan, slot)  Get decoded slot inputs
│   │       ├── getProducersOf(plan, slot)    Get producer slots
│   │       ├── getDependentsOf(plan, slot)   Get dependent slots
│   │       ├── getSegments(plan)             Get segment information
│   │       ├── getSegmentsWithReplayState(plan, handle)
│   │       ├── getMemoryTimeline(plan)       Get memory timeline
│   │       ├── getDevicePlacement(plan)      Get device placement map
│   │       ├── getParallelGroups(plan)       Get parallel execution groups
│   │       ├── toAsciiArt(plan)              ASCII visualization
│   │       └── Native handle introspection:
│   │           ├── getPlanNumSegments(handle)
│   │           ├── getPlanNumSlots(handle)
│   │           ├── getPlanNumCapturedGraphSegments(handle)
│   │           ├── getPlanTotalGraphReplays(handle)
│   │           ├── getPlanSegmentBackendName(handle, idx)
│   │           ├── getPlanSegmentReplayState(handle, idx)
│   │           ├── getPlanSegmentStatisticsJson(handle, idx)
│   │           └── getPlanSegmentsSummaryJson(handle)
│   │
│   ├── opExecutioner()              Op execution control
│   │   ├── get()                    Get OpExecutioner instance
│   │   ├── getOperationsExecuted()  Count of executed ops
│   │   ├── getTotalExecutionTimeNanos()
│   │   └── stats()                  Execution statistics
│   │
│   └── kernels()                    Kernel management
│       ├── getSelector()            Get KernelSelector
│       ├── getPluginManager()       Get KernelPluginManager
│       └── stats()                  Kernel statistics
│
├── memory()                         [MemorySubsystem]
│   ├── manager()                    Direct MemoryManager access
│   │   ├── get()                    Get MemoryManager instance
│   │   ├── getCurrentWorkspace()    Get current workspace
│   │   ├── setCurrentWorkspace(ws)  Set current workspace
│   │   ├── invokeGc()               Invoke GC
│   │   ├── invokeGcOccasionally()   Conditional GC
│   │   ├── getGcFrequency()         Get GC frequency
│   │   ├── setGcFrequency(freq)     Set GC frequency
│   │   ├── togglePeriodicGc(enabled)
│   │   └── purgeCaches()            Purge memory caches
│   │
│   ├── deallocator()                DeallocatorService access
│   │   ├── get()                    Get DeallocatorService
│   │   ├── getTotalAllocations()    Total allocation count
│   │   ├── getTotalDeallocations()  Total deallocation count
│   │   ├── getLiveReferenceCount()  Live reference count
│   │   ├── isShutdownInProgress()   Check shutdown state
│   │   ├── blockDeallocator(block)  Block deallocator
│   │   └── getThreadCount()        Deallocator thread count
│   │
│   ├── tracker()                    MemoryTracker access
│   │   ├── get()                    Get MemoryTracker
│   │   ├── getAllocatedAmount(deviceId)
│   │   ├── getCachedAmount(deviceId)
│   │   ├── getWorkspaceAllocatedAmount(deviceId)
│   │   ├── getTotalMemory(deviceId)
│   │   ├── getFreeMemory(deviceId)
│   │   ├── getActiveMemory(deviceId)
│   │   ├── getManagedMemory(deviceId)
│   │   ├── getApproximateFreeMemory(deviceId)
│   │   ├── getPreciseFreeMemory(deviceId)
│   │   ├── getAllocatedHostAmount()
│   │   ├── getCachedHostAmount()
│   │   └── getMemoryPerDevice()     Formatted summary
│   │
│   └── samples()                    Historical sampling
│       ├── start()                  Start background sampling
│       ├── start(intervalMs)        Start with custom interval
│       ├── stop()                   Stop sampling
│       ├── isSamplingActive()       Check if active
│       ├── getRecent(n)             Get recent samples
│       ├── getAll()                 Get all samples
│       ├── clear()                  Clear history
│       ├── getSampleCount()         Get sample count
│       ├── setMaxSamples(max)       Set retention limit
│       └── recordSample()           Manual sample recording
│
├── device()                         [DeviceSubsystem]
│   ├── affinity()                   AffinityManager access
│   │   ├── get()                    Get AffinityManager
│   │   ├── getDeviceForCurrentThread()
│   │   ├── getDeviceForThread(threadId)
│   │   ├── getDeviceForArray(array)
│   │   ├── getNumberOfDevices()
│   │   ├── getAvailableDeviceIds()
│   │   ├── touch(array)             Associate array with device
│   │   ├── replicateToDevice(deviceId, array)
│   │   ├── tagLocation(array, location)
│   │   ├── ensureLocation(array, location)
│   │   ├── getActiveLocation(array)
│   │   ├── getDeviceDescriptor(deviceId)
│   │   ├── getDeviceType(deviceId)
│   │   ├── isCudaDevice(deviceId)
│   │   ├── isCpuDevice(deviceId)
│   │   ├── setDeviceForCurrentThread(deviceId)
│   │   └── getDeviceMemoryInfo(deviceId)
│   │
│   ├── memoryManager()              DeviceMemoryManager access
│   │   ├── get()                    Get DeviceMemoryManager
│   │   ├── switchDevice(id, caller, reason)
│   │   ├── getCurrentDeviceId()
│   │   ├── getCurrentDeviceContext()
│   │   ├── getFreshExecutionStream()
│   │   ├── getDefaultDevice()
│   │   ├── setDefaultDevice(device)
│   │   ├── getFallbackDevice()
│   │   ├── setMemoryCap(device, bytes)
│   │   ├── getMemoryCap(device)
│   │   ├── setDevicePriority(device, priority)
│   │   ├── getDevicePriority(device)
│   │   ├── selectDeviceForAllocation(sizeBytes)
│   │   ├── getAllocatedMemory(device)
│   │   ├── getPeakMemory(device)
│   │   ├── getActualFreeMemory(device)
│   │   ├── getAvailableMemory(device)
│   │   ├── hasMemoryPressure(device)
│   │   ├── setAutoFallbackEnabled(enabled)
│   │   ├── setMemoryPressureThreshold(threshold)
│   │   ├── enableMemorySimulation(enabled)
│   │   ├── setSimulatedFreeMemory(deviceId, bytes)
│   │   ├── registerMemoryPressureCallback(callback)
│   │   ├── getDeviceRoutingPolicy()
│   │   ├── setDeviceRoutingPolicy(policy)
│   │   └── getMemorySummary()
│   │
│   └── info()                       Device information
│       ├── count()                  Device count
│       ├── getType(deviceId)        Device type name
│       ├── getDescriptor(deviceId)  Device descriptor
│       ├── getSummary(deviceId)     Formatted summary
│       ├── getAllSummaries()        All device summaries
│       ├── isMultiDevice()          Multi-device check
│       └── hasGpu()                 GPU availability check
│
├── workspaces()                     [WorkspaceSubsystem]
│   ├── manager()                    Get WorkspaceManager
│   ├── currentWorkspace()           Get current workspace
│   ├── stats()                      Workspace statistics
│   ├── spillStats()                 Spill behavior stats
│   ├── destroyCurrentThreadWorkspaces()
│   ├── enableSampling()             Enable workspace sampling
│   └── disableSampling()            Disable workspace sampling
│
├── profiling()                      [ProfilingSubsystem]
│   ├── enableOpTiming(detailed)     Enable op timing
│   ├── disableOpTiming()            Disable op timing
│   ├── isEnabled()                  Check if enabled
│   ├── stats()                      Profiling statistics
│   ├── getBandwidth()               Memory bandwidth stats
│   └── clear()                      Clear profiling data
│
├── lifecycle()                      [LifecycleSubsystem]
│   ├── enableTracking()             Enable lifecycle tracking
│   ├── disableTracking()            Disable tracking
│   ├── enableStackTraceCapture()    Enable stack traces
│   ├── totalCreated()               Total arrays created
│   ├── totalDestroyed()             Total arrays destroyed
│   ├── liveCount()                  Current live arrays
│   ├── getHistory(limit)            Recent lifecycle events
│   ├── stats()                      Lifecycle statistics
│   └── clearHistory()               Clear event history
│
└── diagnostics()                    [DiagnosticSubsystem]
    ├── runLeakDetection()           Comprehensive leak analysis
    ├── getActiveIssues()            Active diagnostic issues
    └── health()                     Framework health status
```

---

## Files Created

### Core Framework
- `org.nd4j.linalg.framework.Framework` - Main unified accessor
- `org.nd4j.linalg.framework.DiagnosticSubsystem` - Diagnostics and health
- `org.nd4j.lifecycle.LifecycleSubsystem` - Array lifecycle tracking

### Execution Subsystem (`org.nd4j.linalg.framework.exec`)
- `ExecutionSubsystem` - Execution, DSP, kernels
- `DspStats` - DSP statistics
- `ExecutionStats` - Execution statistics  
- `KernelStats` - Kernel statistics

### Memory Subsystem (`org.nd4j.linalg.framework.memory`)
- `MemorySubsystem` - Memory management, deallocator, tracker, sampling

### Device Subsystem (`org.nd4j.linalg.framework.device`)
- `DeviceSubsystem` - AffinityManager and device info access

### Workspace Subsystem (`org.nd4j.linalg.framework.workspace`)
- `WorkspaceSubsystem` - Workspace management

### Profiling Subsystem (`org.nd4j.linalg.framework.profiling`)
- `ProfilingSubsystem` - Op timing and performance

### History/Tracking (`org.nd4j.linalg.framework.history`)
- `MemorySample` - Memory snapshot
- `WorkspaceSample` - Workspace snapshot
- `ArrayLifecycleEvent` - Lifecycle event record
- `LifecycleStats` - Lifecycle statistics
- `MemorySampler` - Background memory sampling
- `ArrayLifecycleTracker` - Lifecycle event tracking
- `WorkspaceSampler` - Workspace sampling
- `SpillStats` - Spill statistics

### Leak Detection (`org.nd4j.linalg.framework.leak`)
- `PotentialLeak` - Leak candidate
- `LeakReport` - Leak detection report
- `LeakDetector` - Leak detection engine

### Statistics (`org.nd4j.linalg.framework.stats`)
- `MemoryStats` - Memory statistics
- `ProfilingStats` - Profiling statistics
- `WorkspaceStats` - Workspace statistics
- `FrameworkSnapshot` - Complete framework snapshot
- `DiagnosticIssue` - Diagnostic issue record

---

## Usage Examples

### DSP Diagnostics

```java
// Initialize DSP diagnostics from system properties
Nd4j.framework.execution().dsp().initialize();

// Enable all diagnostic categories
Nd4j.framework.execution().dsp().enableAll();

// Or enable specific categories
Nd4j.framework.execution().dsp().enableCategories(
    DspSubsystem.MEMORY | DspSubsystem.EXECUTE | DspSubsystem.COMPILE
);

// Set detail level (0=summary, 1=detailed, 2=full)
Nd4j.framework.execution().dsp().setLevel(DspSubsystem.LEVEL_FULL);

// Record diagnostic events
Nd4j.framework.execution().dsp().record(DspSubsystem.MEMORY, 
    "Custom memory event");

// Get plan report
String report = Nd4j.framework.execution().dsp().getPlanReport();
System.out.println(report);

// Get JSON report
String jsonReport = Nd4j.framework.execution().dsp().getJsonReport();

// DSP Plan Introspection (NEW)
DspPlanIntrospection introspection = Nd4j.framework.execution().dsp().introspection();

// Get plan summary
String summary = introspection.getSummary(plan);
System.out.println(summary);

// Get detailed plan info
Map<String, Object> planInfo = introspection.getPlanInfo(plan);
System.out.println("Slots: " + planInfo.get("numSlots"));
System.out.println("External inputs: " + planInfo.get("numExternalInputs"));

// Get segment information
List<SegmentInfo> segments = introspection.getSegments(plan);
for (SegmentInfo seg : segments) {
    System.out.println("Segment " + seg.getStartSlot() + "-" + seg.getEndSlot() + 
                      ": " + seg.getReplayStateName());
}

// Get memory timeline
List<MemoryEvent> timeline = introspection.getMemoryTimeline(plan);
System.out.println(introspection.getMemoryTimelineSummary(plan));

// Get device placement
Map<Integer, List<Integer>> placement = introspection.getDevicePlacement(plan);
System.out.println(introspection.getDevicePlacementSummary(plan));

// Get parallel execution groups
List<List<Integer>> parallelGroups = introspection.getParallelGroups(plan);
System.out.println(introspection.getParallelGroupsSummary(plan));

// ASCII visualization
System.out.println(introspection.toAsciiArt(plan));

// Native handle introspection (when you have a plan handle)
Pointer handle = getNativePlanHandle();
int numSegments = introspection.getPlanNumSegments(handle);
int capturedGraphs = introspection.getPlanNumCapturedGraphSegments(handle);
int totalReplays = introspection.getPlanTotalGraphReplays(handle);

for (int i = 0; i < numSegments; i++) {
    String backend = introspection.getPlanSegmentBackendName(handle, i);
    int replayState = introspection.getPlanSegmentReplayState(handle, i);
    String stats = introspection.getPlanSegmentStatisticsJson(handle, i);
    System.out.println("Segment " + i + ": backend=" + backend + 
                      ", state=" + replayState + ", stats=" + stats);
}
```

### Memory Management

```java
// Direct MemoryManager access
MemoryManager mm = Nd4j.framework.memory().manager().get();

// Get current workspace
MemoryWorkspace ws = Nd4j.framework.memory().manager().getCurrentWorkspace();

// MemoryTracker access
long freeMemory = Nd4j.framework.memory().tracker().getFreeMemory(0);
long totalMemory = Nd4j.framework.memory().tracker().getTotalMemory(0);
System.out.println(Nd4j.framework.memory().tracker().getMemoryPerDevice());

// DeallocatorService access
long liveArrays = Nd4j.framework.memory().deallocator().getLiveReferenceCount();
long totalAllocated = Nd4j.framework.memory().deallocator().getTotalAllocations();

// Start background memory sampling
Nd4j.framework.memory().samples().start(1000); // 1 second interval

// Get recent memory samples
List<MemorySample> history = Nd4j.framework.memory().samples().getRecent(100);

// Get memory growth trend (bytes/second)
long growthRate = Nd4j.framework.memory().getGrowthTrend(60);
if (growthRate > 0) {
    System.out.println("Memory growing at " + growthRate + " bytes/sec");
}
```

### Device Management

```java
// Get device for current thread
int device = Nd4j.framework.device().affinity().getDeviceForCurrentThread();

// Get number of devices
int numDevices = Nd4j.framework.device().affinity().getNumberOfDevices();

// Get available device IDs
List<Integer> deviceIds = Nd4j.framework.device().affinity().getAvailableDeviceIds();

// Replicate array to specific device
INDArray onDevice1 = Nd4j.framework.device().affinity()
    .replicateToDevice(1, originalArray);

// Ensure array location
Nd4j.framework.device().affinity().ensureLocation(array, AffinityManager.Location.HOST);

// Get device info
System.out.println(Nd4j.framework.device().info().getSummary(0));
System.out.println("Has GPU: " + Nd4j.framework.device().info().hasGpu());

// DeviceMemoryManager - Device switching
DeviceContext ctx = Nd4j.framework.device().memoryManager()
    .switchDevice(1, "MyClass", "processing");

// Get current device ID
int currentDevice = Nd4j.framework.device().memoryManager().getCurrentDeviceId();

// Get fresh execution stream
Pointer stream = Nd4j.framework.device().memoryManager().getFreshExecutionStream();

// Set memory cap for device (8GB)
DeviceDescriptor gpu0 = DeviceDescriptor.cuda(0);
Nd4j.framework.device().memoryManager().setMemoryCap(gpu0, 8L * 1024 * 1024 * 1024);

// Set device priorities (higher = preferred)
Nd4j.framework.device().memoryManager().setDevicePriority(gpu0, 100);
Nd4j.framework.device().memoryManager().setDevicePriority(DeviceDescriptor.cuda(1), 90);
Nd4j.framework.device().memoryManager().setDevicePriority(DeviceDescriptor.cpu(), 10);

// Select best device for allocation
DeviceDescriptor bestDevice = Nd4j.framework.device().memoryManager()
    .selectDeviceForAllocation(1024 * 1024);

// Get memory stats for device
long allocated = Nd4j.framework.device().memoryManager().getAllocatedMemory(gpu0);
long peak = Nd4j.framework.device().memoryManager().getPeakMemory(gpu0);
long available = Nd4j.framework.device().memoryManager().getAvailableMemory(gpu0);

// Check for memory pressure
boolean hasPressure = Nd4j.framework.device().memoryManager().hasMemoryPressure(gpu0);

// Get memory summary for all devices
System.out.println(Nd4j.framework.device().memoryManager().getMemorySummary());

// Enable auto fallback when device is full
Nd4j.framework.device().memoryManager().setAutoFallbackEnabled(true);

// Set memory pressure threshold (90%)
Nd4j.framework.device().memoryManager().setMemoryPressureThreshold(0.9);
```

### Leak Detection

```java
// Run leak detection
LeakReport report = Nd4j.framework.diagnostics().runLeakDetection();

if (report.isLeaksDetected()) {
    System.out.println("Leaks detected: " + report.getSummary());
    System.out.println("Severity: " + report.getSeverity());
    
    for (PotentialLeak leak : report.getPotentialLeaks()) {
        System.out.println("  - Array " + leak.getArrayId() + 
                          ": " + leak.getSizeHumanReadable() +
                          ", age: " + leak.getAgeHumanReadable() +
                          ", confidence: " + (leak.getConfidence() * 100) + "%");
    }
}

// Get framework health
HealthStatus health = Nd4j.framework.diagnostics().health();
System.out.println(health.getSummary());
```

### Complete Framework Snapshot

```java
// Get complete framework state
FrameworkSnapshot snapshot = Nd4j.framework.snapshot();
System.out.println(snapshot.getSummary());

// Print status to log
Nd4j.framework.printStatus();
```

---

## Diagnostic Categories (DSP)

| Category | Constant | Description |
|----------|----------|-------------|
| COMPILE | `DspSubsystem.COMPILE` | Backend compilation events |
| JIT | `DspSubsystem.JIT` | Kernel generation, PTX/cubin |
| EXECUTE | `DspSubsystem.EXECUTE` | Per-step execution flow |
| TIMING | `DspSubsystem.TIMING` | Detailed timing breakdowns |
| MEMORY | `DspSubsystem.MEMORY` | Allocations, OOM, pool state |
| BACKEND | `DspSubsystem.BACKEND` | Backend selection, device placement |
| SHAPE | `DspSubsystem.SHAPE` | Shape analysis, static/dynamic |
| SEGMENT | `DspSubsystem.SEGMENT` | Segment building, boundaries |
| FUSION | `DspSubsystem.FUSION` | Op fusion, identity elimination |
| VERIFY | `DspSubsystem.VERIFY` | Golden comparison, validation |
| KV_CACHE | `DspSubsystem.KV_CACHE` | KV cache config, retention |
| FALLBACK | `DspSubsystem.FALLBACK` | Fallback events, error recovery |

---

## What's Included

| Component | Access Path | Status |
|-----------|-------------|--------|
| **MemoryManager** | `Nd4j.framework.memory().manager().get()` | ✅ Complete |
| **DeallocatorService** | `Nd4j.framework.memory().deallocator().get()` | ✅ Complete |
| **MemoryTracker** | `Nd4j.framework.memory().tracker().get()` | ✅ Complete |
| **AffinityManager** | `Nd4j.framework.device().affinity().get()` | ✅ Complete |
| **DeviceMemoryManager** | `Nd4j.framework.device().memoryManager().get()` | ✅ Complete |
| **OpExecutioner** | `Nd4j.framework.execution().opExecutioner().get()` | ✅ Complete |
| **KernelSelector** | `Nd4j.framework.execution().kernels().getSelector()` | ✅ Complete |
| **DSP Diagnostics** | `Nd4j.framework.execution().dsp()` | ✅ Complete |
| **WorkspaceManager** | `Nd4j.framework.workspaces().manager()` | ✅ Complete |
| **Memory Sampling** | `Nd4j.framework.memory().samples()` | ✅ Complete |
| **Lifecycle Tracking** | `Nd4j.framework.lifecycle()` | ✅ Complete |
| **Leak Detection** | `Nd4j.framework.diagnostics().runLeakDetection()` | ✅ Complete |
| **Constant Caches** | `Nd4j.framework.constants()` | ✅ Complete |
| **OpTimingTracker** | `Nd4j.framework.profiling()` | ⏳ Java wrapper ready, native pending |
| **CudaMemoryPool** | Via native bindings | ⏳ Pending native bindings |

---

## Next Steps (Native Bindings)

The following native JNI bindings need to be added for complete OpTimingTracker and CudaMemoryPool exposure:

### OpTimingTracker Bindings
```java
// In NativeOps.java
public native void enableOpTiming(boolean detailed);
public native void flushOpTiming();
public native String getOpHotspots(int topN);
public native void exportOpTimingTrace(String filename);
public native void exportOpTimingCSV(String filename);
```

### CudaMemoryPool Bindings
```java
// In NativeOps.java  
public native long getCudaPoolUsedBytes(int deviceId);
public native long getCudaPoolReservedBytes(int deviceId);
public native void trimCudaPool(int deviceId);
public native String getCudaPoolStats(int deviceId);
```

---

## Summary

The Framework API provides **complete transparency** into ND4J internals with:

- **Hierarchical structure**: Organized by function (execution, memory, device, etc.)
- **Direct access**: Full access to underlying managers (MemoryManager, AffinityManager, etc.)
- **Historical tracking**: Time-series sampling for memory and workspace usage
- **Lifecycle tracking**: Array creation/destruction with optional stack traces
- **DSP integration**: Full access to DSP diagnostics system
- **Leak detection**: Automated analysis with confidence scoring
- **Health monitoring**: Framework health status with issue detection

All accessible from a single unified entry point: `Nd4j.framework`

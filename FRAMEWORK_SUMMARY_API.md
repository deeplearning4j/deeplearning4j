# ND4J Framework Summary API

## Overview

The `Framework.summary()` method provides a comprehensive, one-call snapshot of the entire ND4J framework state. It returns a `FrameworkSummary` POJO with human-readable `toString()` output covering all subsystems.

## Usage

### Basic Usage

```java
// Get complete framework state summary
FrameworkSummary summary = Nd4j.framework.summary();

// Print human-readable output
System.out.println(summary);

// Get compact one-line summary
System.out.println(summary.getCompactSummary());
```

### Example Output

```
╔══════════════════════════════════════════════════════════╗
║           ND4J FRAMEWORK STATE SNAPSHOT                  ║
╠══════════════════════════════════════════════════════════╣
║ Timestamp: Thu Apr 02 15:30:45 UTC 2026                  ║
╠══════════════════════════════════════════════════════════╣
║ MEMORY:                                                    ║
║   Heap: 512.0/2048.0 MB (25.0%), Off-heap: 128.0 MB,    ║
║   Device: 1024.0 MB, Workspace: 256.0 MB, Arrays: 1500, ║
║   GC: 100/ON, Pressure: 90%                              ║
╠══════════════════════════════════════════════════════════╣
║ DEVICE:                                                    ║
║   Devices: 2 (GPU), Current: 0, Default: 0, Routing:    ║
║   MEMORY_BASED, Fallback: ON, Pressure: 90%, CUDA Pool: ║
║   4096 MB                                                ║
╠══════════════════════════════════════════════════════════╣
║ EXECUTION:                                                 ║
║   DSP: MEMORY,EXECUTE/full, Profiling: ON, Backend:     ║
║   cuda, Ops: 50000 (12.50 s, 250.0 µs/op), Type:        ║
║   CudaExecutioner                                        ║
╠══════════════════════════════════════════════════════════╣
║ WORKSPACE:                                                 ║
║   Workspaces: 1, Memory: 256.0 MB, Spilled: 12.0 MB    ║
║   (4.7%), Initial: 1024.0 MB, Learning: ON, Debug:      ║
║   DISABLED                                               ║
╠══════════════════════════════════════════════════════════╣
║ PROFILING:                                                 ║
║   Enabled: YES, Frequency: 1000, Ops: 50000 (12.50 s,  ║
║   250.0 µs/op), Helper: 45.0%                            ║
╠══════════════════════════════════════════════════════════╣
║ LIFECYCLE:                                                 ║
║   Created: 2000, Destroyed: 500, Live: 1500, Efficiency:║
║   25.0%, Tracking: ON, StackTrace: OFF, Leak Threshold: ║
║   300 s                                                  ║
╠══════════════════════════════════════════════════════════╣
║ DIAGNOSTICS:                                               ║
║   Enabled: YES, Level: WARNING, Health: WARNING (3     ║
║   issues: 0 crit, 0 err, 3 warn), Monitoring: ON (60 s) ║
╚══════════════════════════════════════════════════════════╝
```

### Compact Summary

```java
// Returns: Framework[Mem=512.0MB, Dev=2, Exec=MEMORY,EXECUTE/full, WS=1, Life=1500, Diag=WARNING]
String compact = summary.getCompactSummary();
```

## State POJOs

Each subsystem state is a POJO with getters and a human-readable `toString()`:

### MemoryState

```java
MemoryState mem = summary.getMemory();

// Getters
long heapUsed = mem.getHeapUsedBytes();
double heapMb = mem.getHeapUsedMb();
double heapPercent = mem.getHeapUtilizationPercent();
double offHeapMb = mem.getOffHeapMb();
double deviceMb = mem.getDeviceMb();
double workspaceMb = mem.getWorkspaceMb();
double totalMb = mem.getTotalUsedMb();
long liveArrays = mem.getLiveArrayCount();
int gcFreq = mem.getGcFrequency();
boolean periodicGc = mem.isPeriodicGcEnabled();
boolean noArrayGc = mem.isNoArrayGc();
double pressureThreshold = mem.getMemoryPressureThreshold();

// toString() output:
// "Heap: 512.0/2048.0 MB (25.0%), Off-heap: 128.0 MB, Device: 1024.0 MB, 
//  Workspace: 256.0 MB, Arrays: 1500, GC: 100/ON, Pressure: 90%"
```

### DeviceState

```java
DeviceState dev = summary.getDevice();

// Getters
int numDevices = dev.getNumDevices();
int currentDevice = dev.getCurrentDeviceId();
int defaultDevice = dev.getDefaultDeviceId();
String routing = dev.getRoutingPolicy();
boolean autoFallback = dev.isAutoFallbackEnabled();
double pressureThreshold = dev.getMemoryPressureThreshold();
int cudaPoolMb = dev.getCudaMemoryPoolSizeMb();
boolean hasGpu = dev.isHasGpu();

// toString() output:
// "Devices: 2 (GPU), Current: 0, Default: 0, Routing: MEMORY_BASED, 
//  Fallback: ON, Pressure: 90%, CUDA Pool: 4096 MB"
```

### ExecutionState

```java
ExecutionState exec = summary.getExecution();

// Getters
boolean dspEnabled = exec.isDspEnabled();
String dspDiag = exec.getDspDiagnostics();
String dspLevel = exec.getDspDiagnosticsLevel();
String dspStatus = exec.getDspStatus();  // Combined status string
boolean profilingEnabled = exec.isOpProfilingEnabled();
String backend = exec.getPreferredBackend();
long opsExecuted = exec.getOperationsExecuted();
double execSeconds = exec.getTotalExecutionTimeSeconds();
double avgMicros = exec.getAvgExecutionTimeMicros();
String execType = exec.getExecutionerType();

// toString() output:
// "DSP: MEMORY,EXECUTE/full, Profiling: ON, Backend: cuda, 
//  Ops: 50000 (12.50 s, 250.0 µs/op), Type: CudaExecutioner"
```

### WorkspaceState

```java
WorkspaceState ws = summary.getWorkspace();

// Getters
int numWorkspaces = ws.getNumWorkspaces();
double totalMb = ws.getTotalWorkspaceMb();
double spilledMb = ws.getTotalSpilledBytes() / (1024.0 * 1024.0);
double spillPercent = ws.getSpillPercent();
double initialMb = ws.getInitialSizeMb();
boolean learning = ws.isLearningEnabled();
String debugMode = ws.getDebugMode();

// toString() output:
// "Workspaces: 1, Memory: 256.0 MB, Spilled: 12.0 MB (4.7%), 
//  Initial: 1024.0 MB, Learning: ON, Debug: DISABLED"
```

### ProfilingState

```java
ProfilingState prof = summary.getProfiling();

// Getters
boolean enabled = prof.isEnabled();
int frequency = prof.getFrequency();
long totalOps = prof.getTotalOps();
double totalSeconds = prof.getTotalTimeSeconds();
double avgMicros = prof.getAvgTimeMicros();
double helperPercent = prof.getHelperUsagePercent();

// toString() output:
// "Enabled: YES, Frequency: 1000, Ops: 50000 (12.50 s, 250.0 µs/op), Helper: 45.0%"
```

### LifecycleState

```java
LifecycleState life = summary.getLifecycle();

// Getters
long totalCreated = life.getTotalCreated();
long totalDestroyed = life.getTotalDestroyed();
long liveArrays = life.getLiveArrays();
double efficiency = life.getDestructionEfficiency();
long netCreation = life.getNetCreation();
boolean tracking = life.isTrackingEnabled();
boolean stackTrace = life.isStackTraceCapture();
double leakThresholdSec = life.getLeakDetectionAgeThresholdSeconds();

// toString() output:
// "Created: 2000, Destroyed: 500, Live: 1500, Efficiency: 25.0%, 
//  Tracking: ON, StackTrace: OFF, Leak Threshold: 300 s"
```

### DiagnosticState

```java
DiagnosticState diag = summary.getDiagnostics();

// Getters
boolean enabled = diag.isEnabled();
String level = diag.getLevel();
int issueCount = diag.getIssueCount();
int critical = diag.getCriticalIssues();
int errors = diag.getErrorIssues();
int warnings = diag.getWarningIssues();
boolean healthMonitoring = diag.isHealthMonitoringEnabled();
double checkIntervalSec = diag.getHealthCheckIntervalSeconds();
boolean healthy = diag.isHealthy();
String healthStatus = diag.getHealthStatus();  // HEALTHY, WARNING, ERROR, CRITICAL

// toString() output:
// "Enabled: YES, Level: WARNING, Health: WARNING (3 issues: 0 crit, 0 err, 3 warn), 
//  Monitoring: ON (60 s)"
```

## Use Cases

### 1. Debugging Memory Issues

```java
FrameworkSummary summary = Nd4j.framework.summary();
MemoryState mem = summary.getMemory();

if (mem.getHeapUtilizationPercent() > 80) {
    log.warn("High heap utilization: {}%", mem.getHeapUtilizationPercent());
}

if (mem.getOffHeapMb() > 1024) {
    log.warn("High off-heap memory: {} MB", mem.getOffHeapMb());
}

// Check for potential leaks
LifecycleState life = summary.getLifecycle();
if (life.getNetCreation() > 1000 && life.getDestructionEfficiency() < 0.5) {
    log.warn("Potential memory leak: {} live arrays, {}% destruction efficiency",
             life.getLiveArrays(), life.getDestructionEfficiency() * 100);
}
```

### 2. Performance Monitoring

```java
FrameworkSummary summary = Nd4j.framework.summary();
ExecutionState exec = summary.getExecution();

log.info("Execution: {} ops, {} s total, {} µs/op",
         exec.getOperationsExecuted(),
         exec.getTotalExecutionTimeSeconds(),
         exec.getAvgExecutionTimeMicros());

if (exec.getAvgExecutionTimeMicros() > 1000) {
    log.warn("Slow execution: {} µs/op", exec.getAvgExecutionTimeMicros());
}

// Check GPU utilization
DeviceState dev = summary.getDevice();
if (dev.isHasGpu() && dev.getNumDevices() > 0) {
    log.info("GPU available: {} devices, CUDA pool: {} MB",
             dev.getNumDevices(), dev.getCudaMemoryPoolSizeMb());
}
```

### 3. Health Check Endpoint

```java
@GetMapping("/framework/health")
public ResponseEntity<String> healthCheck() {
    FrameworkSummary summary = Nd4j.framework.summary();
    DiagnosticState diag = summary.getDiagnostics();
    
    if (!diag.isHealthy()) {
        return ResponseEntity.status(503)
            .body("Framework unhealthy: " + diag.getHealthStatus());
    }
    
    return ResponseEntity.ok(summary.getCompactSummary());
}
```

### 4. Periodic Monitoring

```java
ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

scheduler.scheduleAtFixedRate(() -> {
    FrameworkSummary summary = Nd4j.framework.summary();
    
    // Log compact summary
    log.info("Framework status: {}", summary.getCompactSummary());
    
    // Alert on issues
    DiagnosticState diag = summary.getDiagnostics();
    if (diag.getCriticalIssues() > 0) {
        alertService.sendAlert("Critical framework issues: " + diag.getCriticalIssues());
    }
    
    // Check memory pressure
    MemoryState mem = summary.getMemory();
    if (mem.getHeapUtilizationPercent() > 90) {
        alertService.sendAlert("High memory pressure: " + mem.getHeapUtilizationPercent() + "%");
    }
    
}, 0, 60, TimeUnit.SECONDS);
```

### 5. Pre/Post Operation Comparison

```java
// Before operation
FrameworkSummary before = Nd4j.framework.summary();

// Run operation
runMyOperation();

// After operation
FrameworkSummary after = Nd4j.framework.summary();

// Compare
long arraysCreated = after.getLifecycle().getLiveArrays() - before.getLifecycle().getLiveArrays();
double memoryDelta = after.getMemory().getHeapUsedMb() - before.getMemory().getHeapUsedMb();

log.info("Operation impact: {} arrays, {} MB heap", arraysCreated, memoryDelta);
```

## Programmatic Access

All state POJOs provide typed getters for programmatic access:

```java
FrameworkSummary summary = Nd4j.framework.summary();

// Access individual metrics
double heapPercent = summary.getMemory().getHeapUtilizationPercent();
int numDevices = summary.getDevice().getNumDevices();
long opsExecuted = summary.getExecution().getOperationsExecuted();
int liveArrays = summary.getLifecycle().getLiveArrays();
int issueCount = summary.getDiagnostics().getIssueCount();

// Use in monitoring/alerting
if (heapPercent > 85 || liveArrays > 10000 || issueCount > 0) {
    // Trigger alert
}
```

## Serialization

All state POJOs are simple Java beans that can be easily serialized:

```java
// JSON serialization (with Jackson)
ObjectMapper mapper = new ObjectMapper();
String json = mapper.writeValueAsString(summary);

// Or individual states
String memoryJson = mapper.writeValueAsString(summary.getMemory());
```

## Performance

The `summary()` method is designed for low overhead:
- All state capture is non-blocking
- Exception handling prevents failures from affecting the application
- Suitable for periodic monitoring (every 30-60 seconds recommended)

For high-frequency monitoring, consider using individual subsystem accessors instead.

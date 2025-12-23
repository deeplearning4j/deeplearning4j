# AddressSanitizer Memory Leak Detection

## Status

Implemented

Proposed by: Adam Gibson (13-10-2025)
Discussed with: N/A

## Context

Memory leak detection is critical for maintaining the stability and performance of the deeplearning4j native libraries. Traditional tools like Valgrind are slow and can make debugging impractical. AddressSanitizer (ASAN) provides fast memory leak detection with minimal overhead, but requires careful configuration to work correctly with the JVM's complex shutdown sequence.

We encountered specific challenges:
1. JVM shutdown deadlocks when ASAN performs leak checks at exit
2. Complex interaction between Java destructors and native C++ cleanup
3. ThreadPool singleton destruction timing issues during exit
4. Need for leak reports without blocking process termination

## Decision

We implement AddressSanitizer-based memory leak detection with the following configuration and tooling:

### ASAN Configuration

Use the following environment variable settings:

```bash
LD_PRELOAD=/usr/lib64/libasan.so.8
ASAN_OPTIONS="alloc_dealloc_mismatch=0:detect_leaks=1:new_delete_type_mismatch=0:halt_on_error=0:exitcode=0:report_objects=1:use_stacks=1:use_registers=1:leak_check_at_exit=1:fast_unwind_on_malloc=0:log_path=/tmp/asan.log"
```

### Key Configuration Parameters

1. **alloc_dealloc_mismatch=0**: Ignore C/C++ allocation mismatches (necessary for JNI)
2. **detect_leaks=1**: Enable leak detection
3. **new_delete_type_mismatch=0**: Ignore new/delete type mismatches
4. **halt_on_error=0**: Continue execution after errors
5. **exitcode=0**: Don't change exit code on errors
6. **report_objects=1**: Include object addresses in reports
7. **use_stacks=1**: Use stack traces
8. **use_registers=1**: Use register information
9. **leak_check_at_exit=1**: Perform leak check before exit
10. **fast_unwind_on_malloc=0**: Use slow unwinding for better stack traces
11. **log_path=/tmp/asan.log**: Write reports to files (avoids stderr issues)

### ThreadPool Shutdown Fix

Modified the ThreadPool and CallableInterface classes to properly signal worker threads before destroying condition variables:

**CallableInterface.h additions:**
```cpp
std::atomic<bool> _shutdown;
void shutdown();
bool isShutdown();
```

**ThreadPool.cpp destructor:**
```cpp
ThreadPool::~ThreadPool() {
  // Signal all threads to shutdown BEFORE destroying anything
  for (size_t e = 0; e < _interfaces.size(); e++) {
    _interfaces[e]->shutdown();
  }

  // Wait for all threads to finish
  for (size_t e = 0; e < _threads.size(); e++) {
    if (_threads[e].joinable()) {
      _threads[e].join();
    }
  }

  // Now it's safe to delete resources
  for (size_t e = 0; e < _queues.size(); e++) {
    delete _queues[e];
    delete _interfaces[e];
  }

  while (!_tickets.empty()) {
    auto t = _tickets.front();
    _tickets.pop();
    delete t;
  }
}
```

### Tooling Integration

Integrated ASAN into the platform-tests test runner to allow easy switching between different memory debugging tools.

## Implementation Details

### Maven Build Commands

Build with AddressSanitizer enabled (no special flags needed - just use LD_PRELOAD at runtime):

```bash
mvn clean install -DskipTests
```

### Test Execution

Run tests with ASAN:

```bash
export TEST_RUNNER_PREFIX="asan"
mvn test
```

Or for specific tests:

```bash
TEST_RUNNER_PREFIX="asan" mvn test -Dtest=YourTestClass
```

### Leak Report Location

Leak reports are written to:
```
/tmp/asan.log.<pid>
```

### Handling Exit Deadlock

If the process still hangs during shutdown:

1. Leak reports are written **before** the deadlock occurs
2. Kill the hung process: `kill -9 <pid>`
3. Leak information is already captured in log files

### Workflow

1. Run application/tests with ASAN enabled
2. Leak detection happens automatically at exit
3. Reports written to `/tmp/asan.log.*` files
4. If process hangs after leak report, kill it manually
5. Review leak reports for memory issues

## Consequences

### Advantages

1. **Fast**: 2-3x slowdown vs Valgrind's 20-50x slowdown
2. **Accurate**: Precise leak detection with stack traces
3. **Integrated**: Works seamlessly with JVM and native code
4. **Automated**: No code instrumentation required
5. **Reliable**: Leak reports generated before any deadlock
6. **File-based**: Logs persist even if process is killed

### Drawbacks

1. **Deadlock at exit**: Process may hang after leak report (workaround: kill process)
2. **Library dependency**: Requires libasan.so.8 installation
3. **Build requirement**: Must be compiled with GCC/Clang supporting ASAN
4. **Memory overhead**: ~2x memory usage during execution
5. **Linux-specific**: LD_PRELOAD mechanism is Linux-specific

### Comparison with Valgrind

| Feature | ASAN | Valgrind |
|---------|------|----------|
| Speed | 2-3x slowdown | 20-50x slowdown |
| Memory overhead | 2x | 10x+ |
| Stack traces | Excellent | Good |
| Leak detection | Yes | Yes |
| Use-after-free | Yes | Yes |
| Setup complexity | Low | Medium |
| Exit handling | May deadlock | Clean exit |

## Usage Examples

### Basic Usage

```bash
# Set environment
export LD_PRELOAD=/usr/lib64/libasan.so.8
export ASAN_OPTIONS="alloc_dealloc_mismatch=0:detect_leaks=1:new_delete_type_mismatch=0:halt_on_error=0:exitcode=0:report_objects=1:use_stacks=1:use_registers=1:leak_check_at_exit=1:fast_unwind_on_malloc=0:log_path=/tmp/asan.log"

# Run your application
java -cp your-app.jar YourMainClass

# Check for leaks
cat /tmp/asan.log.*
```

### Maven Tests

```bash
# Via test runner prefix
TEST_RUNNER_PREFIX="asan" mvn test

# Via environment
export LD_PRELOAD=/usr/lib64/libasan.so.8
export ASAN_OPTIONS="..."
mvn test
```

### Interpreting Results

Leak reports show:
- Memory allocation stack traces
- Size and count of leaked allocations
- Direct vs indirect leaks
- Possible leaks (when pointers still exist but are unreachable)

Example output:
```
Direct leak of 1024 bytes in 1 object(s) allocated from:
    #0 0x7f... in malloc
    #1 0x7f... in operator new(unsigned long)
    #2 0x7f... in sd::NativeOps::createContext()
    #3 0x7f... in Java_org_nd4j_nativeblas_Nd4jCpu_createContext
```

## Installation Requirements

### RedHat/Fedora
```bash
sudo dnf install libasan
```

### Ubuntu/Debian
```bash
sudo apt-get install libasan6
```

### Verification
```bash
ls -la /usr/lib*/libasan.so*
```

## References

- AddressSanitizer documentation: https://github.com/google/sanitizers/wiki/AddressSanitizer
- LeakSanitizer documentation: https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer
- ThreadPool.cpp implementation
- CallableInterface.h/cpp implementation
- platform-tests/bin/java wrapper script

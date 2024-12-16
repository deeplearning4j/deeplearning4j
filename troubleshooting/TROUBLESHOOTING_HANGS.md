# Troubleshooting Hanging Java Processes with Native Code

This guide covers various approaches to debugging Java processes that hang due to native code issues in libnd4j.

## Understanding Process Hangs

When a native crash occurs in JNI code, the Java process often appears to "hang" instead of crashing outright. This happens because:
1. The JVM continues running even though the native code has encountered a fatal error
2. The native crash may only affect one thread while others continue running
3. The usual Java exception handling mechanisms don't catch native crashes

## Debug Tools and Approaches

### 1. GDB Debugging

#### Using ptrace
```bash
# Attach to a running process
sudo gdb -p <process-id>

# Once in GDB
(gdb) thread apply all bt
```

#### Direct Process Attachment
```bash
gdb -p <process-id>
```

Key GDB commands:
- `thread apply all bt` - Get backtraces from all threads
- `info threads` - List all threads
- `thread <number>` - Switch to a specific thread
- `bt` - Show backtrace of current thread

### 2. Valgrind Integration

#### Running Tests with Valgrind
```bash
mvn test -Dtest.prefix="valgrind --tool=memcheck"
```

The platform-tests/bin/java script provides special handling for Valgrind:
- Automatically generates suppression files for JVM-related false positives
- Adds important Valgrind flags:
  - `--track-origins=yes`: Track the origins of uninitialized values
  - `--keep-stacktraces=alloc-and-free`: Maintain allocation/free stacktraces
  - `--error-limit=no`: Show all errors
- Disables JIT compilation with `-Djava.compiler=NONE`

### 3. Address Sanitizer (ASAN)

#### Building with ASAN Support
```bash
mvn clean install -Dlibnd4j.sanitize=ON -Dlibnd4j.sanitizers="address,undefined,float-divide-by-zero,float-cast-overflow"
```

Key ASAN Features (from CMakeLists.txt):
- Adds compilation flags:
  - `-fsanitize=address`
  - `-static-libasan`
  - `-ftls-model=local-dynamic`
- Requires preloading the ASAN library:
  ```bash
  export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/*/libasan.so
  ```

Important Notes:
- Cannot use thread and address sanitizer simultaneously
- Address and undefined sanitizers must be used carefully together

### 4. CUDA Compute Sanitizer

For CUDA-specific issues:

```bash
compute-sanitizer --tool memcheck ./your-program
```

Or attach to running process:
```bash
compute-sanitizer --tool memcheck --attach-pid <process-id>
```

Features:
- Memory access checking
- Race condition detection
- Leak detection
- Initialization checking

## Build Configurations

### CPU Builds with Sanitizers
```bash
# Basic sanitizer build
mvn clean install -Dlibnd4j.sanitize=ON

# With specific sanitizers
mvn clean install -Dlibnd4j.sanitize=ON -Dlibnd4j.sanitizers="address,undefined,float-divide-by-zero,float-cast-overflow"
```

### CUDA Builds with Debugging
```bash
# Enable CUDA debugging symbols
mvn clean install -Dlibnd4j.chip=cuda -Dlibnd4j.cuda=cudnn -Dlibnd4j.build=debug
```

## Best Practices

1. **Systematic Approach**:
   - Start with ASAN/Valgrind for memory issues
   - Use GDB for immediate investigation of hangs
   - Use Compute Sanitizer for CUDA-specific problems

2. **Log Collection**:
   - Collect all thread dumps
   - Save sanitizer outputs
   - Keep core dumps if generated

3. **Build Considerations**:
   - Debug builds contain more information but run slower
   - Sanitizer builds have significant overhead
   - Consider using both debug symbols and sanitizers for thorough investigation

## Common Issues and Solutions

1. **Memory Access Violations**:
   - Use ASAN or Valgrind to detect
   - Check array bounds and pointer arithmetic
   - Look for use-after-free scenarios

2. **CUDA Synchronization Issues**:
   - Use Compute Sanitizer's race detection
   - Check for proper stream synchronization
   - Verify kernel launch parameters

3. **Resource Leaks**:
   - Use Valgrind's memcheck
   - Check CUDA memory management
   - Verify native memory deallocations
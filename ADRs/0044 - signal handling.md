# ADR 0034: Signal Handling Implementation in ND4J

## Status

Implemented

Proposed by: Adam Gibson (09-01-2025)
Discussed with: Paul Dubs

## Context
The ND4J library requires robust signal handling to provide meaningful debugging information when crashes or unexpected behavior occur during neural network operations. Signal handling is particularly important for identifying the root causes of crashes in native code, where stack traces and error messages are not always readily available.

Previously, the signal handling system in ND4J was minimal and underdeveloped:
- Signal handlers were essentially no-ops by default
- Limited documentation and exposure to users
- Only accessible through low-level release flags
- No standardized way to enable or configure signal handling
- No stack trace or detailed crash information available

The primary challenges include:

1. Identifying the source of crashes in native C++ code
2. Providing meaningful debugging information for SIGSEGV, SIGINT, and other signals
3. Ensuring thread safety during signal handling
4. Maintaining compatibility across different operating systems (Linux, macOS, Windows)
5. Minimizing performance impact when signal handling is not needed
6. Providing stack traces for debugging purposes
7. Making signal handling more accessible and configurable

## Decision
We implement a comprehensive signal handling system in the OpRegistrator class to manage and process system signals. The system is designed with the following characteristics:

### Key Components
- Signal handler functions for common signals (SIGSEGV, SIGINT, SIGABRT, etc.)
- Environment-controlled signal handling activation
- Platform-specific implementations for Linux/macOS and Windows
- Stack trace capture and printing functionality using `backward.hpp`
- Thread-safe signal handling mechanism
- Conditional compilation using `SD_GCC_FUNCTRACE` for stack trace functionality
- User-friendly configuration through Environment interface

### Implementation Details
1. Signal handling is controlled through the Environment interface
2. Handlers are registered for common signals:
    - SIGSEGV (Segmentation fault)
    - SIGINT (Interrupt)
    - SIGABRT (Abort)
    - SIGFPE (Floating-point exception)
    - SIGILL (Illegal instruction)
    - SIGTERM (Termination)
3. Stack trace capture is implemented using the `backward.hpp` library (enabled via `SD_GCC_FUNCTRACE`)
4. Platform-specific implementations:
    - Linux/macOS: Uses sigaction
    - Windows: Uses Vectored Exception Handling
5. Environment control:
    - Signal handling can be enabled/disabled via Environment::setSignalHandling()
    - Default state is disabled to minimize performance impact
    - Simple API for users to enable/configure signal handling

### Signal Handling Flow
1. Signal occurs (e.g., segmentation fault)
2. Signal handler is invoked
3. Check if signal handling is enabled in Environment
4. If `SD_GCC_FUNCTRACE` is enabled:
    - Capture stack trace using `backward.hpp`
    - Print detailed stack trace information
    - Print faulting memory address (if available)
5. Print signal information
6. Continue or terminate based on signal type

### SD_GCC_FUNCTRACE and Stack Trace Resolution
The `SD_GCC_FUNCTRACE` flag enables the use of the `backward.hpp` library for stack trace resolution. This provides:
- Detailed stack traces showing where crashes occur
- Function names, file names, and line numbers
- Memory address information for segmentation faults
- Thread-safe stack trace capture

Example stack trace output:
```
BEGIN SIGINT STACK TRACE
#0 0x00007f8e1b2c5f45 in __GI_raise
#1 0x00007f8e1b2a7835 in __GI_abort
#2 0x000055d6c4b2a1d9 in sigIntHandler
#3 0x00007f8e1b2c5f45 in __GI_raise
#4 0x00007f8e1b2a7835 in __GI_abort
END SIGINT STACK TRACE
```

## Thread Safety Implementation

The signal handling system is inherently thread-safe due to:
1. Atomic signal handler registration
2. Read-only access to Environment settings
3. Thread-local stack trace capture
4. Platform-specific thread-safe signal handling mechanisms
5. `backward.hpp`'s thread-safe stack trace capture

## Consequences

### Advantages
1. Improved Debugging:
    - Provides detailed stack traces for crashes
    - Identifies faulting memory addresses
    - Helps diagnose native code issues
    - Shows exact location of crashes with `SD_GCC_FUNCTRACE`

2. Flexibility:
    - Signal handling can be enabled/disabled at runtime
    - Minimal overhead when disabled
    - Works across multiple platforms
    - Stack trace resolution can be enabled/disabled via `SD_GCC_FUNCTRACE`
    - Easy to configure through Environment interface

3. Safety:
    - Proper cleanup of resources
    - Controlled signal handling activation
    - Prevents signal handler conflicts

4. Compatibility:
    - Works on Linux, macOS, and Windows
    - Supports both GCC and MSVC compilers
    - `backward.hpp` works across multiple platforms

5. Accessibility:
    - No longer requires low-level build flags
    - Simple API for enabling signal handling
    - Better documentation and exposure

### Disadvantages
1. Performance Impact:
    - Stack trace capture can be slow
    - Signal handling adds some overhead
    - Memory usage for stack trace storage
    - `SD_GCC_FUNCTRACE` adds additional overhead

2. Implementation Complexity:
    - Platform-specific code required
    - Signal handler safety considerations
    - Thread safety requirements
    - Integration with `backward.hpp`

3. Limitations:
    - Stack trace quality depends on compiler
    - Limited information in release builds
    - Windows support less comprehensive than Unix
    - `SD_GCC_FUNCTRACE` requires GCC-compatible compiler

## Technical Details

### Signal Handlers
```cpp
void sigIntHandler(int sig, siginfo_t* info, void* ucontext);
void sigSegVHandler(int sig, siginfo_t* info, void* ucontext);
```

### Handler Registration
```cpp
void registerSignalHandlers() {
  if (!Environment::getInstance().isSignalHandling()) return;
  
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = [](int sig, siginfo_t* info, void* ucontext) {
    if (sig == SIGSEGV) {
      sigSegVHandler(sig, info, ucontext);
    } else {
      sigIntHandler(sig, info, ucontext);
    }
  };
  
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  // ... other signals
}
```

### Environment Control
```java
public interface Environment {
  boolean isSignalHandling();
  void setSignalHandling(boolean reallyEnable);
}
```

## Alternatives Considered

1. No Signal Handling:
    - Pros: No performance impact
    - Cons: Difficult to debug native crashes

2. Simple Signal Handling:
    - Pros: Easier implementation
    - Cons: Less debugging information

3. External Debugging Tools:
    - Pros: More comprehensive debugging
    - Cons: Harder to integrate, requires external setup

4. Custom Crash Reporter:
    - Pros: More control over reporting
    - Cons: Higher implementation complexity

5. Signal Handling Libraries:
    - Pros: Pre-built solution
    - Cons: Additional dependencies, less control

The chosen solution provides a balance between debugging capability, performance impact, and implementation complexity, while maintaining compatibility across different platforms and build configurations. The use of `SD_GCC_FUNCTRACE` with `backward.hpp` provides particularly valuable debugging information when enabled, at the cost of additional overhead. The new implementation significantly improves upon the previous minimal signal handling system by making it more accessible, configurable, and useful for debugging.
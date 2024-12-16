# Debugging Build Flags for libnd4j

This guide covers important build flags used for debugging crashes and performance issues in libnd4j.

## Core Debug Flags

### 1. libnd4j.calltrace
- **Purpose**: Enables function call tracing for debugging segmentation faults and crashes
- **Implementation**: Maps to `SD_GCC_FUNC_TRACE` cmake flag
- **How to use**: 
  ```bash
  mvn -Dlibnd4j.calltrace=ON clean install
  ```
- **Technical details**: 
  - Sets optimization level to 0 (`-O0`) to preserve debugging information
  - Enables `-finstrument-functions` compiler flag
  - Adds debugging symbols with `-g`
  - Sets up function call tracing infrastructure

### 2. libnd4j.printmath
- **Purpose**: Prints individual math operations for debugging numerical issues
- **How to use**:
  ```bash
  mvn -Dlibnd4j.printmath=ON clean install
  ```
- **Technical details**:
  - Adds `SD_PRINT_MATH` compile definition
  - Enables debugging output for operations in platformmath.h and templatemath.h
  - Helps track calculation flow and intermediate results

### 3. libnd4j.printindices
- **Purpose**: Prints indices during array operations to detect out-of-bounds access
- **How to use**:
  ```bash
  mvn -Dlibnd4j.printindices=ON clean install
  ```
- **Technical details**:
  - Adds `PRINT_INDICES` compile definition
  - Helps identify where array index violations occur
  - Useful for tracking down segmentation faults related to memory access

## Usage Scenarios

### Debugging Segmentation Faults
1. Enable calltrace to get function call history:
```bash
mvn -Dlibnd4j.calltrace=ON clean install
```

### Debugging Numerical Issues
1. Enable math operation printing:
```bash
mvn -Dlibnd4j.printmath=ON clean install
```

### Finding Array Access Violations
1. Enable index printing:
```bash
mvn -Dlibnd4j.printindices=ON clean install
```

### Combined Debugging
For comprehensive debugging, you can combine multiple flags:
```bash
mvn -Dlibnd4j.calltrace=ON -Dlibnd4j.printmath=ON -Dlibnd4j.printindices=ON clean install
```

## Additional Notes

- These flags will impact performance significantly and should only be used for debugging
- The calltrace flag is particularly useful with tools like gdb and valgrind
- When using printmath, focus on specific functions by setting PRINT_MATH_FUNCTION_NAME
- For production builds, ensure all debug flags are turned OFF
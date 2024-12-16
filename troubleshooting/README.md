# Deeplearning4j Troubleshooting Documentation

This directory contains comprehensive guides for debugging and troubleshooting libnd4j, particularly focusing on crashes, hangs, and build-related issues.

## Available Documentation

### 1. Build Debugging Flags (README.md)
Covers the core debugging build flags for libnd4j including:
- `libnd4j.calltrace` for function call tracing
- `libnd4j.printmath` for debugging numerical operations
- `libnd4j.printindices` for array bounds checking
- Build configuration examples and usage scenarios

### 2. Process Hang Troubleshooting (TROUBLESHOOTING_HANGS.md)
Comprehensive guide for debugging hanging Java processes, including:
- Using GDB with ptrace and direct process attachment
- Valgrind integration with test suite
- Address Sanitizer (ASAN) configuration and usage
- CUDA Compute Sanitizer for GPU code
- Best practices and common issues

### 3. Preprocessor Debugging (PREPROCESSOR_DEBUGGING.md)
Detailed guide for using the preprocessor debugging capabilities:
- Using `libnd4j.preprocess` flag
- Understanding preprocessor output
- Debugging macro expansions
- Troubleshooting include paths and dependencies
- Tips for handling large codebases

## Quick Start

### For Build Issues:
```bash
# Enable all debugging flags
mvn -Dlibnd4j.calltrace=ON -Dlibnd4j.printmath=ON -Dlibnd4j.printindices=ON clean install
```

### For Process Hangs:
```bash
# Attach to hanging process
sudo gdb -p <process-id>
thread apply all bt
```

### For Macro Issues:
```bash
# Generate preprocessor output
mvn clean install -Dlibnd4j.preprocess=ON
```

## Common Use Cases

1. **Debugging Crashes**
   - Start with build flags from README.md
   - Use generated debug information with GDB

2. **Investigating Hangs**
   - Follow TROUBLESHOOTING_HANGS.md
   - Use appropriate tool based on symptom (GDB, Valgrind, ASAN)

3. **Build/Macro Problems**
   - Use PREPROCESSOR_DEBUGGING.md
   - Examine preprocessor output for issues

## Contributing

When adding new troubleshooting documentation:
1. Create a focused markdown file for the specific topic
2. Update this README.md with a summary
3. Include practical examples and command-line instructions
4. Add cross-references to related documentation
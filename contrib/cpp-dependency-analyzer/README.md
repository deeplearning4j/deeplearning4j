# C++ Dependency Analyzer

A Maven-based tool for analyzing C++ include dependencies within the deeplearning4j/libnd4j project.

## Purpose

This tool analyzes C++ source files to:
- Map include dependencies between files
- Identify module-level dependencies (based on directory structure)
- Detect circular dependencies
- Generate a comprehensive dependency report

## Building

```bash
mvn clean package
```

This creates a shaded JAR with all dependencies included.

## Usage

### Basic Usage
```bash
java -jar target/cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar /path/to/libnd4j
```

### Options
- `-v, --verbose`: Enable verbose output during analysis
- `-o, --output <file>`: Write report to file instead of stdout
- `--include-external`: Include external dependencies in the report
- `-h, --help`: Show help message

### Examples

Analyze libnd4j with verbose output:
```bash
java -jar target/cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar -v ../libnd4j
```

Generate report file:
```bash
java -jar target/cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar -o dependency-report.txt ../libnd4j
```

Include external dependencies:
```bash
java -jar target/cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar --include-external ../libnd4j
```

## How It Works

1. **File Discovery**: Recursively finds all C++ files (.cpp, .cxx, .cc, .c, .hpp, .h, .hxx)
2. **Module Mapping**: Maps files to modules based on their directory structure
3. **Include Resolution**: Parses #include statements and resolves paths:
   - Quoted includes (`#include "file.h"`) - resolved relative to source file, then project root
   - Angle includes (`#include <file.h>`) - resolved relative to project root and common paths
4. **Dependency Analysis**: Builds module-level dependency graph
5. **Report Generation**: Creates comprehensive report with:
   - Module summary (file counts)
   - Module dependencies
   - External dependencies (optional)
   - Circular dependency detection

## Output Format

The tool generates a text report containing:

```
C++ Dependency Analysis Report
==============================

Root Directory: /path/to/project
Analysis Date: Thu Jun 05 10:30:00 JST 2025

Module Summary:
---------------
  blas: 45 files
  cpu: 123 files
  cuda: 87 files
  ...

Module Dependencies:
--------------------
cpu depends on:
  -> blas
  -> include

cuda depends on:
  -> blas
  -> cpu
  -> include

...

Circular Dependencies:
----------------------
No circular dependencies detected.
```

## Include Resolution Strategy

The tool uses a simple but effective strategy for resolving includes:

1. For quoted includes (`"file.h"`):
   - Try relative to the source file's directory
   - Fall back to project root resolution

2. For angle includes (`<file.h>`):
   - Try relative to project root
   - Try common include paths: `include/`, `src/`, `lib/`

3. Unresolved includes are marked as external dependencies

## Limitations

- Simple comment removal (only `//` style)
- No preprocessor macro expansion
- Basic path resolution (doesn't handle complex build system includes)
- Module detection based on directory structure only

## Use Cases

- Understanding project structure and dependencies
- Identifying tightly coupled modules
- Finding circular dependencies that could cause build issues
- Planning refactoring efforts
- Generating documentation about project architecture

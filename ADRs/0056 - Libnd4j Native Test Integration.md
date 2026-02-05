# libnd4j Native Test Integration with Java Platform Tests

## Status
Accepted

Proposed by: Adam Gibson (31-12-2024)

Discussed with: N/A

## Context

The deeplearning4j project consists of two major components:

1. **libnd4j**: A C++ library that provides the core numerical operations, compiled as a native library
2. **Java components**: The Java/Kotlin codebase that wraps and extends libnd4j functionality

Previously, these test suites ran independently:

- libnd4j C++ tests used Google Test (GTest) and were executed via CMake/CTest or shell scripts
- Java tests used JUnit 5 and were executed via Maven Surefire

This separation created several challenges:

### Problems with Separate Test Execution

1. **Fragmented CI/CD pipelines**: Required separate jobs for C++ and Java tests, complicating build configurations
2. **Inconsistent test reporting**: GTest XML output and JUnit XML output were generated separately, making aggregation difficult
3. **Manual coordination**: Developers had to remember to run both test suites
4. **No unified view**: No single command to verify both native and Java functionality
5. **Environment synchronization**: Native test environment variables and Java test properties were managed separately
6. **Debugging complexity**: When issues spanned the JNI boundary, correlating native and Java test failures was difficult

### Requirements

- Run C++ tests alongside Java tests using Maven
- Maintain compatibility with existing GTest infrastructure
- Support test filtering for both test suites
- Generate unified test reports
- Support both CPU and CUDA backends
- Allow selective execution (Java-only, native-only, or both)
- Preserve existing developer workflows

## Decision

We integrate libnd4j C++ test execution into the Java platform-tests module using JUnit 5's DynamicTest feature to discover and execute native tests.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Maven Surefire Plugin                        │
├─────────────────────────────────────────────────────────────────┤
│                        JUnit 5 Platform                          │
├─────────────────────┬───────────────────────────────────────────┤
│   Java Tests        │      Native Test Integration              │
│   (existing)        │                                           │
│                     │  ┌─────────────────────────────────────┐  │
│                     │  │   LibNd4jNativeTestRunner           │  │
│                     │  │   - DynamicTest factory             │  │
│                     │  │   - Test suite discovery            │  │
│                     │  └─────────────┬───────────────────────┘  │
│                     │                │                           │
│                     │  ┌─────────────▼───────────────────────┐  │
│                     │  │   Libnd4jTestHelper                 │  │
│                     │  │   - Executable location             │  │
│                     │  │   - Process execution               │  │
│                     │  │   - Environment setup               │  │
│                     │  └─────────────┬───────────────────────┘  │
│                     │                │                           │
│                     │  ┌─────────────▼───────────────────────┐  │
│                     │  │   GTestResultParser                 │  │
│                     │  │   - XML parsing                     │  │
│                     │  │   - Result aggregation              │  │
│                     │  └─────────────────────────────────────┘  │
└─────────────────────┴───────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    libnd4j/blasbuild/{chip}/                     │
│                    tests_cpu/layers_tests/runtests               │
│                         (GTest executable)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| `LibNd4jNativeTestRunner` | JUnit 5 test class using `@TestFactory` to generate DynamicTests for each native test suite |
| `LibNd4jCategoryTests` | Categorized test runner with nested test classes for different test categories (Operations, Arrays, Convolutions, etc.) |
| `Libnd4jTestHelper` | Utility class for discovering test executable location, running tests, managing timeouts |
| `GTestResultParser` | Parser for Google Test XML output format, converts to Java objects for reporting |
| `NativeBuildHelper` | Optional CMake integration for building native tests if not present |

### Test Discovery Flow

1. JUnit 5 discovers `LibNd4jNativeTestRunner` via standard classpath scanning
2. `@TestFactory` method calls `Libnd4jTestHelper.discoverTestSuites()`
3. Helper executes `runtests --gtest_list_tests` to enumerate available tests
4. Dynamic tests are generated for each discovered suite
5. When executed, each dynamic test spawns the native executable with appropriate filters
6. Results are parsed from GTest XML output and mapped to JUnit assertions

### Configuration

Configuration via system properties (set via Maven or command line):

| Property | Default | Description |
|----------|---------|-------------|
| `org.nd4j.libnd4j.home` | Auto-detected | Path to libnd4j directory |
| `org.nd4j.libnd4j.chip` | `cpu` | Target chip (cpu/cuda) |
| `org.nd4j.libnd4j.test.filter` | `*` | GTest filter pattern |
| `org.nd4j.libnd4j.test.timeout` | `30` | Timeout in minutes |
| `org.nd4j.libnd4j.test.enabled` | `true` | Enable/disable native tests |
| `org.nd4j.libnd4j.autobuild` | `false` | Auto-build if executable missing |

### Maven Profiles

| Profile | Purpose |
|---------|---------|
| `native-tests` | Run only native C++ tests |
| `native-tests-cuda` | Run native tests with CUDA backend |
| `all-tests` | Run all tests including native |
| `skip-native-tests` | Disable native test execution |

## Consequences

### Advantages

* **Unified test execution**: Single `mvn test` command runs both Java and C++ tests
* **Consistent reporting**: All test results appear in Surefire reports
* **CI/CD simplification**: One test job instead of two
* **Better developer experience**: IDE integration shows native tests in test runners
* **Flexible filtering**: Can run specific native test suites via Maven properties
* **Preserved compatibility**: Existing GTest infrastructure unchanged; can still run tests directly
* **Environment consistency**: Native test environment managed alongside Java tests

### Disadvantages

* **Indirection layer**: Native tests run through Java, adding slight overhead
* **Build dependency**: Requires pre-built native test executable (unless auto-build enabled)
* **Error translation**: Native errors must be translated to Java assertions
* **Timeout handling**: Long-running native tests may hit JUnit timeouts

## Implementation Details

### File Structure

```
platform-tests/
├── src/test/java/org/eclipse/deeplearning4j/nd4j/libnd4j/
│   ├── package-info.java
│   ├── Libnd4jTestHelper.java
│   ├── GTestResultParser.java
│   ├── LibNd4jNativeTestRunner.java
│   ├── LibNd4jCategoryTests.java
│   └── NativeBuildHelper.java
├── run-all-tests.sh
└── pom.xml (updated with properties and profiles)
```

### Test Categories

Native tests are organized into categories for selective execution:

| Category | Test Suites |
|----------|-------------|
| Declarable Operations | DeclarableOpsTests1-19 |
| Arrays | NDArrayTest, NDArrayTest2, ArrayOptionsTests, MultiDataTypeTests |
| Convolutions | ConvolutionTests1, ConvolutionTests2 |
| Attention & NLP | AttentionTests, NlpTests |
| Helpers | HelpersTests1-2, ShapeTests, ShapeUtilsTests |
| Memory | WorkspaceTests, MemoryUtilsTests, DataBufferTests |
| Graphs | GraphTests, FlatBuffersTest |
| Broadcasts | BroadcastableOpsTests, BroadcastMultiDimTest |
| Legacy | LegacyOpsTests, ParityOpsTests, BooleanOpsTests |
| RNG | RNGTests, GraphRandomGeneratorTests |
| Backprop | BackpropTests |

### Usage Examples

```bash
# Run all tests (Java + Native)
mvn test

# Run only native tests
mvn test -Pnative-tests

# Run specific native test suite
mvn test -Dtest=LibNd4jNativeTestRunner \
         -Dorg.nd4j.libnd4j.test.filter=ArrayOptionsTests

# Run native tests for CUDA
mvn test -Pnative-tests-cuda

# Run categorized tests
mvn test -Dtest=LibNd4jCategoryTests

# Disable native tests
mvn test -Pskip-native-tests

# Use shell script for combined execution
./run-all-tests.sh --parallel
```

### Prerequisites

Before running native tests, the test executable must be built:

```bash
# Option 1: Using build script
cd libnd4j && ./buildnativeoperations.sh -t

# Option 2: Using CMake directly
cd libnd4j
cmake -B blasbuild/cpu -S . -DBUILD_TESTS=ON -DSD_CPU=ON
cmake --build blasbuild/cpu --target runtests -j$(nproc)
```

### Error Handling

1. **Missing executable**: Tests are skipped with informative message
2. **Test timeout**: Process killed and failure reported with partial output
3. **Test failures**: GTest XML parsed for detailed failure messages
4. **Segmentation faults**: Captured as test failures with exit code

### Future Considerations

1. **Parallel native test execution**: Could spawn multiple processes for different test suites
2. **Incremental test discovery**: Cache test list to avoid repeated discovery
3. **Native code coverage**: Integration with gcov/llvm-cov for coverage reporting
4. **Memory sanitizer integration**: Enable ASAN/MSAN for native tests via Maven properties

## References

- [JUnit 5 Dynamic Tests](https://junit.org/junit5/docs/current/user-guide/#writing-tests-dynamic-tests)
- [Google Test Documentation](https://google.github.io/googletest/)
- [Maven Surefire Plugin](https://maven.apache.org/surefire/maven-surefire-plugin/)
- [ADR 0006 - Test Architecture](0006%20-%20Test%20architecture.md) - Related JUnit 5 tag usage

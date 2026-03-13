# ADR: Libnd4j Native Test Integration

## Status

Accepted (March 13, 2026)

Proposed by: Adam Gibson

## Context

Libnd4j has a comprehensive Google Test suite in `libnd4j/tests_cpu/` that tests C++ operations directly. However, the test executable was never built as part of the standard Maven build — the `SD_BUILD_TESTS` CMake option existed in `Options.cmake` and was wired through `buildnativeoperations.sh`, but `CMakeLists.txt` never conditionally included the `tests_cpu` subdirectory.

On the Java side, `platform-tests` already has a complete integration framework:
- `LibNd4jNativeTestRunner` — JUnit runner that executes the native GTest binary
- `LibNd4jCategoryTests` — Categorized test suites (smoke, full, specific categories)
- `GTestResultParser` — Parses GTest XML output into JUnit-compatible results
- `NativeBuildHelper` — Locates the test executable in the build tree
- Maven profiles (`native-tests`, `native-tests-cuda`, `all-tests`, `skip-native-tests`)

The gap was purely in the CMake build: the test target was never compiled.

## Decision

### 1. CMakeLists.txt — Conditional test subdirectory

Add a guarded `add_subdirectory(tests_cpu)` block in `libnd4j/CMakeLists.txt` that activates when `SD_BUILD_TESTS=ON`. The `option()` definition already exists in `cmake/Options.cmake`, so we only add the conditional inclusion and `enable_testing()`.

### 2. Maven property — `libnd4j.tests`

Define `<libnd4j.tests></libnd4j.tests>` in `libnd4j/pom.xml` properties. This defaults to empty (tests not built). When set to `--tests` via `-Dlibnd4j.tests=--tests`, the build script's `-t/--tests` flag sets `SD_BUILD_TESTS=ON` in CMake.

The property was already passed as an argument to `buildnativeoperations.sh` in three exec-maven-plugin configurations (CPU, CUDA, generic) but was never defined, so it resolved to an empty string.

### 3. End-to-end flow

```
mvn -Dlibnd4j.tests=--tests ...
  → buildnativeoperations.sh receives --tests flag
  → sets SD_BUILD_TESTS=ON in CMake invocation
  → CMakeLists.txt: add_subdirectory(tests_cpu)
  → GTest downloads, builds, links against libnd4j
  → runtests executable produced in blasbuild/{cpu,cuda}/tests_cpu/layers_tests/

cd platform-tests && mvn test -Dtest=LibNd4jNativeTestRunner
  → NativeBuildHelper locates runtests executable
  → Runs with --gtest_output=xml, optional --gtest_filter
  → GTestResultParser converts XML to JUnit results
```

## Consequences

- Native tests are opt-in (no build time impact by default)
- CI can enable via `-Dlibnd4j.tests=--tests` in the build step
- Developers can run native tests through the same `mvn test` workflow as Java tests
- GTest results appear in surefire reports alongside Java test results

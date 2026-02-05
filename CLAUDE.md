# CLAUDE.md - Development Guide for Deeplearning4j

## Build Commands

**Ask the user for a build command if one isn't provided.** The user is often working on something specific and the build target varies.

Common build patterns:

- **C++ + CUDA rebuild:**
  ```bash
  /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=4 \
    -pl libnd4j,:nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-api \
    -Dlibnd4j.log=libnd4j-build.log clean install -DskipTests
  ```

- **Java-only module install (no native compile):**
  ```bash
  mvn install -DskipTests -pl <module>
  ```

- **Selecting a specific backend:** Use `-Dbackend.artifactId=` (e.g., `-Dbackend.artifactId=nd4j-cuda-12.9` or `-Dbackend.artifactId=nd4j-native`).

- Always `install`, never just `compile` -- downstream modules need the jar in the local repo.

- If building C++, always rebuild CUDA bindings too.

## Testing

**ALWAYS run tests from `platform-tests` ONLY:**
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && mvn test -Dtest=<TestClass>#<method>
```

- **NEVER** run `mvn test` from the project root -- it triggers full rebuilds of native code and runs everything.
- Tests run once. Use surefire logs for debugging: `platform-tests/target/surefire-reports/<TestClass>-output.txt`
- Never pipe test output through `tail` -- always capture full output to a file.

## Development Rules

1. **No workarounds** -- fix root causes, not symptoms.
2. **Trace values to roots** -- always search for the origin of a value before attempting a fix.
3. **No `.arr` or `.shape` in model import code** -- always use `sd.shape(..)` and `sd.rank(..)`. Everything must be variable-based for dynamic shape support.
4. **No fully qualified class names in code** -- use imports.
5. **Never disable verbose/debug as a workaround** -- fix the underlying issue.
6. **`MALLOC_CHECK_=3` does NOT work reliably** -- don't rely on it.
7. **Compute-sanitizer** via `-Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer` for CUDA memory debugging.

## CUDA-Specific Notes

- Heap corruption is often from buffer overruns in native ops, not double-frees. The glibc `(!prev)` message means corrupted malloc metadata from a prior write.
- Views from `.get()` / `.getRow()` on CUDA may have stale device buffers. Use `.dup()` after view operations when the result will be used outside the current SameDiff execution scope.
- `Nd4j.argMax()` has issues with views/non-contiguous arrays. Manual iteration may be needed.
- `setPrimaryBuffer` / `setSpecialBuffer` must keep allocation sizes in sync -- mismatched sizes cause overruns during sync.

## ONNX Import

- ONNX Gather with 2D constant indices `[[0]]` produces higher-rank output than expected. Squeeze single-element constant indices.
- ONNX Softmax opset 13+ defaults axis to -1. The libnd4j softmax op normalizes negative dimensions.
- Mixed-type ops (FLOAT + LONG) silently truncate. Cast explicitly.
- Attention masks must be FLOAT, not LONG, to work with FLOAT attention scores.

## Project Structure

- `libnd4j/` -- C++ native library (CPU and CUDA kernels)
- `nd4j/` -- Java ND4J API, backends, SameDiff
- `nd4j/samediff-import/samediff-import-onnx/` -- ONNX model import
- `deeplearning4j/` -- High-level DL4J layers and model import (Keras etc.)
- `platform-tests/` -- All tests run here
- `codegen/op-codegen/` -- Op code generation (run `./generate.sh all` after changes)

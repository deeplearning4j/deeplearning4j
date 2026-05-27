# PR12: ND4J Java API & Core Infrastructure

**Estimated files:** ~380
**Merge layer:** 4
**Complexity:** High — core API surface
**Reviewers:** Java API team

## Description

The ND4J Java API surface: INDArray, NDArray, DataBuffer, Nd4j factory,
OpExecutioner, Environment, ND4JSystemProperties, memory management,
workspace, shape utilities, device management, and all supporting infrastructure
that is NOT covered by other Java PRs (ops, backends, SameDiff, DSP).

This is the "everything else" in nd4j-api that doesn't fit neatly into
another category.

## Key Sub-Areas

### Factory & core API (~33)
- `Nd4j.java`, `BaseNd4jFactory.java`, `NDArrayFactory.java`
- `INDArray.java`, `BaseNDArray.java`
- `DataBuffer.java`, `DataType.java`
- `Environment.java`, `ND4JEnvironment.java`
- `ND4JSystemProperties.java`

### Memory & workspace (~29+7)
- Memory manager interfaces and implementations
- Workspace interfaces and implementations
- DeallocatorService, MemoryTracker
- `DeviceMemoryManager.java`, `StubDeviceDescriptor.java`, etc.

### Op executioner infrastructure (~46)
- `OpExecutioner.java`, `OpContext.java`, `OpExecutionerUtil.java`
- `DefaultOpExecutioner.java`
- Op type enums, metadata

### Shape & indexing
- Shape utilities, ShapeDescriptor, TAD
- NDIndex, indexing functions

### Activations
- **Note:** Activation implementations (`ActivationCube` through `ActivationThresholdedReLU`)
  are assigned to **PR13** (Java Op Definitions), not PR12, to avoid overlap.

### Presets/codegen (~17)
- `nd4j-cuda-preset/`, `nd4j-native-preset/`, `nd4j-minimizer-preset/`
- `nd4j-hexagon-preset/`, `nd4j-tpu-preset/`, `nd4j-sdx-preset/`
- `tokenizers-native-preset/`

### Other API surface
- `linalg/factory/` — Nd4j factory methods, BLAS wrappers
- `linalg/api/buffer/` — buffer interfaces
- `linalg/api/ndarray/` — NDArray utilities
- `linalg/api/shape/` — shape functions
- `nativeblas/` — native BLAS bindings
- `context/` — thread context
- `profiler/` — profiling interfaces
- `util/` — various utilities

### Java Environment & properties (~23)
- `nd4j/.../linalg/factory/Environment.java`
- `nd4j/.../linalg/factory/ND4JSystemProperties.java`
- Backend-specific environment classes

### Resources/config (~25)
- `nd4j-op-def.pbtxt`
- `onnx-mapping-ruleset.pbtxt`
- `tensorflow-mapping-ruleset.pbtxt`
- Native-image configuration files
- Backend `.properties` files
- OmniHub/pipeline service-loader registrations

### ADRs (0)
No ADRs in this PR are changed in the diff. ADRs 0008 and 0016 exist on master unchanged.

## Review Focus

- INDArray/DataBuffer API changes affect all downstream code
- Environment/ND4JSystemProperties — new properties must match C++ side
- Memory management changes — verify no leak paths
- Preset changes — must match NativeOps.h signature

# Device Pinning Plan

## Goals
- Allow arrays to be pinned to a specific device (GPU) so routing and transfers
  respect that placement and fail fast on conflicts.
- Expose a simple API to pin/unpin arrays and query pinned devices.
- Ensure routing logic uses pinned devices when determining execution targets.

## Steps
1. Extend the hybrid buffer contract with pinning metadata and helpers
   (pinned device, pin/unpin, effective device).
2. Implement pinning in the CUDA buffer layer with enforcement on
   transfers and device access.
3. Update routing and transfer code paths to honor pinned devices and
   surface conflicts early.
4. Add user-facing API helpers to pin/unpin and query pinned status.

## Files to Modify
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/buffer/HybridDataBuffer.java
- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/OpExecutionDelegator.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultMultiBackendExecutioner.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultBackendRoutingStrategy.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/routing/DataLocalityPolicy.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/routing/PerformancePolicy.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/BackendManager.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/DeviceAwareNd4j.java
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/DeviceAwareNDArrayFactory.java

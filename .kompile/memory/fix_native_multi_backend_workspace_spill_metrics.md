---
name: fix_native_multi_backend_workspace_spill_metrics
description: Root-caused and patched NativeMultiBackendWorkspace CPU spill metric failure in CUDA build
type: project
---

MultiBackendWorkspaceIntegrationTest#testSpillMetrics failed because NativeMultiBackendWorkspace created with DEVICE_TYPE_CPU in a CUDA backend used the CUDA Workspace implementation. That implementation treats primary storage as device memory and secondary as host memory. CPU allocations use MemoryType::HOST, so a 1024-byte allocation against an initial size of 256 spilled into _spillsSizeSecondary, while MultiBackendWorkspace::getTotalAllocatedSize() only summed Workspace::getAllocatedSize() (primary current + primary spills). Result was exactly 256.

Patch: libnd4j/include/memory/cpu/MultiBackendWorkspace.cpp now constructs CPU workspaces as host/secondary-backed when compiled for CUDA, expands based on secondary size for CPU descriptors, and includes secondary allocated/spilled/offset metrics. libnd4j/include/memory/cuda/Workspace.cu now carries _cycleAllocationsSecondary through scopeIn() so host-side spill demand participates in workspace learning.

Validation was not run because an existing parent platform-tests Maven/Surefire run was still active. Pending command: cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MultiBackendWorkspaceIntegrationTest#testSpillMetrics 2>&1 | tee /tmp/fix-workspace-spill-metrics.log

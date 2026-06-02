---
name: triton_hopper_tablegen_dependency_fix_20260528
description: Committed Triton NVHopperTransforms tablegen dependency build fix
type: project
---

Standard CUDA Maven build failed before compiling DSP candidate code because Triton Hopper transform sources included NVWS generated headers before tablegen generated them: nvidia/include/Dialect/NVWS/IR/Dialect.h.inc missing. The existing libnd4j/cmake/patch_triton.cmake already appends all tablegen dependencies to NVGPUToLLVM and TritonNVIDIAGPUToLLVM. Added the same patch_add_tablegen_deps call for third_party/nvidia/hopper/lib/Transforms/CMakeLists.txt target NVHopperTransforms.

Verification: /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests passed. Commit: a6627af4a8 fix: add Triton Hopper tablegen dependency.

---
name: op-executioner-mean-sum-simple-investigation-2026-05-28
description: Investigated OpExecutionerTests.testMeanSumSimple expected [256] vs [16] in full platform-tests log
type: project
---

Log /tmp/dsp-full-platform-tests-after-training-replay.log shows OpExecutionerTests.testMeanSumSimple line 652 failed with expected [256.0000] but actual [16.0000]. The line constructs expected via Nd4j.ones(1).muli(16), so expected should normally be [16], and actual arr.sum(1,2) is mathematically correct for ones(1,4,4). This points to corruption/mutation/aliasing of the expected scalar or scalar allocation state, not a bad sum result. The failure occurs before OpExecutionerTests.testDistance in the same class.

The background testDistance exception comes from line 159 in an executor task: EuclideanDistance(matrix [400,10], rowVector [10], result [400,1], -1). Current BaseOp/Shape semantics normalize -1 to last axis, so the native CUDA reduce3 path sees dimension 1 and rejects it for the rank-1 y shape [10]. The test does not await the executor, so the exception is logged on pool-2-thread-1 after the method returns.

DirectTadTrie.cpp is already modified by another worker to include shape/stride/order/dtype in TAD cache keys; older logs also show corrupted dimension values such as 262145, consistent with TAD/dimension metadata corruption, but this investigation did not prove the line-652 expected-scalar mutation is caused by that file. Parent Maven run was still active, so no build/test was run.

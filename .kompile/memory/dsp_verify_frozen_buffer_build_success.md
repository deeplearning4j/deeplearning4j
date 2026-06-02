---
name: dsp-verify-frozen-buffer-build-success
description: CUDA native build succeeded after VERIFY frozen-buffer diagnostic fix
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j && /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-verify-frozen-buffer-build.log
**Result:** BUILD SUCCESS. libnd4j SUCCESS [12:01], nd4j-cuda-12.9 SUCCESS [03:29], total 15:31.
**Change under test:** NativeDynamicShapePlan.cpp VERIFY requested-output value dump now avoids arr->buffer()/bufferAsT on frozen device-only buffers and skips host value dumps when primary is absent.
**Next validation:** rerun TestDspMergedSegmentReplay#testVerifyModeNoMismatch from platform-tests with tee.

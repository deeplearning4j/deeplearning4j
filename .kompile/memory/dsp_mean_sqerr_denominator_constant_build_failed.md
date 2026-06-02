---
name: dsp_mean_sqerr_denominator_constant_build_failed
description: CUDA rebuild failed after mean_sqerr denominator repair
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-mean-sqerr-denom-constant-cuda-build.log\n**Result:** FAIL in libnd4j before nd4j-cuda-12.9.\n**Next:** Read tee/native build logs for first compiler error. Current single optimization remains invalid; do not benchmark or move on.

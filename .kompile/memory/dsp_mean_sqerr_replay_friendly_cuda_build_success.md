---
name: dsp_mean_sqerr_replay_friendly_cuda_build_success
description: CUDA build passed after making mean_sqerr loss reductions device-side/replay-friendly.
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-mean-sqerr-replay-friendly-cuda-build.log
**Result:** BUILD SUCCESS, libnd4j and nd4j-cuda-12.9 installed.
**Change:** `mean_sqerr_loss` and `mean_sqerr_loss_grad` now avoid host reads from reduction outputs in weighted mean and mean-by-nonzero branches. They use `DivideNoNan`, pairwise/broadcast operations, and a device-side nonzero-weight count helper instead.
**Why:** DSP CUDA graph replay captured host scalar reads/assignments around loss reductions, causing replayed training loss to become 0.0. User asked to make kernels more replay-friendly before breaking them out of capture.
**Next:** Run `DspTrainingE2ETest` from `platform-tests` with tee; do not benchmark until training blocker is green.

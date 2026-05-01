---
name: build-config-cpu-native-dsp-triton
description: CPU native build command with Triton enabled for CPU DSP work
type: project
---

# Build Config: CPU Native + DSP Triton

Use this CPU build when working on DSP paths that need Triton enabled. This is a separate configuration from the normal CPU build. Do not include platform-tests in the build module list.

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j && \
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcpu \
  -Dlibnd4j.triton=ON \
  -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
  clean install -DskipTests 2>&1 | tee cpu-dsp-triton-build-output.log
```

Rules from CLAUDE.md/AGENTS.md: CPU has full DSP support; use Triton for CPU DSP work when needed; always install; always use tee; never use make directly; use a 60+ minute timeout for native builds.

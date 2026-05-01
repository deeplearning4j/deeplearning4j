---
name: build-config-cuda-native
description: CUDA native build command with Triton, libnd4j, CUDA bindings, build log, and tee output
type: project
---

# Build Config: CUDA Native

Use for CUDA/native changes, including DSP/Triton CUDA work. Do not include platform-tests in the build module list.

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j && \
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda \
  -Dlibnd4j.triton=ON \
  -Dlibnd4j.chip=cuda \
  -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cuda-12.9 \
  clean install -DskipTests 2>&1 | tee cuda-build-output.log
```

Rules from CLAUDE.md/AGENTS.md: always install, never compile only; always use tee; never use make directly; use a 60+ minute timeout for native builds; never change CUDA compute capability; never clear ccache.

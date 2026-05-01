---
name: build-config-cpu-native
description: CPU native build command without Triton for standard CPU backend builds
type: project
---

# Build Config: CPU Native

Use for normal CPU/native backend builds when DSP Triton is not needed. Do not include platform-tests in the build module list.

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j && \
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcpu \
  -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
  clean install -DskipTests 2>&1 | tee cpu-build-output.log
```

Rules from CLAUDE.md/AGENTS.md: always install, never compile only; always use tee; never use make directly; use a 60+ minute timeout for native builds; do not include platform-tests in build -pl lists.

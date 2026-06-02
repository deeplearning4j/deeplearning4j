---
name: dsp-merged-segment-replay-verify-fix-pass
description: "TestDspMergedSegmentReplay#testVerifyModeNoMismatch passed after VERIFY diagnostic no-primary guard"
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestDspMergedSegmentReplay#testVerifyModeNoMismatch -Dbackend.artifactId=nd4j-cuda-12.9 -Dnd4j.dsp.diagnostics=VERIFY,EXECUTE,FALLBACK -Dnd4j.dsp.diagnostics.level=full 2>&1 | tee /tmp/dsp-merged-verify-fix.log
**Result:** PASS. Tests run: 1, Failures: 0, Errors: 0, BUILD SUCCESS, total 58.615s.
**Relevant observation:** VERIFY diagnostics reported device-only output value dumps skipped instead of triggering DataBuffer allocatePrimary on frozen buffers.
**Implication:** The replay audit failure in TestDspMergedSegmentReplay was caused by diagnostic host materialization, not replay execution corruption.

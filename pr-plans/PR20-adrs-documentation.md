# PR20: Cross-Cutting ADRs, Documentation & Cleanup

**Estimated files:** ~50
**Merge layer:** 0 (independent, can merge anytime)
**Complexity:** Low
**Reviewers:** Anyone

## Description

Cross-cutting ADRs (debugging tools, namespace migration, profiling infrastructure),
the new ADR index (`ADRs/README.md`), investigation journals, stray documentation
files at repo root, and editor config files. No runtime code changes.

Most feature-specific ADRs have been assigned to their corresponding feature PRs:
- Build/platform ADRs → PR01
- CUDA macros → PR02
- FlatBuffers/serialization → PR03
- Memory management → PR04
- Helpers/kernel selection → PR06
- Op implementations → PR07
- Platform backends → PR08
- DSP execution → PR09
- Triton → PR10
- DSP runtime SDK → PR11
- Java API → PR12
- SameDiff core/training → PR15
- DSP runtime Java → PR16
- ONNX import → PR17
- GGML import → PR18
- LLM/VLM generation → PR19
- OmniHub → PR21
- Test architecture → PR22

### Consolidation TODO (not yet performed — must be done in this PR):
- Duplicate 0057 (×4) needs renumbering — Mixed Precision stays 0057; MLIR, Workspace, ZLUDA need new numbers
- Duplicate 0073 (×2) needs renumbering — DSP SDK stays 0073; Hexagon needs a new number
- Duplicate 0075 (same as 0056) should be deleted
- 4 stray root-level ADRs (`ADR-CudaGraphReplay.md`, `ADR-DeviceTransferManagement.md`, `ADR-LlamaCppBackend.md`, `ADR-OpTimingTracker.md`) should be moved into ADRs/ with proper numbering
- `ADRs/README.md` already exists on master — update if needed

## Cross-Cutting ADRs (this PR only — 5 actually changed in the diff)

Only 5 of the original 14 cross-cutting ADRs are changed in the diff.
ADRs 0024 (both), 0025, 0026, 0027, 0032, 0036 exist on master unchanged.

### Debugging & Profiling (3)
- `ADRs/0037 - Ppstep integration with recording.md` — Interactive macro debugger with recording/break-on-error
- `ADRs/0049 - AddressSanitizer Memory Leak Detection.md` — ASAN configuration tuned for JNI with ThreadPool fixes
- `ADRs/0050 - Clang Sanitizers for JNI Memory Debugging.md` — CMake SD_SANITIZERS flag for ASAN/MSAN/LSAN

### Memory Lifecycle Tracking (1)
- `ADRs/0051 - NDArray and DataBuffer Lifecycle Tracking for Memory Leak Detection.md` — Two-level tracker with stack traces, flamegraph output, and JNI API

### Namespace Migration (1)
- `ADRs/0038 - Namespace migration to Eclipse.md` — Two-phase Eclipse Foundation namespace migration plan

## Investigation/journal files (~24)

These are scratch/investigation files at the repo root. Most should NOT be
committed to master — review whether to keep, move to a wiki, or delete.

- `CUDA_HEAP_CORRUPTION_INVESTIGATION.md`
- `degenerate-output-investigation.md`
- `DEVELOPMENT_JOURNAL.md`
- `DSP_ARCHITECTURE_OVERVIEW.md`
- `DSP_CPU_BACKEND_READINESS.md`
- `DSP_SELF_CONTAINED_DEPLOYMENT_RFC.md`
- `FRAMEWORK_API_COMPLETE.md`
- `FRAMEWORK_API_SUMMARY.md`
- `FRAMEWORK_SUMMARY_API.md`
- `MEMORY_ANALYSIS.md`
- `optimization-journal.md`
- `PHASE2_HANDOFF.md`
- `pinning_plan.md`
- `PR_SPLIT_PLAN.md`
- `QWEN.md`
- `TRITON_KV_HANDOFF.md`
- `VLM_DEBUG_JOURNAL.md`
- `vlm-performance-optimization.md`
- `session-openclaw.md`
- `old_session.txt`
- `scalar-issues.md`
- `segfault_report.md`
- `test-journal.md`
- `sample-compilation.txt`

## Editor/tool config (~9)
- `.codex`
- `.cursorrules`
- `.gitignore`
- `.windsurfrules`
- `.mcp.json`
- `.qwen/settings.json`
- `opencode.json`
- `package.json`
- `package-lock.json`

## Root-level build scripts (~15)
- `build-cuda.sh`
- `build-scripts/build-cuda-backend*.sh` (11 variants: debug, address-sanitizer, thread-sanitizer, cudnn combos)
- `change-cuda-versions.sh`
- `update-op-registry.sh`, `update-op-registry.bat`
- `troubleshooting/build-scripts-troubleshooting/` (2 files)
- `test-ant.xml`

## Stray artifacts (DELETE — should be .gitignore'd)
- `ibnd4j.buildthreads=4` — in diff
- `lasspath` — in diff
- `as` — UNTRACKED only (not in diff, working directory artifact)
- `instead` — UNTRACKED only (not in diff, working directory artifact)
- `parsed` — UNTRACKED only (not in diff, working directory artifact)

## Project config
- `AGENTS.md`
- `CLAUDE.md`
- `AGENTS.md.kompile-skills-backup`

## Resources (non-module)
- `resources/dsp/manifest.schema.json`
- `resources/src/main/java/org/deeplearning4j/omnihub/OmnihubConfig.java`
- `resources/src/main/java9/module-info.java`

## Review Notes

- Most journal/investigation files should NOT be committed — recommend deleting or moving to wiki
- Stray root artifacts (`ibnd4j.buildthreads=4`, `lasspath`, `as`, `instead`, `parsed`) should be deleted
- `.cursorrules`, `.codex`, `.windsurfrules`, `opencode.json`, `.qwen/` may not belong in the repo

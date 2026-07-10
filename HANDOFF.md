# HANDOFF — DSP correctness campaign + open perf question (2026-07-05)

Working state for whoever picks this up (human or agent). Branch: `ag_new_release_updates_2`.
Read AGENTS.md first; test tiers and protocols are in `docs/DSP_VALIDATION_PLAYBOOK.md`.

---

## 1. What landed (committed, verified, gate-green)

| Commit | What |
|---|---|
| `374a69cd8d` | #53 both roots: audit-driven merged-replay device-actuality (uninit D2H reads, initcheck 18,485→0) + NVRTC/PTX ModeContract `requiresDeterministicCublas`. Encapsulated capture-audit API on `CudaGraphHandle`; merged-capture TLS/handle/clears consolidated into logged helpers. |
| `ff5589b441` | Alias-test JIT_TOL recalibrated 5× tighter ({2e-4,1e-3}) after the PEDANTIC fix — measured residual 6.6e-5 on one deep softmax chain. |
| `400be2bd9f` | #55: cuBLAS handles are THREAD-LOCAL; PEDANTIC never reached pool-thread GEMMs. Fix = atomic deterministic window applied at handle acquisition (`CublasHelper::enter/exit/inDeterministicWindow`). Ships `ND4JSystemProperties.DSP_READBACK_TRACE` instrument. |
| `b3d14d8923` | #54 expr B: cast-cache skip-assign ABA across plan teardowns (global epoch, per-thread lazy guard drop) + permanent recycled-input integrity assertion. |
| `44867e6f1c` | `docs/DSP_VALIDATION_PLAYBOOK.md` (#29). |
| `0c19dd8999` | #54 family root: cached plans kept raw pointers to donor's freed buffers. Fast-path addr-key demotion (segments.cpp) + `CACHED_REUSE_CTX_REBIND` (slotexec.cpp) + constant-free cast-epoch hook (DataBuffer.cu `deleteSpecial` → extern hook in MmulHelper.cu). **New permanent guard:** `DspSlotLifecycleAuditTest#testCachedPlanReuseAfterDonorCloseWithPoison` — 49 params, 11s, converts the whole "flaky batch corruption" class into a deterministic check. |

Last full gate on this stack: **1642/0/0/0** (11-class batch incl. the 49 poison params).
Key insight that cracked #54: the stale pointers were dereferenced on EVERY run — flakiness was only whether the freed block still held lucky-correct content. Poison-fill makes it deterministic.

## 2. UNCOMMITTED working tree (intentional — do not commit yet, do not discard)

```
M libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu   <- WS-N4 trims #2/#4/#5
M libnd4j/include/legacy/cuda/NativeOps_dsp.cu                <- warning comment only (trim #1 reverted)
M nd4j/.../autodiff/samediff/serde/FlatBuffersMapper.java     <- serde opNum fix (REAL bug, see below)
M .mcp.json.kompile-backup                                    <- not mine, leave alone
```

- **Trims (#38):** plan-owned `ownedCrossStreamEvent_`(+deviceId, mirrors `executionCompleteEvent_` lifecycle incl. teardown in `platformFreePlanResources` and device-change recreate); single `cudaGetDevice` in `platformBeginExecution` (`currentDev`); `platformEndExecution` reuses `ctx->deviceId` (`endDev`). µs-scale, value-neutral, poison-test green.
- **Serde fix:** `BooleanNot`(legacy TRANSFORM_BOOL, opNum 7) and `LogicalNot`(CUSTOM) both claim `"boolean_not"` since the `3a167f2d10` rename. Name-resolved `getOpNum` picked the custom class and wrote its HASH into legacy FlatNodes → `"No known transform bool op for op number: 2090978343"` → SDNB round-trip/import broken. Fix: `asFlatNode` uses `node.opNum()` for legacy types (instance-accurate), name lookup only for CUSTOM/LOGIC/control kinds. Without this the VLM benchmark cannot import.
- **Why held:** AGENTS.md requires no-regression benchmark evidence for DSP-path commits, and the benchmark story is open (below). Commit plan once settled: serde fix as its own commit; trims as `perf(dsp): WS-N4 safe trims` with the benchmark table; re-run poison test + one gate before pushing either.
- **Trim #1 is FORBIDDEN as tried:** gating the `NativeOps_dsp.cu:199` pointer-validation loop re-opened the #54 family in 36s (poison test caught it) — the loop's `setSpecialBuffer(nullptr)` recovery heals borrowed-plan stale device pointers every exec. The proper form (still the dominant win, ~10²-10³ sync driver calls/token): sanitize borrowed-plan externals ONCE at plan-cache checkout, then the per-exec loop can be gated. Task #38 has the design constraints.

## 3. THE OPEN QUESTION — decode throughput 39/30 vs the 62-64 band

Measured (250 tok, `--skip-audit`, reproduced): **OPTIMAL 39.29 tok/s, SLOT_BY_SLOT 30.04/29.99** vs Jul-4 band **62.3-64.1**. Correctness PASS. GPU util only ~38% (host/sync-bound signature).

**Exonerated:** hardware (4090 boosts 2655-2775MHz under load, no throttle, correct device, CPU turbo OK); the uncommitted trims (µs-scale).

**Suspects, ranked:**
1. **The band was never legitimate.** It predates `400be2bd9f`; pool-thread gap GEMMs (~33 live matmuls/step) then ran TF32/DEFAULT *because of the bug #55 fixed*. PEDANTIC on tiny GEMMs plausibly costs this much.
   **Discriminator READY (~15 min total):** in `libnd4j/include/helpers/cuda/cublasHelper.cu` make `inDeterministicWindow()` `return false;` → rebuild → ONE SBS run → **REVERT THE LINE** (an aborted attempt at this experiment is why the tree needed a heal build — always revert before anything else).
   - ≈57-60 ⇒ confirmed: 39/30 is the legitimate correct-math baseline. Do NOT un-fix determinism; recovery = capture-coverage work (Part II #14-#16) pulling gap GEMMs into captured graphs.
   - ≈30 ⇒ window innocent → suspect 2.
2. **Fresh import lineage.** The old cached decoder died with the serde bug; the new import may segment/fuse differently. Old lineage is unrecoverable. Probe: diag-enabled run → compare plan `fingerprint=0x…` / segment+launch counts. (`triton: launches=211 hits=1` under SBS — no historical reference exists; record one.)
3. #54 per-exec additions (fast-path addr-key hash, cast-epoch checks) — small, measure last.

**Benchmark gotchas learned:** decoder cache with pre-rename ops ⇒ `--clear-decoder` once; `platform-tests/pathfinder-mythic.pdf` is a symlink to `~/Documents/RPGs/` (intact); never `pkill` a pattern whose text appears in your own wrapper command line (exit 144, twice); ccache has no `base_dir` ⇒ **git worktree builds = full 2h rebuild, never viable**; hand-revert/re-apply edits in-tree instead (git checkout/stash banned).

## 4. Verification arsenal built this session (use these, don't soak)

- **Poison test** `testCachedPlanReuseAfterDonorCloseWithPoison*` — 11s, deterministic, covers the entire cached-plan stale-pointer class. Run after ANY plan-lifecycle/buffer change.
- **Matrix narrowing** `-Daudit.fixture=A,B -Daudit.mode=M1,M2` (comma lists) on all `DspSlotLifecycleAuditTest` methods — order-dependence bisects.
- **Readback trace** `-Dnd4j.dsp.readbackTrace=true` (`ND4JSystemProperties.DSP_READBACK_TRACE`) — src/dst addresses on the frozen zero-copy refresh (directOutputMode only).
- **Recycled-input assertion** — permanent in `testWarmupRecycleInput`.
- The full 11-class gate is **pre-commit hygiene, run ONCE** — never a repro vehicle. Failing full-diag one-offs: check task #56 signatures first.
- Diag levels: `detailed` buffers to ring (silent); `full` echoes but distorts timing/concurrency. The gate's ALL/full tee log doubles as a forensic trail — extract windows by `PARAM_BANNER` line numbers.

## 5. Task board (ledger is authoritative — task #38 and #56 have full detail)

- **#38 (parked, in-tree):** WS-N4 trims — blocked on §3. Then: checkout-time sanitize enabling trim #1; agent inventory in task has file:line for remaining items (#3 cuBLAS set/restore unification with `tl_appliedMode`, #6/#7 composite-replay handle setup — needs-care class).
- **#56 (watch):** `testFp16MultiLayerStability` STUCK + `testConcurrentMixedModes` — each seen once, full-diag only. Signatures + forensics in task. Only act if they recur naturally.
- **#5:** CPU benchmark record — quick, needs quiet box.
- **Part II (#14-#18):** perf ledger (G3) first → PGO → replay-unit auction. *Directly motivated by §3 suspect 1.*
- **Part III (#20-#22):** split-KV flash-decoding flagship + kernel bundle.
- **Part IV (#24-#28):** triton-cpu spike gates everything else.
- **WS-M (#31-#36):** training perf — harness (M9) first.
- **WS-N (#39-#43):** N1 keystone (gate `checkErrorCode`'s unconditional sync); N6 undersync bug.
- **WS-O (#45-#50):** failover — tests (O5) first; O6 asserts folded into any pool work (see #54's free/reuse findings).

## 6. Immediate next actions (in order)

1. Confirm heal build green (`/tmp/n38-heal-build.log`, task `bmn58mry9`) — restores blasbuild coherence after the aborted experiment; `.m2` already matches the tree.
2. Run the §3 discriminator (15 min) when ready to spend it — everything else in the perf story hangs on it.
3. Depending on answer: either record the new legitimate baseline + commit held work (serde first, then trims) + start Part II G3, or continue the bisect to import-lineage.
